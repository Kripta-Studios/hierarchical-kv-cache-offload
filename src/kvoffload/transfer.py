"""Data plane: the three physical KV pools and the engine that copies blocks.

Physical layout: every pool is an array of blocks, and one block holds ALL layers
for `block_size` tokens: [n_layers, 2 (K/V), block_size, n_kv_heads, head_dim].
A block is therefore one contiguous chunk (1.75 MiB for Qwen3-0.6B at 16 tokens),
so a tier transfer is a single large DMA instead of 2*n_layers tiny ones.

* GPU pool   - a torch tensor in VRAM, read/written by the decode loop.
* CPU pool   - a *pinned* (page-locked) torch tensor. Only pinned memory can be the
               target of a truly asynchronous cudaMemcpyAsync; pageable memory forces
               the driver to bounce through its own staging buffer and the copy
               becomes synchronous with respect to the host.
* SSD pool   - a numpy.memmap over a file. Reads/writes are plain memcpy into the
               mapping; the OS pages data in/out. The SSD never talks to the GPU
               directly here (no GPUDirect Storage on Windows), so SSD<->GPU traffic
               always stages through a CPU slot.

Asynchronous mode (`async_=True`):
* All copies run in FIFO order on ONE worker thread and ONE side CUDA stream, so
  copies that touch the same slot are naturally ordered.
* Copy <-> compute hazards are handled with CUDA events, never with a full device
  sync:
    - Before a copy touches a GPU slot it waits (on the copy stream) for an event
      recorded on the compute stream at submit time -> it sees finished writes and
      does not overwrite data the compute stream is still reading.
    - Before compute touches a GPU slot that has a pending copy, `fence_gpu` makes
      the compute stream wait on that copy's event. The host only blocks if the
      worker has not even *issued* the copy yet (that wait is the "stall" we time).
    - Host-side memcpy into/out of a CPU slot (SSD traffic) first synchronizes the
      last GPU copy that used that slot.
Synchronous mode runs each copy inline and blocks until it is complete: the
"no overlap" baseline.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np
import torch

from .manager import Tier


class CudaKVEngine:
    def __init__(self, block_shape: tuple[int, ...], n_gpu: int, n_cpu: int, n_ssd: int,
                 ssd_path: str | None = None, dtype: torch.dtype = torch.bfloat16,
                 async_: bool = True) -> None:
        self.block_shape = block_shape
        self.gpu = torch.zeros((n_gpu, *block_shape), dtype=dtype, device="cuda")
        self.cpu = torch.empty((max(n_cpu, 1), *block_shape), dtype=dtype, pin_memory=True)
        # numpy has no bfloat16: view the same bytes as int16 for host-side memcpy.
        self._cpu_np = self.cpu.view(torch.int16).numpy().reshape(self.cpu.shape[0], -1)
        self.ssd_path = ssd_path
        self.ssd: np.memmap | None = None
        if n_ssd:
            assert ssd_path, "SSD tier needs a file path"
            self.ssd = np.memmap(ssd_path, dtype=np.int16, mode="w+",
                                 shape=(n_ssd, self._cpu_np.shape[1]))
        self.async_ = async_
        self.stream = torch.cuda.Stream()
        self._pool = ThreadPoolExecutor(max_workers=1) if async_ else None
        self._gpu_pending: dict[int, Future] = {}
        self._cpu_last: dict[int, torch.cuda.Event] = {}
        self.stall_s = 0.0          # host time spent waiting for transfers

    @property
    def block_bytes(self) -> int:
        return self.gpu[0].numel() * self.gpu.element_size()

    # ------------------------------------------------------------------ copies
    def copy(self, src: Tier, src_slot: int, dst: Tier, dst_slot: int) -> None:
        after = None
        if Tier.GPU in (src, dst):
            after = torch.cuda.Event()
            after.record(torch.cuda.current_stream())
        if self._pool is not None:
            fut = self._pool.submit(self._do_copy, src, src_slot, dst, dst_slot, after)
        else:
            t0 = time.perf_counter()
            fut = Future()
            fut.set_result(self._do_copy(src, src_slot, dst, dst_slot, after))
            self.stall_s += time.perf_counter() - t0
        if src == Tier.GPU:
            self._gpu_pending[src_slot] = fut
        if dst == Tier.GPU:
            self._gpu_pending[dst_slot] = fut

    def _wait_cpu_slot(self, slot: int) -> None:
        ev = self._cpu_last.pop(slot, None)
        if ev is not None:
            ev.synchronize()

    def _do_copy(self, src: Tier, s: int, dst: Tier, d: int,
                 after: torch.cuda.Event | None) -> torch.cuda.Event | None:
        ev = None
        if (src, dst) in ((Tier.GPU, Tier.CPU), (Tier.CPU, Tier.GPU)):
            with torch.cuda.stream(self.stream):
                self.stream.wait_event(after)
                if src == Tier.GPU:
                    self.cpu[d].copy_(self.gpu[s], non_blocking=True)
                else:
                    self.gpu[d].copy_(self.cpu[s], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self.stream)
            self._cpu_last[d if dst == Tier.CPU else s] = ev
        elif (src, dst) == (Tier.CPU, Tier.SSD):
            self._wait_cpu_slot(s)
            self.ssd[d] = self._cpu_np[s]
        elif (src, dst) == (Tier.SSD, Tier.CPU):
            self._wait_cpu_slot(d)
            self._cpu_np[d] = self.ssd[s]
        else:
            raise ValueError(f"unsupported copy {src.name}->{dst.name}")
        if not self.async_ and ev is not None:
            ev.synchronize()        # blocking baseline: host waits for the DMA
        return ev

    # ------------------------------------------------------------------ fences
    def fence_gpu(self, slots: list[int]) -> None:
        """Make the compute stream wait for pending copies on these GPU slots."""
        stream = torch.cuda.current_stream()
        for s in slots:
            fut = self._gpu_pending.pop(s, None)
            if fut is None:
                continue
            if not fut.done():
                t0 = time.perf_counter()
                ev = fut.result()
                self.stall_s += time.perf_counter() - t0
            else:
                ev = fut.result()
            if ev is not None:
                stream.wait_event(ev)

    def drain(self) -> None:
        """Wait for every queued copy (end of benchmark / before teardown)."""
        if self._pool is not None:
            self._pool.submit(lambda: None).result()
        torch.cuda.synchronize()
        self._gpu_pending.clear()
        self._cpu_last.clear()

    def close(self) -> None:
        self.drain()
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        if self.ssd is not None:
            self.ssd._mmap.close()   # release the mapping so Windows lets us delete the file
            self.ssd = None
            os.remove(self.ssd_path)
        del self.gpu, self.cpu, self._cpu_np
        torch.cuda.empty_cache()
