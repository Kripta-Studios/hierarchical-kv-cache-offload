"""Measure the tier bandwidths of THIS machine.

1. H2D / D2H with pinned vs pageable host memory, for a tiny copy, one KV block
   (1.75 MiB, Qwen3-0.6B, 16 tokens) and a large 64 MiB copy.
2. Copy/compute overlap: a bf16 matmul loop on the compute stream vs. H2D copies
   on a side stream, alone and together.
3. SSD: numpy.memmap write(+flush)/read (goes through the OS page cache) and, on
   Windows, unbuffered direct I/O (FILE_FLAG_NO_BUFFERING) for true device speed.

Run through the machine-wide lock:
  python _tools/gpu_lock.py -- python benchmarks/bench_bandwidth.py
"""
from __future__ import annotations

import ctypes
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kvoffload.kv_math import QWEN3_0_6B  # noqa: E402

BLOCK = QWEN3_0_6B.block_bytes(16)
SCRATCH = ROOT / ".scratch"
SSD_BYTES = 1 << 30                      # 1 GiB test file, deleted afterwards


def gbps(n_bytes: float, seconds: float) -> float:
    return n_bytes / seconds / 1e9


def time_cuda(fn, reps: int) -> float:
    """Seconds per call, measured with CUDA events after a warm-up."""
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / 1e3 / reps


def pcie() -> list[dict]:
    rows = []
    for label, n in (("64 KiB", 64 << 10), ("1 KV block (1.75 MiB)", BLOCK), ("64 MiB", 64 << 20)):
        dev = torch.empty(n, dtype=torch.uint8, device="cuda")
        reps = max(5, min(200, (2 << 30) // n // 4))
        for mem in ("pinned", "pageable"):
            host = torch.empty(n, dtype=torch.uint8, pin_memory=(mem == "pinned"))
            h2d = time_cuda(lambda: dev.copy_(host, non_blocking=True), reps)
            d2h = time_cuda(lambda: host.copy_(dev, non_blocking=True), reps)
            rows.append({"size": label, "bytes": n, "host_memory": mem,
                         "h2d_GBps": round(gbps(n, h2d), 2), "d2h_GBps": round(gbps(n, d2h), 2),
                         "h2d_us": round(h2d * 1e6, 1)})
            del host
        del dev
    return rows


def overlap() -> dict:
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    host = torch.empty(256 << 20, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty_like(host, device="cuda")
    side = torch.cuda.Stream()

    def compute():
        for _ in range(8):
            a @ a

    def copy():
        with torch.cuda.stream(side):
            for _ in range(2):
                dev.copy_(host, non_blocking=True)

    def both():
        side.wait_stream(torch.cuda.current_stream())
        copy()
        compute()
        torch.cuda.current_stream().wait_stream(side)

    # Interleave the three variants and take medians: a laptop GPU's clocks drift
    # with temperature, which would otherwise bias whichever variant runs last.
    samples: dict[str, list[float]] = {"compute_only": [], "copy_only": [], "both": []}
    fns = {"compute_only": compute, "copy_only": copy, "both": both}
    for fn in fns.values():
        fn()
    for _ in range(7):
        for name, fn in fns.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter() - t0) * 1e3)
    t = {k: float(np.median(v)) for k, v in samples.items()}
    serial = t["compute_only"] + t["copy_only"]
    t["hidden_fraction_of_shorter"] = round((serial - t["both"]) / min(t["compute_only"], t["copy_only"]), 3)
    del a, host, dev
    return {k: round(v, 2) for k, v in t.items()}


def ssd_memmap() -> dict:
    path = SCRATCH / "memmap_test.bin"
    n_blocks = SSD_BYTES // BLOCK
    buf = np.random.default_rng(0).integers(0, 255, BLOCK, dtype=np.uint8)
    mm = np.memmap(path, dtype=np.uint8, mode="w+", shape=(n_blocks, BLOCK))
    t0 = time.perf_counter()
    for i in range(n_blocks):
        mm[i] = buf
    mm.flush()                     # FlushViewOfFile: push dirty pages to the device
    t_write = time.perf_counter() - t0
    out = np.empty(BLOCK, dtype=np.uint8)
    t0 = time.perf_counter()
    for i in range(n_blocks):
        out[:] = mm[i]
    t_read = time.perf_counter() - t0
    mm._mmap.close()
    del mm
    os.remove(path)
    n = n_blocks * BLOCK
    return {"bytes": n, "write_flush_GBps": round(gbps(n, t_write), 2),
            "read_GBps_page_cache_warm": round(gbps(n, t_read), 2)}


def ssd_direct_windows() -> dict | None:
    """Unbuffered, write-through sequential I/O in KV-block-sized chunks (QD=1)."""
    if platform.system() != "Windows":
        return None
    from ctypes import wintypes as wt
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wt.HANDLE
    k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.HANDLE]
    k32.VirtualAlloc.restype = ctypes.c_void_p
    k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
    k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]
    for f in (k32.ReadFile, k32.WriteFile):
        f.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
    k32.SetFilePointerEx.argtypes = [wt.HANDLE, ctypes.c_longlong, ctypes.c_void_p, wt.DWORD]
    k32.CloseHandle.argtypes = [wt.HANDLE]
    path = str(SCRATCH / "direct_test.bin")
    flags = 0x20000000 | 0x80000000          # NO_BUFFERING | WRITE_THROUGH
    h = k32.CreateFileW(path, 0xC0000000, 0, None, 2, flags, None)  # CREATE_ALWAYS
    if h in (None, ctypes.c_void_p(-1).value):
        return {"error": f"CreateFileW failed ({ctypes.get_last_error()})"}
    buf = k32.VirtualAlloc(None, BLOCK, 0x3000, 0x04)  # page-aligned, as NO_BUFFERING requires
    ctypes.memset(buf, 7, BLOCK)
    n_blocks, done = SSD_BYTES // BLOCK, wt.DWORD()
    res = {}
    try:
        for name, op in (("write", k32.WriteFile), ("read", k32.ReadFile)):
            k32.SetFilePointerEx(h, 0, None, 0)
            t0 = time.perf_counter()
            for _ in range(n_blocks):
                if not op(h, buf, BLOCK, ctypes.byref(done), None) or done.value != BLOCK:
                    raise OSError(f"{name} failed ({ctypes.get_last_error()})")
            res[f"{name}_GBps"] = round(gbps(n_blocks * BLOCK, time.perf_counter() - t0), 2)
        res["block_read_us"] = round(BLOCK / (res["read_GBps"] * 1e9) * 1e6, 1)
    finally:
        k32.CloseHandle(h)
        k32.VirtualFree(buf, 0, 0x8000)
        os.remove(path)
    return res


def main() -> None:
    SCRATCH.mkdir(exist_ok=True)
    out = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
           "cuda": torch.version.cuda, "os": platform.platform(), "block_bytes": BLOCK,
           "pcie": pcie(), "overlap_ms": overlap(),
           "ssd_memmap": ssd_memmap(), "ssd_direct_io": ssd_direct_windows()}
    torch.cuda.empty_cache()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "bandwidth.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
