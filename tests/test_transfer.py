"""Data-plane tests on tiny tensors (a few KB). Skipped without CUDA."""
import os

import pytest
import torch

from kvoffload.hashing import chain_hashes
from kvoffload.manager import BlockManager, Tier

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
SHAPE = (2, 2, 4, 2, 8)  # layers, K/V, block_size, kv_heads, head_dim


@pytest.mark.parametrize("async_", [False, True])
def test_bytes_survive_round_trip_through_all_tiers(tmp_path, async_):
    from kvoffload.transfer import CudaKVEngine
    eng = CudaKVEngine(SHAPE, n_gpu=2, n_cpu=2, n_ssd=8, ssd_path=str(tmp_path / "kv.bin"), async_=async_)
    mgr = BlockManager({Tier.GPU: 2, Tier.CPU: 2, Tier.SSD: 8}, engine=eng, write_through=async_)
    hs = chain_hashes(list(range(4 * 6)), 4)
    truth = {}
    for i, h in enumerate(hs):
        b = mgr.new_block("s")
        eng.fence_gpu([b.slots[Tier.GPU]])
        data = torch.randn(SHAPE, device="cuda").to(torch.bfloat16) + i
        eng.gpu[b.slots[Tier.GPU]].copy_(data)   # "compute" writes the block
        truth[h] = data.cpu()
        mgr.commit(b, h)
        mgr.release([b])
    assert min(mgr.index[hs[0]].slots) == Tier.SSD   # oldest went all the way down
    for h in hs:                                      # bring each back and check bytes
        (b,) = mgr.acquire([h], "s")
        eng.fence_gpu([b.slots[Tier.GPU]])
        assert torch.equal(eng.gpu[b.slots[Tier.GPU]].cpu(), truth[h])
        mgr.release([b])
    eng.close()
    assert not os.path.exists(tmp_path / "kv.bin")
