"""Control-plane tests: pure Python, no CUDA (manager + NullEngine)."""
import pytest

from kvoffload.hashing import block_hash, chain_hashes
from kvoffload.kv_math import LLAMA3_70B, QWEN3_0_6B
from kvoffload.manager import BlockManager, OutOfBlocks, Tier
from kvoffload.policies import LRUPolicy, SchedulerAwarePolicy
from kvoffload.manager import Block
from kvoffload.runtime import AgentWorkload, Runtime


def test_kv_math_worked_examples():
    assert QWEN3_0_6B.bytes_per_token == 2 * 28 * 8 * 128 * 2 == 114_688
    assert LLAMA3_70B.bytes_per_token == 327_680           # 320 KiB / token
    assert LLAMA3_70B.bytes_for(128 * 1024) == 40 * 1024**3  # exactly 40 GiB


def test_chain_hash_prefix_semantics():
    a = chain_hashes(list(range(64)), 16)
    b = chain_hashes(list(range(32)) + [999] + list(range(33, 64)), 16)
    assert len(a) == 4 and a[:2] == b[:2] and a[2] != b[2] and a[3] != b[3]
    # same block content under a different parent -> different identity
    assert block_hash(None, [1, 2]) != block_hash(a[0], [1, 2])
    assert block_hash(None, [1, 2], salt=b"tenant") != block_hash(None, [1, 2])
    # incremental extension equals one-shot hashing
    assert chain_hashes(list(range(64)), 16, start=2, parent=a[1]) == a[2:]
    assert len(chain_hashes(list(range(20)), 16)) == 1    # partial block not hashed


def _fill(mgr, sid, hashes):
    blocks = []
    for h in hashes:
        b = mgr.new_block(sid)
        mgr.commit(b, h)
        blocks.append(b)
    return blocks


def test_eviction_cascades_down_and_promotes_back():
    mgr = BlockManager({Tier.GPU: 2, Tier.CPU: 2, Tier.SSD: 4})
    hs = chain_hashes(list(range(16 * 6)), 16)
    mgr.release(_fill(mgr, "a", hs[:2]))
    mgr.release(_fill(mgr, "b", hs[2:4]))          # a's blocks -> CPU
    mgr.release(_fill(mgr, "c", hs[4:6]))          # a -> SSD, b -> CPU
    tiers = [min(mgr.index[h].slots) for h in hs]
    assert tiers == [Tier.SSD] * 2 + [Tier.CPU] * 2 + [Tier.GPU] * 2
    got = mgr.acquire(hs[:2], "a")                 # SSD -> CPU -> GPU
    assert all(Tier.GPU in b.slots and b.ref == 1 for b in got)
    assert mgr.stats.hits["SSD"] == 2
    assert mgr.stats.copies["SSD->CPU"] == 2 and mgr.stats.copies["CPU->GPU"] == 2


def test_last_tier_victims_are_dropped():
    mgr = BlockManager({Tier.GPU: 1, Tier.CPU: 1})
    hs = chain_hashes(list(range(48)), 16)
    for h in hs:
        mgr.release(_fill(mgr, "x", [h]))
    assert hs[0] not in mgr.index and mgr.stats.dropped == 1
    assert mgr.lookup_prefix(hs) == 0               # chain broken at block 0


def test_refcount_pins_blocks_and_prefix_is_shared():
    mgr = BlockManager({Tier.GPU: 3, Tier.CPU: 4})
    hs = chain_hashes(list(range(32)), 16)
    mgr.release(_fill(mgr, "a", hs))
    b1 = mgr.acquire(hs, "s1")
    b2 = mgr.acquire(hs, "s2")                     # same physical blocks, ref = 2
    assert [id(x) for x in b1] == [id(x) for x in b2] and b1[0].ref == 2
    mgr.new_block("s3")
    with pytest.raises(OutOfBlocks):               # GPU full of pinned blocks
        mgr.new_block("s3")
    mgr.release(b1)
    mgr.release(b2)
    mgr.new_block("s3")                            # now an unpinned block can be evicted


def test_write_through_makes_gpu_eviction_free():
    mgr = BlockManager({Tier.GPU: 1, Tier.CPU: 4}, write_through=True)
    hs = chain_hashes(list(range(32)), 16)
    mgr.release(_fill(mgr, "a", hs[:1]))
    assert mgr.stats.copies["GPU->CPU"] == 1        # copied when it became full
    mgr.release(_fill(mgr, "a", hs[1:2]))           # evicts block 0: no extra copy
    assert mgr.stats.copies["GPU->CPU"] == 2


def test_scheduler_aware_policy_keeps_soon_needed_blocks():
    lru, sched = LRUPolicy(), SchedulerAwarePolicy()
    blocks = [Block(sessions={s}) for s in ("a", "b", "c")]
    for p in (lru, sched):
        for b in blocks:
            p.insert(b)
        p.set_schedule(["a", "c"])
    ok = lambda b: True
    assert lru.victim(ok) is blocks[0]              # oldest, although needed next
    assert sched.victim(ok) is blocks[1]            # "b" is not needed in the window


def test_runtime_simulation_reuses_system_prompt_and_tiers():
    wl = AgentWorkload(n_sessions=4, n_turns=3, system_tokens=64, user_tokens=32,
                       tool_tokens=48, decode_tokens=8, vocab=1000)
    big = Runtime(BlockManager({Tier.GPU: 10_000}), 16)
    big.run(wl)
    assert big.mgr.stats.misses == 0                # infinite GPU: never recompute
    # only the unhashed partial tail block (< block_size tokens) is ever redone
    assert all(r.recomputed < 16 for r in big.records)
    # first turns of later sessions reuse the shared 64-token system prompt
    firsts = {}
    for r in big.records:
        firsts.setdefault(r.sid, r)
    assert sorted(r.hit_tokens for r in firsts.values()) == [0, 64, 64, 64]
    small = Runtime(BlockManager({Tier.GPU: 20}), 16)
    small.run(wl)
    tiered = Runtime(BlockManager({Tier.GPU: 20, Tier.CPU: 40, Tier.SSD: 100}), 16)
    tiered.run(wl)
    assert small.mgr.stats.misses > 0 and tiered.mgr.stats.misses == 0
