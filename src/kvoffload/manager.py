"""Hierarchical paged KV block manager (metadata / control plane only).

Tiers, fastest to largest:  GPU VRAM  ->  pinned CPU RAM  ->  NVMe SSD (mmap file)

Design choices (and why):
* Paged blocks of `block_size` tokens, addressed by (tier, slot). A sequence is a
  list of blocks, so memory is allocated without fragmentation (PagedAttention).
* Blocks are identified by their prefix-chained hash -> one physical copy per
  distinct prefix, shared across sessions with a reference count (prefix caching).
* Inclusive hierarchy: a block may have copies in several tiers at once. Full
  blocks are immutable, so a copy never goes stale; evicting a block from GPU is
  free if a CPU copy already exists. That is what makes *write-through* (copy to
  CPU as soon as a block fills, asynchronously) attractive: eviction later costs
  nothing and the D2H copy overlapped with compute that was happening anyway.
* Eviction cascades down: GPU victim -> CPU, CPU victim -> SSD, SSD victim -> dropped
  (it will be recomputed by prefill if needed again).
* A block with ref > 0 is in use by a running sequence; its GPU copy is pinned.

The manager never touches tensor data: it calls `engine.copy(...)` and the engine
(see transfer.py) moves bytes, synchronously or asynchronously. A NullEngine lets
the same logic run as a pure-Python cache simulator.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Protocol

from .policies import EvictionPolicy, make_policy


class Tier(IntEnum):
    GPU = 0
    CPU = 1
    SSD = 2


class OutOfBlocks(RuntimeError):
    pass


@dataclass(eq=False)
class Block:
    block_hash: bytes | None = None          # None while the block is still being filled
    slots: dict[Tier, int] = field(default_factory=dict)
    ref: int = 0
    sessions: set[str] = field(default_factory=set)


class Engine(Protocol):
    def copy(self, src: Tier, src_slot: int, dst: Tier, dst_slot: int) -> None: ...


class NullEngine:
    """Moves no data: turns the manager into a cache simulator."""

    def copy(self, src: Tier, src_slot: int, dst: Tier, dst_slot: int) -> None:
        pass


@dataclass
class Stats:
    hits: Counter = field(default_factory=Counter)        # tier where an acquired block was found
    copies: Counter = field(default_factory=Counter)      # "GPU->CPU", "SSD->CPU", ...
    prefetched: int = 0
    dropped: int = 0
    misses: int = 0                                        # blocks that had to be recomputed


class BlockManager:
    def __init__(self, capacity: dict[Tier, int], engine: Engine | None = None,
                 policy: str = "lru", write_through: bool = False) -> None:
        if capacity.get(Tier.SSD, 0) and not capacity.get(Tier.CPU, 0):
            raise ValueError("the SSD tier stages through pinned CPU memory; give CPU capacity > 0")
        self.capacity = {t: capacity.get(t, 0) for t in Tier}
        self.tiers = [t for t in Tier if self.capacity[t] > 0]
        self.free: dict[Tier, list[int]] = {t: list(range(self.capacity[t]))[::-1] for t in Tier}
        self.policy: dict[Tier, EvictionPolicy] = {t: make_policy(policy) for t in Tier}
        self.engine: Engine = engine or NullEngine()
        self.write_through = write_through
        self.index: dict[bytes, Block] = {}
        self.stats = Stats()

    # ------------------------------------------------------------------ lookup
    def lookup_prefix(self, hashes: Sequence[bytes]) -> int:
        """Number of leading blocks of a chain that are cached in *some* tier."""
        n = 0
        for h in hashes:
            if h not in self.index:
                break
            n += 1
        return n

    # -------------------------------------------------------------- allocation
    def _evictable(self, tier: Tier):
        def ok(b: Block) -> bool:
            if b.ref == 0:
                return True
            # A block in use must stay on GPU and must keep at least one copy
            # above the GPU until it has been promoted there.
            return tier != Tier.GPU and Tier.GPU in b.slots
        return ok

    def _alloc_slot(self, tier: Tier) -> int:
        if self.free[tier]:
            return self.free[tier].pop()
        victim = self.policy[tier].victim(self._evictable(tier))
        if victim is None:
            raise OutOfBlocks(f"{tier.name} tier full of pinned blocks")
        return self._demote(victim, tier)

    def _demote(self, block: Block, tier: Tier) -> int:
        """Remove `block` from `tier`, copying it one tier down if that tier lacks a
        copy. Returns the freed slot (the engine orders reuse after the copy)."""
        slot = block.slots.pop(tier)
        self.policy[tier].remove(block)
        lower = Tier(tier + 1) if tier + 1 < len(Tier) else None
        if lower is not None and self.capacity[lower] and lower not in block.slots:
            dst = self._alloc_slot(lower)          # may cascade further down
            self._copy(block, tier, slot, lower, dst)
        if not block.slots:                        # fell off the last tier
            if block.block_hash is not None:
                self.index.pop(block.block_hash, None)
            self.stats.dropped += 1
        return slot

    def _copy(self, block: Block, src: Tier, src_slot: int, dst: Tier, dst_slot: int) -> None:
        self.engine.copy(src, src_slot, dst, dst_slot)
        block.slots[dst] = dst_slot
        self.policy[dst].insert(block)
        self.stats.copies[f"{src.name}->{dst.name}"] += 1

    def _promote(self, block: Block, target: Tier) -> None:
        """Bring a copy of `block` up to `target`, staging SSD reads through CPU."""
        if target in block.slots:
            self.policy[target].touch(block)
            return
        # Temporarily pin it: making room in an upper tier cascades evictions
        # downward, and those must not pick the very block we are lifting.
        block.ref += 1
        try:
            cur = min(block.slots)                 # fastest tier holding a copy
            while cur > target:                    # walk up one tier at a time
                dst = Tier(cur - 1)
                self._copy(block, cur, block.slots[cur], dst, self._alloc_slot(dst))
                cur = dst
        finally:
            block.ref -= 1

    def new_block(self, session: str) -> Block:
        """A fresh, unhashed GPU block that a running sequence will fill."""
        b = Block(ref=1, sessions={session})
        b.slots[Tier.GPU] = self._alloc_slot(Tier.GPU)
        self.policy[Tier.GPU].insert(b)
        return b

    # ---------------------------------------------------------------- lifecycle
    def acquire(self, hashes: Sequence[bytes], session: str) -> list[Block]:
        """Pin the cached prefix `hashes` (all must be present) on the GPU."""
        blocks = [self.index[h] for h in hashes]
        for b in blocks:                           # pin first, so promotions below
            b.ref += 1                             # cannot evict a block we still need
            b.sessions.add(session)
        for b in blocks:
            self.stats.hits[min(b.slots).name] += 1
            self._promote(b, Tier.GPU)
        return blocks

    def commit(self, block: Block, block_hash: bytes) -> Block:
        """Register a block that just became full. Returns the canonical block."""
        existing = self.index.get(block_hash)
        if existing is not None and existing is not block:
            # Same content computed twice (two sessions raced on one prefix).
            # Keep the already-registered copy; ours stays private and is freed
            # on release. A real engine would also remap the block table.
            return existing
        block.block_hash = block_hash
        self.index[block_hash] = block
        if self.write_through and self.capacity[Tier.CPU] and Tier.CPU not in block.slots:
            self._copy(block, Tier.GPU, block.slots[Tier.GPU], Tier.CPU, self._alloc_slot(Tier.CPU))
        return block

    def release(self, blocks: Sequence[Block]) -> None:
        for b in blocks:
            b.ref -= 1
            if b.ref == 0 and b.block_hash is None:  # partial block: nobody can reuse it
                for t, s in b.slots.items():
                    self.policy[t].remove(b)
                    self.free[t].append(s)
                b.slots.clear()

    def prefetch(self, hashes: Sequence[bytes], to_gpu: bool = True) -> int:
        """Warm up a session that is about to run: stage SSD copies into CPU and,
        if allowed, CPU copies into free/evictable GPU slots. Never pins."""
        n = 0
        for h in hashes:
            b = self.index.get(h)
            if b is None:
                break
            target = Tier.GPU if to_gpu else Tier.CPU
            if min(b.slots) <= target:
                continue
            try:
                self._promote(b, target)
            except OutOfBlocks:
                break
            n += 1
        self.stats.prefetched += n
        return n

    def set_schedule(self, upcoming_sessions: Sequence[str]) -> None:
        for p in self.policy.values():
            p.set_schedule(upcoming_sessions)

    def resident(self, tier: Tier) -> int:
        return self.capacity[tier] - len(self.free[tier])
