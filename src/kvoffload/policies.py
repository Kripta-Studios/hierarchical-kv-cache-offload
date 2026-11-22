"""Per-tier eviction policies.

Each tier (GPU, CPU, SSD) owns one policy object that orders the blocks resident
in that tier. The manager asks `victim(evictable)` when the tier is full.

* LRUPolicy: classic least-recently-used; what vLLM's free-block queue does.
* SchedulerAwarePolicy: uses a hint from the request scheduler - the list of
  sessions that will run next - and evicts the block whose owners are needed
  furthest in the future (Belady's MIN restricted to a lookahead window), falling
  back to LRU. CachedAttention/AttentionStore (USENIX ATC'24) uses the same idea:
  in multi-turn serving the job queue *tells you* the future access order.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .manager import Block


class EvictionPolicy(Protocol):
    def insert(self, block: Block) -> None: ...
    def touch(self, block: Block) -> None: ...
    def remove(self, block: Block) -> None: ...
    def victim(self, evictable: Callable[[Block], bool]) -> Block | None: ...
    def set_schedule(self, upcoming: Sequence[str]) -> None: ...


class LRUPolicy:
    def __init__(self) -> None:
        # Insertion-ordered: front = least recently used.
        self._order: OrderedDict[int, Block] = OrderedDict()

    def insert(self, block: Block) -> None:
        self._order[id(block)] = block
        self._order.move_to_end(id(block))

    def touch(self, block: Block) -> None:
        if id(block) in self._order:
            self._order.move_to_end(id(block))

    def remove(self, block: Block) -> None:
        self._order.pop(id(block), None)

    def victim(self, evictable: Callable[[Block], bool]) -> Block | None:
        for b in self._order.values():
            if evictable(b):
                return b
        return None

    def set_schedule(self, upcoming: Sequence[str]) -> None:
        pass  # LRU ignores the future

    def __len__(self) -> int:
        return len(self._order)


class SchedulerAwarePolicy(LRUPolicy):
    def __init__(self) -> None:
        super().__init__()
        self._next_use: dict[str, int] = {}

    def set_schedule(self, upcoming: Sequence[str]) -> None:
        self._next_use = {}
        for i, sid in enumerate(upcoming):
            self._next_use.setdefault(sid, i)

    def victim(self, evictable: Callable[[Block], bool]) -> Block | None:
        never = len(self._next_use) + 1
        best, best_dist = None, -1
        # Scan in LRU order so ties (e.g. all "not in the window") go to the LRU block.
        for b in self._order.values():
            if not evictable(b):
                continue
            dist = min((self._next_use.get(s, never) for s in b.sessions), default=never)
            if dist > best_dist:
                best, best_dist = b, dist
                if dist == never:
                    break  # cannot do better than "not needed in the window"
        return best


def make_policy(name: str) -> EvictionPolicy:
    return {"lru": LRUPolicy, "sched": SchedulerAwarePolicy}[name]()
