"""Multi-session agent workload + the turn loop that drives the block manager.

An "agent session" is a long conversation: a system prompt shared by every
session (tool schemas, instructions), then turns of [tool output -> model reply].
Between two turns of one session, other sessions run and compete for VRAM, so the
session's KV must either survive in some tier or be recomputed (prefill again).

The same `Runtime` drives both the real Qwen3 decode loop (compute=PagedQwen3)
and a pure-Python simulator (compute=None) used for policy/capacity sweeps.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Protocol

from .hashing import chain_hashes
from .manager import Block, BlockManager, Tier


@dataclass
class AgentWorkload:
    n_sessions: int = 8
    n_turns: int = 6
    system_tokens: int = 256        # shared by all sessions -> prefix sharing
    user_tokens: int = 128          # first user message, private
    tool_tokens: int = 192          # tool output appended each later turn
    decode_tokens: int = 32         # model reply length per turn
    order: str = "random"           # "round_robin" or "random" interleaving
    seed: int = 0
    vocab: int = 32000

    def schedule(self) -> list[str]:
        """Order in which session turns run (each session appears n_turns times)."""
        turns = [f"s{i}" for _ in range(self.n_turns) for i in range(self.n_sessions)]
        if self.order == "random":
            random.Random(self.seed).shuffle(turns)
        return turns

    def turn_tokens(self, sid: str, turn: int) -> list[int]:
        rng = random.Random(f"{self.seed}/{sid}/{turn}")
        if turn == 0:
            sys_rng = random.Random(f"{self.seed}/system")
            system = [sys_rng.randrange(self.vocab) for _ in range(self.system_tokens)]
            return system + [rng.randrange(self.vocab) for _ in range(self.user_tokens)]
        return [rng.randrange(self.vocab) for _ in range(self.tool_tokens)]


class Compute(Protocol):
    def forward(self, table: list[int], start: int, tokens: list[int]) -> int: ...


@dataclass
class Session:
    sid: str
    tokens: list[int] = field(default_factory=list)   # tokens whose KV has been computed
    pending: list[int] = field(default_factory=list)  # generated but not yet fed back
    hashes: list[bytes] = field(default_factory=list)  # chain over full blocks of `tokens`
    turns: int = 0


@dataclass
class TurnRecord:
    sid: str
    context: int
    hit_tokens: int
    recomputed: int
    restore_ms: float
    ttft_ms: float
    step_ms: list[float]


class Runtime:
    def __init__(self, mgr: BlockManager, block_size: int, compute: Compute | None = None,
                 engine=None, prefetch: bool = False, lookahead: int = 4,
                 prefill_chunk: int = 512) -> None:
        self.mgr, self.B, self.compute, self.engine = mgr, block_size, compute, engine
        self.prefetch, self.lookahead, self.chunk = prefetch, lookahead, prefill_chunk
        self.sessions: dict[str, Session] = {}
        self.records: list[TurnRecord] = []
        self._sim_rng = random.Random(1)
        self._committed = 0

    def _sync(self) -> None:
        if self.compute is not None:
            import torch
            torch.cuda.synchronize()

    def _fence(self, blocks: list[Block]) -> None:
        if self.engine is not None:
            self.engine.fence_gpu([b.slots[Tier.GPU] for b in blocks])

    def _forward(self, table: list[Block], start: int, toks: list[int]) -> int:
        if self.compute is None:
            return self._sim_rng.randrange(32000)
        return self.compute.forward([b.slots[Tier.GPU] for b in table], start, toks)

    def _grow(self, sess: Session, table: list[Block], n_tokens: int) -> None:
        """Make sure `table` has GPU blocks for n_tokens positions."""
        new = []
        while len(table) * self.B < n_tokens:
            b = self.mgr.new_block(sess.sid)
            table.append(b)
            new.append(b)
        self._fence(new)   # the slot may still be the source of an in-flight D2H

    def _commit_full(self, sess: Session, table: list[Block]) -> None:
        """Hash and register every block that became full since the last call."""
        n_full = len(sess.tokens) // self.B
        if n_full > len(sess.hashes):
            parent = sess.hashes[-1] if sess.hashes else None
            sess.hashes += chain_hashes(sess.tokens, self.B, start=len(sess.hashes), parent=parent)
        for i in range(self._committed, n_full):
            # commit() returns the already-registered block on a duplicate; ours
            # then stays private (unhashed) and is freed on release.
            self.mgr.commit(table[i], sess.hashes[i])
        self._committed = max(self._committed, n_full)

    def run_turn(self, sid: str, new_tokens: list[int], decode_tokens: int,
                 upcoming: list[str]) -> TurnRecord:
        sess = self.sessions.setdefault(sid, Session(sid))
        self.mgr.set_schedule(upcoming[: self.lookahead])
        self._sync()
        t0 = time.perf_counter()
        # 1) Restore: hash the whole known sequence (committed KV + new input) and
        #    find the longest cached prefix in any tier. Hashing the new input too
        #    is what lets a fresh session reuse the shared system prompt.
        seq = sess.tokens + sess.pending + new_tokens
        parent = sess.hashes[-1] if sess.hashes else None
        hashes = sess.hashes + chain_hashes(seq, self.B, start=len(sess.hashes), parent=parent)
        n_hit = self.mgr.lookup_prefix(hashes)
        if n_hit * self.B >= len(seq):       # need >= 1 token to prefill for logits
            n_hit -= 1
        table = self.mgr.acquire(hashes[:n_hit], sid)
        n_committed = len(sess.tokens) // self.B
        self.mgr.stats.misses += max(0, n_committed - n_hit)
        self._fence(table)
        # 2) Overlap: while this turn computes, warm up the next session's blocks.
        if self.prefetch and upcoming:
            nxt = self.sessions.get(upcoming[0])
            if nxt is not None and upcoming[0] != sid:
                self.mgr.prefetch(nxt.hashes)
        self._sync()
        t_restore = time.perf_counter()
        # 3) Prefill everything after the cached prefix (lost blocks + new input).
        start = n_hit * self.B
        recomputed = max(0, len(sess.tokens) - start)
        todo = seq[start:]
        sess.tokens, sess.hashes, sess.pending = seq[:start], hashes, []
        self._committed = n_hit
        tok = None
        for i in range(0, len(todo), self.chunk):
            piece = todo[i:i + self.chunk]
            self._grow(sess, table, len(sess.tokens) + len(piece))
            tok = self._forward(table, len(sess.tokens), piece)
            sess.tokens += piece
            self._commit_full(sess, table)
        self._sync()
        t_first = time.perf_counter()
        # 4) Decode: one token per step; the step ends with .item() (a host sync).
        steps = []
        for _ in range(decode_tokens - 1):
            ts = time.perf_counter()
            self._grow(sess, table, len(sess.tokens) + 1)
            nxt_tok = self._forward(table, len(sess.tokens), [tok])
            sess.tokens.append(tok)
            self._commit_full(sess, table)
            tok = nxt_tok
            steps.append((time.perf_counter() - ts) * 1e3)
        sess.pending = [tok]
        self.mgr.release(table)
        sess.turns += 1
        rec = TurnRecord(sid, len(sess.tokens), start, recomputed,
                         (t_restore - t0) * 1e3, (t_first - t0) * 1e3, steps)
        self.records.append(rec)
        return rec

    def run(self, wl: AgentWorkload) -> list[TurnRecord]:
        sched = wl.schedule()
        for i, sid in enumerate(sched):
            turn = self.sessions[sid].turns if sid in self.sessions else 0
            self.run_turn(sid, wl.turn_tokens(sid, turn), wl.decode_tokens, sched[i + 1:])
        return self.records
