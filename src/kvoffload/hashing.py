"""Prefix-chained block hashing (the idea behind vLLM automatic prefix caching).

A block's KV values depend on *every* token before it (attention mixes the whole
prefix into each layer's hidden state), not only on the block's own tokens. So a
block's identity is hash(parent_block_hash, tokens_in_block). Two requests that
share a system prompt produce identical hashes for the shared full blocks and can
reuse the same physical KV; the first differing token changes every hash after it.

Only *full* blocks are hashed: a partially filled block is still being written,
so it is private to its sequence. Once full, a block is immutable - which is why
copies of it in lower tiers can never go stale (no dirty write-back needed).
"""
from __future__ import annotations

import hashlib
import struct
from collections.abc import Sequence


def block_hash(parent: bytes | None, tokens: Sequence[int], salt: bytes = b"") -> bytes:
    """SHA-256 over (parent hash | salt, token ids). `salt` isolates tenants:
    vLLM injects a per-request cache salt into the first block for the same reason."""
    h = hashlib.sha256()
    h.update(parent if parent is not None else b"root:" + salt)
    h.update(struct.pack(f"<{len(tokens)}I", *tokens))
    return h.digest()


def chain_hashes(tokens: Sequence[int], block_size: int, salt: bytes = b"",
                 start: int = 0, parent: bytes | None = None) -> list[bytes]:
    """Hashes of all full blocks of `tokens`, starting at block index `start`
    (whose parent hash is `parent`). Lets callers extend a chain incrementally."""
    out: list[bytes] = []
    for i in range(start, len(tokens) // block_size):
        parent = block_hash(parent, tokens[i * block_size:(i + 1) * block_size], salt)
        out.append(parent)
    return out
