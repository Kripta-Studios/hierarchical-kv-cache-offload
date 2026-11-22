"""KV-cache sizing math.

Every decoder layer stores, for every past token, one key vector and one value
vector per *KV* head. So the cache costs

    bytes_per_token = 2 (K and V) * n_layers * n_kv_heads * head_dim * dtype_bytes

and a sequence of T tokens costs T * bytes_per_token. Grouped-query attention
(GQA) shrinks n_kv_heads relative to n_q_heads and is the single biggest lever
on this number; paging/offloading (this repo) attacks the *capacity* side.

Run `python -m kvoffload.kv_math` for the worked examples used in the README.
"""
from __future__ import annotations

from dataclasses import dataclass

GiB = 1024**3


@dataclass(frozen=True)
class ModelShape:
    name: str
    n_layers: int
    n_kv_heads: int
    head_dim: int
    dtype_bytes: int = 2  # bf16 / fp16

    @property
    def bytes_per_token(self) -> int:
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.dtype_bytes

    def bytes_for(self, n_tokens: int) -> int:
        return n_tokens * self.bytes_per_token

    def block_bytes(self, block_size: int) -> int:
        """Bytes of one paged block holding `block_size` tokens for *all* layers.

        Storing all layers of a block contiguously (vLLM did this in 2025 for its
        offloading connector) turns many tiny per-layer copies into one large DMA,
        which is what lets PCIe run near its peak bandwidth.
        """
        return block_size * self.bytes_per_token

    def tokens_that_fit(self, budget_bytes: float) -> int:
        return int(budget_bytes // self.bytes_per_token)


QWEN3_0_6B = ModelShape("Qwen3-0.6B", n_layers=28, n_kv_heads=8, head_dim=128)
QWEN3_1_7B = ModelShape("Qwen3-1.7B", n_layers=28, n_kv_heads=8, head_dim=128)
LLAMA3_70B = ModelShape("Llama-3-70B (GQA, 8 KV heads)", n_layers=80, n_kv_heads=8, head_dim=128)
LLAMA3_70B_MHA = ModelShape("70B without GQA (64 KV heads)", n_layers=80, n_kv_heads=64, head_dim=128)


def transfer_ms(n_bytes: float, gb_per_s: float) -> float:
    """Time to move n_bytes at a sustained bandwidth (GB/s, 1e9 bytes)."""
    return n_bytes / (gb_per_s * 1e9) * 1e3


def main() -> None:
    ctx = 128 * 1024
    for m in (QWEN3_0_6B, LLAMA3_70B, LLAMA3_70B_MHA):
        per_tok = m.bytes_per_token
        print(f"{m.name:34s} {per_tok/1024:8.1f} KiB/token  "
              f"128k ctx = {m.bytes_for(ctx)/GiB:7.1f} GiB  "
              f"16-token block = {m.block_bytes(16)/2**20:6.2f} MiB")
    # How much fits on this laptop for Qwen3-0.6B (weights ~1.2 GB in bf16).
    for tier, budget in (("GPU (12 GB - 1.2 GB weights - 1.5 GB runtime)", 9.3 * GiB),
                         ("pinned CPU (16 GB of 32 GB)", 16 * GiB),
                         ("NVMe SSD (200 GB)", 200 * GiB)):
        print(f"Qwen3-0.6B tokens in {tier:48s}: {QWEN3_0_6B.tokens_that_fit(budget):>12,d}")
    seq = LLAMA3_70B.bytes_for(ctx)
    for link, bw in (("PCIe 4.0 x16 ~25 GB/s", 25), ("PCIe 5.0 x16 ~50 GB/s", 50), ("NVMe ~7 GB/s", 7)):
        print(f"Reload one 128k-token 70B context over {link}: {transfer_ms(seq, bw):8.0f} ms")


if __name__ == "__main__":
    main()
