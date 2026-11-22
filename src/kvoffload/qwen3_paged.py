"""A real Qwen3 forward pass that reads/writes K/V directly in the paged GPU pool.

We reuse the Hugging Face modules (projections, RMSNorms, MLP, rotary embedding)
for the weights and math, but own the attention data path: new K/V vectors are
scattered into the block slots of the sequence's block table, and attention
gathers K/V for positions [0, T) from those blocks. That is the contract a paged
attention kernel implements; we do it with index ops + SDPA, which costs an extra
gather copy per layer (PagedAttention avoids it by reading blocks in-kernel).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb


class PagedQwen3:
    def __init__(self, gpu_pool: torch.Tensor, block_size: int,
                 model_id: str = "Qwen/Qwen3-0.6B") -> None:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map="cuda").eval()
        cfg = self.model.config
        self.n_layers, self.n_kv, self.n_q = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.num_attention_heads
        self.head_dim, self.B = cfg.head_dim, block_size
        self.pool = gpu_pool
        expected = (self.n_layers, 2, block_size, self.n_kv, self.head_dim)
        assert tuple(gpu_pool.shape[1:]) == expected, (gpu_pool.shape, expected)

    @staticmethod
    def block_shape(model_id: str, block_size: int) -> tuple[int, ...]:
        from transformers import AutoConfig
        c = AutoConfig.from_pretrained(model_id)
        return (c.num_hidden_layers, 2, block_size, c.num_key_value_heads, c.head_dim)

    @torch.inference_mode()
    def forward(self, table: list[int], start: int, tokens: list[int]) -> int:
        """Process `tokens` at positions [start, start+n) for the sequence whose
        blocks live in GPU slots `table`; return the greedy next token."""
        dev, n, B = self.pool.device, len(tokens), self.B
        total = start + n
        pos = torch.arange(start, total, device=dev)
        table_t = torch.tensor(table[: (total + B - 1) // B], device=dev)
        w_slot, w_off = table_t[pos // B], pos % B           # where new K/V go
        m = self.model.model
        h = m.embed_tokens(torch.tensor([tokens], device=dev))
        cos, sin = m.rotary_emb(h, pos[None])
        mask = None
        if n > 1:   # causal mask over the full context for a prefill chunk
            mask = torch.arange(total, device=dev)[None, :] <= pos[:, None]
        for li, layer in enumerate(m.layers):
            at = layer.self_attn
            x = layer.input_layernorm(h)
            q = at.q_norm(at.q_proj(x).view(1, n, self.n_q, self.head_dim)).transpose(1, 2)
            k = at.k_norm(at.k_proj(x).view(1, n, self.n_kv, self.head_dim)).transpose(1, 2)
            v = at.v_proj(x).view(1, n, self.n_kv, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            # Scatter this chunk's K/V into the paged pool (post-RoPE, like vLLM).
            self.pool[w_slot, li, 0, w_off] = k[0].transpose(0, 1)
            self.pool[w_slot, li, 1, w_off] = v[0].transpose(0, 1)
            # Gather the whole context back: [n_blocks, 2, B, H, D] -> [H, T, D].
            kv = self.pool[table_t, li].transpose(0, 1).reshape(2, -1, self.n_kv, self.head_dim)[:, :total]
            kk, vv = kv[0].transpose(0, 1)[None], kv[1].transpose(0, 1)[None]
            o = F.scaled_dot_product_attention(q, kk, vv, attn_mask=mask, scale=at.scaling, enable_gqa=True)
            h = h + at.o_proj(o.transpose(1, 2).reshape(1, n, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        logits = self.model.lm_head(m.norm(h[:, -1]))
        return int(logits.argmax(-1).item())

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
