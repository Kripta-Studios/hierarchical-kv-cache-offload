"""Paged Qwen3 forward == Hugging Face greedy decoding. Loads a model: RUN_SLOW=1."""
import pytest
import torch


@pytest.mark.slow
def test_paged_forward_matches_hf_greedy():
    from kvoffload.qwen3_paged import PagedQwen3
    shape = PagedQwen3.block_shape("Qwen/Qwen3-0.6B", 16)
    pool = torch.zeros((8, *shape), dtype=torch.bfloat16, device="cuda")
    m = PagedQwen3(pool, 16)
    prompt = list(range(1000, 1037))
    table = [5, 2, 7, 0, 3]                       # deliberately scattered blocks
    ours, tok = [], m.forward(table, 0, prompt)
    for i in range(12):
        ours.append(tok)
        tok = m.forward(table, len(prompt) + i, [tok])
    ref = m.model.generate(torch.tensor([prompt], device="cuda"), max_new_tokens=12,
                           do_sample=False)[0, len(prompt):].tolist()
    m.close()
    # bf16 + different kernels can flip a near-tie late; demand a long exact prefix
    agree = next((i for i, (a, b) in enumerate(zip(ours, ref)) if a != b), len(ref))
    assert agree >= 8, (ours, ref)
