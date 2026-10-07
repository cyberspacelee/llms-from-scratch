import torch

from llms_from_scratch.transformer.model import GPT, GPTConfig
from llms_from_scratch.transformer.sampling import (
    apply_repetition_penalty,
    generate,
    kv_cache_bytes,
    sample_next,
    top_k_filter,
    top_p_filter,
)


def test_top_k_keeps_k_largest():
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    kept = torch.isfinite(top_k_filter(logits, 2))
    assert kept.tolist() == [[False, True, False, True, False]]


def test_top_p_keeps_minimal_nucleus():
    probs = torch.tensor([[0.5, 0.05, 0.3, 0.15]])
    logits = probs.log()
    assert torch.isfinite(top_p_filter(logits, 0.75)).tolist() == [[True, False, True, False]]
    assert torch.isfinite(top_p_filter(logits, 0.79)).tolist() == [[True, False, True, False]]
    assert torch.isfinite(top_p_filter(logits, 0.81)).tolist() == [[True, False, True, True]]
    assert torch.isfinite(top_p_filter(logits, 0.01)).sum() == 1  # 至少保留最可能的 token
    renorm = torch.softmax(top_p_filter(logits, 0.75), -1)
    torch.testing.assert_close(renorm[0, [0, 2]], torch.tensor([0.625, 0.375]))


def test_temperature_and_greedy():
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    assert sample_next(logits, temperature=0).item() == 0
    g = torch.Generator().manual_seed(0)
    draws = torch.cat([sample_next(logits, temperature=0.05, generator=g) for _ in range(50)])
    assert torch.all(draws == 0)  # 低温趋近贪心


def test_repetition_penalty_sign_aware():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    out = apply_repetition_penalty(logits, torch.tensor([[0, 1]]), 2.0)
    assert out.tolist() == [[1.0, -4.0, 1.0]]


def test_cached_generation_matches_uncached():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=40, context_length=48, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2))
    prompt = torch.randint(0, 40, (2, 5))
    a = generate(model, prompt, 20, use_cache=True, temperature=0)
    b = generate(model, prompt, 20, use_cache=False, temperature=0)
    assert torch.equal(a, b)
    a = generate(model, prompt, 20, use_cache=True, temperature=1.0, top_k=10,
                 generator=torch.Generator().manual_seed(1))
    b = generate(model, prompt, 20, use_cache=False, temperature=1.0, top_k=10,
                 generator=torch.Generator().manual_seed(1))
    assert torch.equal(a, b)


def test_kv_cache_bytes_formula():
    # LLaMA-2-7B：32 层，32 个 KV 头，d_h = 128，4096 tokens，BF16 → 2 GiB
    assert kv_cache_bytes(32, 32, 128, 4096) == 2 * 1024**3
