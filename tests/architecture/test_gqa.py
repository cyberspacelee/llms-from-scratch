import torch

from llms_from_scratch.architecture.gqa import (
    attention_reference,
    grouped_attention,
    kv_cache_bytes,
    mha_to_gqa,
    sdpa_with_repeat,
)
from llms_from_scratch.transformer.model import CausalSelfAttention, GPTConfig, rope_frequencies


def make(n_kv):
    torch.manual_seed(0)
    cfg = GPTConfig(d_model=32, n_heads=4, n_kv_heads=n_kv)
    return CausalSelfAttention(cfg, 0), rope_frequencies(cfg.head_dim, 10)


def test_gqa_with_full_kv_heads_is_mha():
    attn, freqs = make(4)
    x = torch.randn(2, 10, 32)
    torch.testing.assert_close(attn(x, freqs), attention_reference(attn, x, freqs), atol=1e-5, rtol=1e-5)


def test_mqa_and_gqa_match_reference():
    for n_kv in (1, 2):
        attn, freqs = make(n_kv)
        x = torch.randn(2, 10, 32)
        torch.testing.assert_close(attn(x, freqs), attention_reference(attn, x, freqs),
                                   atol=1e-5, rtol=1e-5)


def test_grouped_attention_without_copy():
    torch.manual_seed(1)
    q = torch.randn(2, 8, 5, 16)
    k, v = torch.randn(2, 2, 7, 16), torch.randn(2, 2, 7, 16)  # 含 2 个缓存位置
    torch.testing.assert_close(grouped_attention(q, k, v), sdpa_with_repeat(q, k, v),
                               atol=1e-5, rtol=1e-5)


def test_mean_pool_is_exact_when_group_heads_agree():
    mha, freqs = make(4)
    with torch.no_grad():  # 让每组两个头的 K/V 投影相同
        for proj in (mha.k_proj, mha.v_proj):
            w = proj.weight.view(2, 2, 8, 32)
            w[:, 1] = w[:, 0]
    gqa = mha_to_gqa(mha, 2)
    x = torch.randn(1, 10, 32)
    torch.testing.assert_close(gqa(x, freqs), mha(x, freqs), atol=1e-5, rtol=1e-5)
    assert gqa.k_proj.weight.shape == (16, 32)


def test_kv_cache_formula():
    # LLaMA-2-70B（80 层、8 个 KV 头、d_h=128），4096 个 token，BF16
    assert kv_cache_bytes(1, 4096, 80, 8, 128) == 1_342_177_280
