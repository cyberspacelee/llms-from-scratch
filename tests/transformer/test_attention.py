import math

import torch
import torch.nn.functional as F

from llms_from_scratch.transformer.attention import (
    MultiHeadAttention,
    SingleHeadAttention,
    causal_mask,
    merge_heads,
    scaled_dot_product_attention,
    softmax,
    split_heads,
)
from llms_from_scratch.transformer.model import CausalSelfAttention, GPTConfig, rope_frequencies


def test_softmax_stable_and_matches_torch():
    x = torch.tensor([[1000.0, 1001.0, 1002.0], [-1.0, 0.0, 1.0]])
    torch.testing.assert_close(softmax(x), torch.softmax(x, -1))
    assert torch.equal(softmax(torch.full((3,), float("-inf"))), torch.zeros(3))


def test_sdpa_matches_torch_with_causal_mask():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 5, 8) for _ in range(3))
    out, w = scaled_dot_product_attention(q, k, v, causal_mask(5))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 3, 5))
    assert torch.all(w.triu(1) == 0)  # 不看未来


def test_scaling_keeps_score_variance_near_one():
    torch.manual_seed(0)
    d = 256
    q, k = torch.randn(4000, d), torch.randn(4000, d)
    raw = (q * k).sum(-1)
    assert abs(raw.var().item() / d - 1) < 0.1
    assert abs((raw / math.sqrt(d)).var().item() - 1) < 0.1


def test_single_head_is_causal():
    torch.manual_seed(0)
    attn = SingleHeadAttention(16, 8)
    x = torch.randn(1, 6, 16)
    y1, _ = attn(x)
    x2 = x.clone()
    x2[:, 4:] = torch.randn(1, 2, 16)  # 改动未来位置
    y2, _ = attn(x2)
    torch.testing.assert_close(y1[:, :4], y2[:, :4])


def test_split_merge_roundtrip():
    x = torch.randn(2, 5, 12)
    assert split_heads(x, 3).shape == (2, 3, 5, 4)
    torch.testing.assert_close(merge_heads(split_heads(x, 3)), x)


def test_mha_matches_shared_model_attention():
    torch.manual_seed(0)
    config = GPTConfig(d_model=32, n_heads=4, context_length=16)
    ref = CausalSelfAttention(config, layer=0)
    mha = MultiHeadAttention(32, 4)
    mha.load_state_dict(ref.state_dict())
    x = torch.randn(2, 10, 32)
    freqs = rope_frequencies(config.head_dim, 10)
    torch.testing.assert_close(mha(x, freqs), ref(x, freqs), atol=1e-5, rtol=1e-4)


def test_padding_mask_ignores_padded_keys():
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 2)
    x = torch.randn(2, 6, 16)
    lengths = torch.tensor([6, 4])
    y = mha(x, key_padding=lengths, causal=False)
    x2 = x.clone()
    x2[1, 4:] = 100.0  # 填充位置的内容不应影响有效位置
    y2 = mha(x2, key_padding=lengths, causal=False)
    torch.testing.assert_close(y[1, :4], y2[1, :4])
    assert torch.all(mha.last_weights[1, :, :, 4:] == 0)
