import torch

from llms_from_scratch.transformer.attention import MultiHeadAttention
from llms_from_scratch.transformer.model import apply_rope, rope_frequencies
from llms_from_scratch.transformer.positional import (
    rope_rotate,
    rope_rotate_half,
    sinusoidal_encoding,
)


def test_attention_without_positions_is_permutation_equivariant():
    torch.manual_seed(0)
    mha = MultiHeadAttention(16, 2)
    x = torch.randn(1, 7, 16)
    perm = torch.randperm(7)
    y = mha(x, causal=False)
    torch.testing.assert_close(mha(x[:, perm], causal=False), y[:, perm], atol=1e-6, rtol=1e-5)
    # 加上 RoPE 之后不再等变
    freqs = rope_frequencies(8, 7)
    assert not torch.allclose(mha(x[:, perm], freqs, causal=False), mha(x, freqs, causal=False)[:, perm],
                              atol=1e-4)


def test_sinusoidal_values_and_relative_rotation():
    pe = sinusoidal_encoding(50, 16)
    assert pe.shape == (50, 16)
    torch.testing.assert_close(pe[0, 0::2], torch.zeros(8))
    torch.testing.assert_close(pe[0, 1::2], torch.ones(8))
    # PE[p + k] = R_k · PE[p]：每对 (sin, cos) 按 k·ω_i 旋转，R_k 与 p 无关
    k = 5
    omega = 10000 ** (-torch.arange(0, 16, 2) / 16)
    c, s = torch.cos(k * omega), torch.sin(k * omega)
    for p in (0, 7, 30):
        sin_p, cos_p = pe[p, 0::2], pe[p, 1::2]
        torch.testing.assert_close(pe[p + k, 0::2], sin_p * c + cos_p * s, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(pe[p + k, 1::2], cos_p * c - sin_p * s, atol=1e-5, rtol=1e-5)


def test_rope_real_matches_complex():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 9, 16)
    torch.testing.assert_close(rope_rotate(x, torch.arange(9)), apply_rope(x, rope_frequencies(16, 9)),
                               atol=1e-5, rtol=1e-5)


def test_rope_preserves_norm_and_is_relative():
    torch.manual_seed(0)
    q, k = torch.randn(32), torch.randn(32)

    def score(m, n):
        qm = rope_rotate(q[None], torch.tensor([m]))[0]
        kn = rope_rotate(k[None], torch.tensor([n]))[0]
        return (qm @ kn).item()

    torch.testing.assert_close(rope_rotate(q[None], torch.tensor([123]))[0].norm(), q.norm())
    for m, n in [(5, 2), (40, 3), (7, 7)]:
        for shift in (1, 17, 300):
            assert abs(score(m, n) - score(m + shift, n + shift)) < 1e-3


def test_rotate_half_layout_is_a_permutation():
    torch.manual_seed(0)
    x = torch.randn(4, 8)
    pos = torch.arange(4)
    perm = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])  # 交错 -> 前后两半
    torch.testing.assert_close(rope_rotate_half(x[:, perm], pos), rope_rotate(x, pos)[:, perm])
