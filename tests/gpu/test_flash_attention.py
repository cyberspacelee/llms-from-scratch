import pytest
import torch

from llms_from_scratch.gpu import flash_attention_triton as fat
from llms_from_scratch.gpu.flash_attention import (
    FlashStats,
    flash_attention_backward,
    flash_attention_forward,
    flash_attention_hbm_elements,
    naive_attention,
    standard_attention_hbm_elements,
)


def make(n=3, t=40, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, t, d, generator=g, dtype=torch.float64) for _ in range(3)]


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("blocks", [(16, 16), (8, 16), (16, 8), (7, 5)])
def test_forward_matches_naive(causal, blocks):
    q, k, v = make()
    o, lse = flash_attention_forward(q, k, v, causal, *blocks)
    torch.testing.assert_close(o, naive_attention(q, k, v, causal))
    s = q @ k.transpose(-2, -1) / 4.0
    if causal:
        s = s.masked_fill(torch.ones(40, 40, dtype=torch.bool).triu(1), float("-inf"))
    torch.testing.assert_close(lse, torch.logsumexp(s, -1))


def test_matches_sdpa():
    q, k, v = make()
    ref = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(flash_attention_forward(q, k, v, True)[0], ref)


@pytest.mark.parametrize("causal", [False, True])
def test_backward_matches_autograd(causal):
    q, k, v = (t.requires_grad_() for t in make(t=33))
    o_ref = naive_attention(q, k, v, causal)
    do = torch.randn_like(o_ref)
    o_ref.backward(do)
    with torch.no_grad():
        o, lse = flash_attention_forward(q, k, v, causal, 8, 8)
        grads = flash_attention_backward(q, k, v, o, do, lse, causal, 8, 8)
    for got, ref in zip(grads, (q.grad, k.grad, v.grad)):
        torch.testing.assert_close(got, ref)


def test_causal_skips_upper_tiles():
    q, k, v = make(t=64)
    stats = FlashStats()
    flash_attention_forward(q, k, v, True, 16, 16, stats)
    # 4×4 个块对中，严格在对角线右上方的 6 个被跳过
    assert stats.kv_tile_loads == 10 and stats.skipped_tiles == 6


def test_hbm_traffic_model():
    t, d = 4096, 128
    standard = standard_attention_hbm_elements(t, d)
    flash = flash_attention_hbm_elements(t, d, block_q=128)
    assert standard == 4 * t * d + 4 * t * t
    assert flash == 2 * t * d + 2 * t * d * 32
    # 只按“每个 Q 块重读一遍 K、V”计（不计 L2 命中），d = Br 时约省一半
    assert 1.9 < standard / flash < 2.1


@pytest.mark.skipif(
    not (fat.HAS_TRITON and torch.cuda.is_available()), reason="需要 Triton 与 CUDA GPU"
)
@pytest.mark.parametrize("causal", [False, True])
def test_triton_flash_attention(causal):
    q, k, v = (
        torch.randn(4, 200, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    )
    o = fat.FlashAttentionTriton.apply(q, k, v, causal)
    ref = naive_attention(q.float(), k.float(), v.float(), causal)
    torch.testing.assert_close(o.float(), ref, atol=2e-2, rtol=2e-2)
    o.sum().backward()
    assert q.grad is not None and q.grad.shape == q.shape
