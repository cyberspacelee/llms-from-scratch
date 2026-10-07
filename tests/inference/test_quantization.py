import pytest
import torch
import torch.nn.functional as F

from llms_from_scratch.inference.quantization import (
    awq_quantize,
    fake_quantize,
    fp8_dequantize,
    fp8_quantize,
    gptq_quantize,
    output_error,
    quantize,
    quantize_kv,
    rtn_quantize,
    smooth_scales,
    w8a8_matmul,
)


def test_symmetric_int8_example():
    qt = quantize(torch.tensor([[-1.0, 0.5, 0.25, 0.1]]), bits=8)
    assert qt.q.flatten().tolist() == [-127, 64, 32, 13]  # scale = 1/127
    assert qt.scale.item() == pytest.approx(1 / 127)


def test_asymmetric_uses_full_range():
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    qt = quantize(x, bits=2, symmetric=False)
    assert qt.q.flatten().tolist() == [0, 1, 2, 3]
    torch.testing.assert_close(qt.dequantize(), x)


def test_rounding_error_variance_is_delta_squared_over_12():
    torch.manual_seed(0)
    x = torch.rand(1, 200_000) * 2 - 1
    qt = quantize(x, bits=6, granularity="tensor")
    mse = (qt.dequantize() - x).pow(2).mean()
    assert mse.item() == pytest.approx(qt.scale.item() ** 2 / 12, rel=0.02)


def test_finer_granularity_reduces_error_with_outlier_rows():
    torch.manual_seed(0)
    w = torch.randn(64, 256) * torch.logspace(-2, 0, 64)[:, None]  # 各行幅度差 100 倍
    err = {g: (fake_quantize(w, bits=4, granularity=g, group_size=32) - w).norm().item()
           for g in ("tensor", "channel", "group")}
    assert err["group"] < err["channel"] < err["tensor"]


def _correlated_activations(n=512, d=128, rank=32, seed=0):
    """真实激活的协方差集中在少数方向上：低秩信号 + 小噪声。"""
    g = torch.Generator().manual_seed(seed)
    signal = torch.randn(n, rank, generator=g) @ torch.randn(rank, d, generator=g)
    return signal + 0.1 * torch.randn(n, d, generator=g)


def test_gptq_beats_round_to_nearest():
    torch.manual_seed(0)
    x = _correlated_activations()
    w = torch.randn(64, 128) / 128**0.5
    rtn = output_error(x, w, rtn_quantize(w, bits=3, group_size=32))
    gptq = output_error(x, w, gptq_quantize(w, x, bits=3, group_size=32)[0])
    assert gptq < 0.5 * rtn


def test_gptq_quantized_weights_lie_on_grid():
    torch.manual_seed(1)
    x, w = _correlated_activations(d=64, rank=16), torch.randn(16, 64)
    w_hat, scales = gptq_quantize(w, x, bits=4, group_size=16)
    assert scales.shape == (16, 4)
    codes = w_hat.reshape(16, 4, 16) / scales[..., None]
    torch.testing.assert_close(codes, codes.round(), atol=1e-4, rtol=0)
    assert codes.abs().max() <= 7 + 1e-4


def test_awq_protects_salient_channels():
    torch.manual_seed(0)
    x = torch.randn(256, 128)
    x[:, :4] *= 30  # 少数“显著”输入通道
    w = torch.randn(64, 128) / 128**0.5
    rtn = output_error(x, w, rtn_quantize(w, bits=3, group_size=32))
    _, s, awq = awq_quantize(w, x, bits=3, group_size=32)
    assert awq < 0.8 * rtn
    assert s[:4].min() > s[4:].max()  # 显著通道被放大，量化时相对误差更小


def test_smoothquant_is_mathematically_equivalent_and_helps_w8a8():
    torch.manual_seed(0)
    x = torch.randn(64, 128)
    x[:, [3, 70]] *= 50  # 激活离群通道（LLM 中常见，且固定在少数通道上）
    w = torch.randn(96, 128) / 128**0.5
    s = smooth_scales(x.abs().amax(0), w, alpha=0.5)
    x_s, w_s = x / s, w * s  # (X·diag(s)⁻¹)(diag(s)·Wᵀ) = XWᵀ
    torch.testing.assert_close(x_s @ w_s.T, x @ w.T, atol=1e-3, rtol=1e-4)
    ref = x @ w.T
    plain = (w8a8_matmul(x, w) - ref).norm() / ref.norm()
    smooth = (w8a8_matmul(x_s, w_s) - ref).norm() / ref.norm()
    assert smooth < 0.5 * plain


def test_fp8_e4m3_relative_error_bound():
    torch.manual_seed(0)
    x = torch.randn(4096)
    q, scale = fp8_quantize(x)
    assert q.dtype == torch.float8_e4m3fn
    back = fp8_dequantize(q, scale)
    normal = x.abs() / scale > 2.0**-6  # E4M3 的最小正规数
    rel = ((back - x).abs() / x.abs())[normal]
    assert rel.max() <= 2.0**-4 + 1e-6  # 3 位尾数：舍入相对误差不超过 2⁻⁴


def test_int8_kv_cache_barely_changes_attention():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 64, 32) for _ in range(3))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = F.scaled_dot_product_attention(q, quantize_kv(k), quantize_kv(v), is_causal=True)
    assert (out - ref).norm() / ref.norm() < 0.01
    out4 = F.scaled_dot_product_attention(q, quantize_kv(k, 4), quantize_kv(v, 4), is_causal=True)
    assert (out4 - ref).norm() > (out - ref).norm()
