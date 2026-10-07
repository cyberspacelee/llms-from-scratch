import pytest
import torch

from llms_from_scratch.pretraining.precision import (
    BF16,
    E4M3,
    E5M2,
    FP16,
    FP32,
    DelayedScaler,
    DynamicLossScaler,
    accumulate_updates,
    autocast_dtypes,
    dequantize_blockwise,
    fp8_gemm_blockwise,
    limited_precision_sum,
    quantize_blockwise,
    quantize_fp8,
    relative_error,
    round_to_format,
)


@pytest.mark.parametrize("fmt,dtype,limit", [
    (FP16, torch.float16, 6e4),
    (BF16, torch.bfloat16, 1e30),
    (E4M3, torch.float8_e4m3fn, 440.0),
    (E5M2, torch.float8_e5m2, 5e4),
])
def test_round_to_format_matches_torch(fmt, dtype, limit):
    torch.manual_seed(0)
    x = (torch.randn(4000) * torch.logspace(-8, 0, 4000) * limit).clamp(-limit, limit)
    ours = round_to_format(x, fmt)
    ref = x.to(dtype).float()
    torch.testing.assert_close(ours, ref, atol=0, rtol=0)


def test_format_constants():
    assert FP16.min_subnormal == 2.0**-24
    assert E4M3.eps == 0.125 and E5M2.eps == 0.25
    assert BF16.min_normal == torch.finfo(torch.bfloat16).smallest_normal


def test_fp16_gradients_underflow_without_scaling():
    g = torch.tensor([1e-8, 2e-8])          # 都小于 FP16 最小次正规数 2^-24 ≈ 6e-8 的一半
    assert (g.half() == 0).all()
    scaled = (g * 2**16).half().float() / 2**16          # 先放大再存 FP16，最后在 FP32 中除回
    torch.testing.assert_close(scaled, g, rtol=1e-2, atol=0)


def test_dynamic_loss_scaler_skips_and_backs_off():
    w = torch.nn.Parameter(torch.ones(2))
    opt = torch.optim.SGD([w], lr=1.0)
    scaler = DynamicLossScaler(init_scale=1024.0, growth_interval=3)
    w.grad = torch.tensor([float("inf"), 1.0])
    assert scaler.step(opt, [w]) is False                 # 溢出：跳过更新
    assert torch.equal(w.detach(), torch.ones(2)) and scaler.scale == 512.0
    for _ in range(3):
        w.grad = torch.full((2,), 512.0)                  # 已缩放的梯度，除回后为 1
        assert scaler.step(opt, [w]) is True
    assert scaler.scale == 1024.0                         # 连续 3 步无溢出：增长一次
    torch.testing.assert_close(w.detach(), torch.full((2,), -2.0), rtol=0, atol=1e-6)


def test_blockwise_quantization_isolates_outliers():
    torch.manual_seed(0)
    x = torch.randn(256, 512) * 1e-3
    x[3, 7] = 1000.0                                      # 一个离群值
    normal = torch.ones_like(x, dtype=torch.bool)
    normal[3, 7] = False
    q, s = quantize_fp8(x)
    per_tensor = relative_error((q / s)[normal], x[normal])
    qb, sb = quantize_blockwise(x, (1, 128))
    blockwise = relative_error(dequantize_blockwise(qb, sb, (1, 128))[normal], x[normal])
    assert per_tensor > 0.5                               # 普通值几乎全部下溢成 0
    assert blockwise < 0.05


def test_fp8_gemm_blockwise_is_accurate():
    torch.manual_seed(0)
    a, b = torch.randn(16, 256), torch.randn(128, 256)
    assert relative_error(fp8_gemm_blockwise(a, b), a @ b.T) < 0.05


def test_delayed_scaling_saturates_on_new_outlier():
    scaler = DelayedScaler(history=4)
    for _ in range(4):
        scaler.quantize(torch.randn(1000))                # 历史 amax 约 3
    x = torch.randn(1000)
    x[0] = 100.0
    q, s = scaler.quantize(x)
    assert (q / s)[0] < 10.0                              # 离群值被截断到历史尺度
    assert max(scaler.amax_history) == 100.0              # 下一步才会适应


def test_promoted_accumulation_reduces_error():
    torch.manual_seed(0)
    v = torch.rand(4096) * 0.01
    exact = v.double().sum().item()
    plain = abs(limited_precision_sum(v, man_bits=13) - exact) / exact
    promoted = abs(limited_precision_sum(v, man_bits=13, promote_every=128) - exact) / exact
    assert promoted < plain / 5


def test_bf16_swamps_small_updates_but_fp32_does_not():
    assert accumulate_updates(1.0, 1e-3, 1000, BF16) == 1.0          # 每一步都被舍入回 1
    assert abs(accumulate_updates(1.0, 1e-3, 1000, FP32) - 2.0) < 1e-3


def test_autocast_policy_on_cpu():
    d = autocast_dtypes()
    assert d["matmul"] == torch.bfloat16 and d["linear"] == torch.bfloat16
    assert d["add_fp32_input"] == torch.float32
    assert d["cross_entropy"] == torch.float32
