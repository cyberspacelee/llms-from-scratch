import math

import pytest
import torch

from llms_from_scratch.math.numerics import (
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    FP16,
    FP32,
    TORCH_DTYPES,
    activation_rms,
    bit_fields,
    init_std,
    kahan_sum,
    logsumexp,
    naive_cross_entropy,
    naive_softmax,
    sequential_sum,
    stable_cross_entropy,
    stable_softmax,
)

FORMATS = [FP32, FP16, BF16, FP8_E4M3, FP8_E5M2]


@pytest.mark.parametrize("fmt", FORMATS, ids=lambda f: f.name)
def test_format_constants_match_finfo(fmt):
    info = torch.finfo(TORCH_DTYPES[fmt.name])
    assert fmt.max_normal == info.max
    assert fmt.min_normal == info.tiny
    assert fmt.eps == info.eps
    if fmt.name in ("FP32", "FP16", "BF16"):
        assert fmt.min_subnormal == info.smallest_normal * info.eps


@pytest.mark.parametrize("fmt", [FP8_E4M3, FP8_E5M2], ids=lambda f: f.name)
def test_decode_every_fp8_code(fmt):
    codes = torch.arange(256, dtype=torch.uint8)
    values = codes.view(TORCH_DTYPES[fmt.name]).float()
    for code, value in zip(codes.tolist(), values.tolist()):
        mine = fmt.decode(code)
        if math.isnan(value):
            assert math.isnan(mine)
        else:
            assert mine == value


def test_bit_fields():
    assert bit_fields(1.0, FP32) == ("0", "01111111", "0" * 23)
    assert bit_fields(-2.0, BF16) == ("1", "10000000", "0000000")
    assert bit_fields(0.15625, FP16) == ("0", "01100", "0100000000")  # 1.01₂ × 2⁻³
    assert bit_fields(448.0, FP8_E4M3) == ("0", "1111", "110")


def test_overflow_underflow():
    assert torch.isinf(torch.tensor(300.0, dtype=torch.float16) ** 2)
    assert torch.tensor(1e-8, dtype=torch.float16).item() == 0.0
    assert torch.tensor(1e-8, dtype=torch.bfloat16).item() != 0.0  # BF16 与 FP32 同指数范围
    assert torch.tensor(1 + 2**-9, dtype=torch.bfloat16).item() == 1.0  # 但精度只有 8 位


def test_bf16_accumulation_stalls():
    values = torch.full((10_000,), 0.01)
    assert abs(sequential_sum(values, torch.float32) - 100) < 0.01
    stalled = sequential_sum(values, torch.bfloat16)
    assert stalled < 10  # 部分和停在 4 附近：0.01 小于半个 ulp
    assert abs(kahan_sum(values, torch.bfloat16) - 100) < 2  # 受限于 100 附近的 ulp = 0.5


def test_stable_softmax_and_logsumexp():
    z = torch.tensor([[1000.0, 1001.0, 1002.0]])
    assert torch.isnan(naive_softmax(z)).any()
    expected = torch.softmax(torch.tensor([[0.0, 1.0, 2.0]]), -1)
    torch.testing.assert_close(stable_softmax(z), expected)
    torch.testing.assert_close(logsumexp(z), torch.logsumexp(z, -1))


def test_stable_cross_entropy():
    logits = torch.tensor([[0.0, 200.0], [3.0, 1.0]])
    targets = torch.tensor([0, 0])
    assert torch.isinf(naive_cross_entropy(logits, targets))
    torch.testing.assert_close(
        stable_cross_entropy(logits, targets), torch.nn.functional.cross_entropy(logits, targets)
    )


def test_init_schemes():
    assert math.isclose(init_std(512, 512, "kaiming"), math.sqrt(2 / 512))
    assert math.isclose(init_std(256, 768, "xavier"), math.sqrt(2 / 1024))
    width, depth = 256, 30
    kaiming = activation_rms(depth, width, init_std(width, width, "kaiming"))
    small = activation_rms(depth, width, 0.01)
    big = activation_rms(depth, width, 0.2)
    assert 0.5 < kaiming[-1] < 2  # 方差大致保持
    assert small[-1] < 1e-20  # 每层缩小 ≈ 0.01·sqrt(256/2) ≈ 0.11 倍
    assert big[-1] > 1e3  # 每层放大 ≈ 0.2·sqrt(128) ≈ 2.3 倍
    lecun_tanh = activation_rms(depth, width, init_std(width, width, "lecun"), "tanh")
    assert lecun_tanh[-1] > 0.1
