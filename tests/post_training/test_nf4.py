import torch

from llms_from_scratch.post_training.nf4 import (
    NF4_REFERENCE,
    bits_per_param,
    dequantize_nf4,
    double_dequantize,
    double_quantize,
    nf4_code,
    pack_nibbles,
    quantize_absmax_int4,
    quantize_nf4,
    unpack_nibbles,
)


def test_code_matches_published_table():
    code = nf4_code()
    assert code.numel() == 16 and code[7] == 0 and code[0] == -1 and code[-1] == 1
    torch.testing.assert_close(code, torch.tensor(NF4_REFERENCE), atol=1e-6, rtol=0)


def test_roundtrip_error_is_bounded_and_beats_uniform_int4():
    torch.manual_seed(0)
    w = torch.randn(256, 64) * 0.02
    q = quantize_nf4(w)
    deq = dequantize_nf4(q)
    # 每块的最大值被精确表示
    blocks = w.reshape(-1, 64)
    idx = blocks.abs().argmax(1)
    torch.testing.assert_close(deq.reshape(-1, 64).gather(1, idx[:, None]),
                               blocks.gather(1, idx[:, None]))
    # 相邻码值最大间距的一半 × absmax 是单个元素误差的上界
    half_gap = (nf4_code().diff().max() / 2).item()
    assert ((deq - w).reshape(-1, 64).abs() <= half_gap * q.absmax[:, None] + 1e-7).all()
    nf4_err = (deq - w).pow(2).mean()
    int4_err = (quantize_absmax_int4(w) - w).pow(2).mean()
    assert nf4_err < int4_err


def test_nibble_packing():
    codes = torch.randint(0, 16, (128,), dtype=torch.uint8)
    packed = pack_nibbles(codes)
    assert packed.numel() == 64
    assert torch.equal(unpack_nibbles(packed), codes)


def test_double_quantization():
    torch.manual_seed(0)
    w = torch.randn(1024 * 64) * 0.02
    q = quantize_nf4(w)
    dq = double_quantize(q.absmax)
    rebuilt = double_dequantize(dq)
    assert (rebuilt - q.absmax).abs().max() / q.absmax.mean() < 0.01
    # 二级量化带来的额外误差远小于 4-bit 量化本身的误差
    err_single = (dequantize_nf4(q) - w).pow(2).mean()
    err_double = (dequantize_nf4(q, rebuilt) - w).pow(2).mean()
    assert err_double < 1.01 * err_single


def test_bits_per_param():
    assert bits_per_param(64) == 4.5
    assert abs(bits_per_param(64, double_quant=True) - 4.127) < 1e-3
