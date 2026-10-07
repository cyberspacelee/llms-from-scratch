import pytest
import torch

from llms_from_scratch.pytorch.accounting import (
    estimate_memory,
    gpt_forward_flops,
    gpt_param_count,
    linear_flops,
    matmul_flops,
    measure_matmul_flops_per_s,
    measured_flops,
    mfu,
    training_flops,
)
from llms_from_scratch.pytorch.autograd_functions import saved_tensor_bytes
from llms_from_scratch.transformer.model import GPT, GPTConfig


@pytest.mark.parametrize("tie,kv", [(True, 4), (False, 2), (True, 1)])
def test_param_formula_matches_model(tie, kv):
    cfg = GPTConfig(vocab_size=97, context_length=32, d_model=64, n_layers=3, n_heads=4,
                    n_kv_heads=kv, tie_embeddings=tie)
    model = GPT(cfg)
    assert gpt_param_count(cfg)["total"] == sum(p.numel() for p in model.parameters())


def test_matmul_flops_counted_by_torch():
    a, b = torch.randn(8, 16), torch.randn(16, 4)
    assert measured_flops(torch.matmul, a, b) == matmul_flops(8, 16, 4)


def test_linear_backward_is_twice_forward():
    lin = torch.nn.Linear(32, 16, bias=False)
    x = torch.randn(10, 32, requires_grad=True)
    fwd = measured_flops(lin, x)
    y = lin(x)
    bwd = measured_flops(lambda: y.sum().backward())
    expected = linear_flops(10, 32, 16)
    assert fwd == expected["forward"]
    assert bwd == expected["backward_input"] + expected["backward_weight"] == 2 * fwd


def test_gpt_linear_flops_match_flop_counter():
    cfg = GPTConfig(vocab_size=100, context_length=32, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    model = GPT(cfg)
    idx = torch.randint(0, 100, (2, 32))
    analytic = gpt_forward_flops(cfg, batch=2, seq_len=32)
    # CPU 上 scaled_dot_product_attention 走融合内核，FlopCounterMode 不统计它，
    # 因此计数器结果恰好等于“权重矩阵乘”部分
    assert measured_flops(model, idx) == analytic["linear"]
    logits = model(idx)
    assert measured_flops(lambda: logits.sum().backward()) == 2 * analytic["linear"]


def test_attention_flops_formula():
    B, h, T, dh = 2, 4, 16, 8
    q, k, v = (torch.randn(B, h, T, dh) for _ in range(3))
    flops = measured_flops(lambda: (q @ k.transpose(-2, -1)).softmax(-1) @ v)
    cfg = GPTConfig(vocab_size=10, context_length=T, d_model=h * dh, n_layers=1, n_heads=h)
    assert flops == gpt_forward_flops(cfg, B, T)["attention"] == 4 * B * T * T * h * dh


def test_six_n_rule_and_memory():
    assert training_flops(70e9, 15e12) == pytest.approx(6.3e24)
    m = estimate_memory(1_000_000)
    assert m.total == 16_000_000
    mp = estimate_memory(1_000_000, mixed_precision=True)
    assert mp.total == 16_000_000 and mp.params == 2_000_000
    assert estimate_memory(10, optimizer="sgd").total == 80


def test_activation_memory_scales_with_batch():
    cfg = GPTConfig(vocab_size=64, context_length=32, d_model=32, n_layers=2, n_heads=4)
    model = GPT(cfg)
    params = list(model.parameters()) + list(model.buffers())
    sizes = []
    for B in (1, 2, 4):
        idx = torch.randint(0, 64, (B, 32))
        sizes.append(saved_tensor_bytes(model, idx, exclude=params))
    # 激活 = a·B + c：c 是与批大小无关的部分（如因果掩码），其余严格正比于 B
    per_sequence = sizes[1] - sizes[0]
    assert sizes[2] - sizes[1] == 2 * per_sequence
    assert per_sequence > 0.9 * sizes[0]


def test_mfu_and_measurement():
    assert mfu(1000.0, 6e9, 1e13) == pytest.approx(0.6)
    assert measure_matmul_flops_per_s(n=128, repeats=3) > 1e6
