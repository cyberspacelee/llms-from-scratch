import math

import pytest
import torch

from llms_from_scratch.inference.cost_model import (
    H100_SXM,
    LLAMA3_8B,
    LLAMA3_70B,
    ModelSpec,
    batch_latency,
    decode_intensity,
    decode_step_time,
    kv_bytes_per_token,
    max_batch_size,
    prefill_time,
    step_cost,
)
from llms_from_scratch.transformer.model import GPT, GPTConfig


def test_llama3_parameter_counts():
    # 官方公布 8.03B 与 70.6B
    assert LLAMA3_8B.num_params() == pytest.approx(8.03e9, rel=2e-3)
    assert LLAMA3_70B.num_params() == pytest.approx(70.55e9, rel=2e-3)


def test_param_count_matches_shared_gpt():
    config = GPTConfig(vocab_size=100, d_model=64, n_layers=3, n_heads=4, n_kv_heads=2)
    spec = ModelSpec("tiny", 3, 64, 4, 2, 16, config.d_ff, 100, tie_embeddings=True)
    assert spec.num_params() == GPT(config).num_params(non_embedding=False)


def test_kv_bytes_per_token():
    assert kv_bytes_per_token(LLAMA3_8B) == 128 * 1024  # 2·32·8·128·2 B = 128 KiB
    assert kv_bytes_per_token(LLAMA3_70B) == 320 * 1024
    assert kv_bytes_per_token(LLAMA3_8B, dtype_bytes=1) == 64 * 1024


def test_attention_flops_count_causal_pairs():
    spec = ModelSpec("t", 1, 8, 2, 1, 4, 16, 10)
    # 提示长度 3：查询依次看到 1、2、3 个键，共 6 对
    prefill = step_cost(spec, [(3, 0)])
    decode = step_cost(spec, [(1, 2)])  # 位置 2 的查询看到 3 个键
    linear_prefill = 2 * spec.body_params * 3 + 2 * spec.lm_head_params
    assert prefill.flops - linear_prefill == 4 * 2 * 4 * 6
    assert decode.flops - (2 * spec.body_params + 2 * spec.lm_head_params) == 4 * 2 * 4 * 3


def test_chunked_prefill_costs_same_flops_as_one_shot():
    one = step_cost(LLAMA3_8B, [(1024, 0)])
    a, b = step_cost(LLAMA3_8B, [(512, 0)]), step_cost(LLAMA3_8B, [(512, 512)])
    lm_head = 2 * LLAMA3_8B.lm_head_params
    assert a.flops + b.flops - lm_head == pytest.approx(one.flops)


def test_decode_intensity_grows_with_batch_and_saturates_with_context():
    # 短上下文时 decode 强度约等于批大小（BF16 权重每字节对应 1 FLOP/批元素）
    assert decode_intensity(LLAMA3_8B, 1, 1) == pytest.approx(1.0, rel=0.02)
    assert decode_intensity(LLAMA3_8B, 64, 1) == pytest.approx(64, rel=0.02)
    # 长上下文时 KV 读取随批线性增长，强度饱和在远低于 H100 拐点的位置
    long = [decode_intensity(LLAMA3_8B, b, 8192) for b in (64, 256, 4096)]
    assert long[0] < long[1] < long[2] < 20 < H100_SXM.ridge_point


def test_prefill_compute_bound_decode_memory_bound():
    prefill = step_cost(LLAMA3_8B, [(2048, 0)])
    decode = step_cost(LLAMA3_8B, [(1, 2048)])
    assert prefill.intensity > H100_SXM.ridge_point > decode.intensity
    # 2048 token 的 prefill 在 100% MFU 下约 30 ms
    assert prefill_time(LLAMA3_8B, H100_SXM, 2048) == pytest.approx(0.030, rel=0.01)
    # batch=1 decode 由读一遍参与矩阵乘的权重（约 15 GB，嵌入表只按行查）决定：约 4.5 ms
    assert decode_step_time(LLAMA3_8B, H100_SXM, 1, 1) == pytest.approx(15.01e9 / 3.35e12,
                                                                       rel=0.01)


def test_max_batch_size():
    # (80·0.9 GB − 16.06 GB) / (8192 · 128 KiB) ≈ 52
    assert max_batch_size(LLAMA3_8B, H100_SXM, 8192) == 52
    assert max_batch_size(LLAMA3_8B, H100_SXM, 8192, kv_dtype_bytes=1) == 104
    # 70B 的 BF16 权重放不进一张 H100
    assert max_batch_size(LLAMA3_70B, H100_SXM, 4096) == 0
    assert max_batch_size(LLAMA3_70B, H100_SXM, 4096, n_gpus=4) > 0


def test_batching_trades_latency_for_throughput():
    small = batch_latency(LLAMA3_8B, H100_SXM, 1, 512, 128)
    big = batch_latency(LLAMA3_8B, H100_SXM, 64, 512, 128)
    assert big.throughput > 30 * small.throughput
    assert big.tpot > small.tpot
    assert math.isclose(small.e2e, small.ttft + small.tpot * 127, rel_tol=1e-9)


def test_step_cost_matches_measured_flops_of_tiny_model():
    """用 torch 的 FLOP 计数器核对线性层部分（注意力由 SDPA 计算，计数器不统计其因果掩码）。"""
    from torch.utils.flop_counter import FlopCounterMode

    config = GPTConfig(vocab_size=64, context_length=32, d_model=32, n_layers=2, n_heads=4,
                       n_kv_heads=2)
    model = GPT(config).eval()
    spec = ModelSpec("tiny", 2, 32, 4, 2, 8, config.d_ff, 64, tie_embeddings=True)
    with FlopCounterMode(display=False) as counter, torch.no_grad():
        model(torch.zeros(1, 16, dtype=torch.long))
    counts = counter.get_flop_counts()["Global"]
    mm = sum(v for k, v in counts.items() if "mm" in str(k))
    # 计数器对每个位置都算 LM head；成本模型只对最后一个位置算
    expected = step_cost(spec, [(16, 0)]).flops - (4 * 2 * 4 * 8 * (16 * 17 // 2))
    expected += 2 * spec.lm_head_params * 15
    assert mm == expected
