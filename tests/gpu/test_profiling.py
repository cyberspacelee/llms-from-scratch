import pytest
import torch

from llms_from_scratch.gpu.hardware import H100
from llms_from_scratch.gpu.profiling import (
    attainable_tflops,
    benchmark,
    h100_layer_report,
    profile_block,
    ridge_point,
    transformer_layer_ops,
)


def test_roofline():
    assert ridge_point(989.4, 3.35) == pytest.approx(295.3, rel=1e-3)
    assert attainable_tflops(10, 989.4, 3.35) == pytest.approx(33.5)
    assert attainable_tflops(1000, 989.4, 3.35) == 989.4


def test_benchmark_cpu():
    x = torch.randn(64, 64)
    timing = benchmark(lambda: x @ x, warmup=2, repeats=5)
    assert len(timing.samples) == 5 and timing.min_ms <= timing.median_ms <= timing.max_ms


def test_layer_estimate_classification():
    ops = {op.name: op for op in transformer_layer_ops(8, 2048, 4096, 32, 11008)}
    assert ops["qkv proj"].bound(H100) == "compute"
    assert ops["rmsnorm (attn)"].bound(H100) == "memory"
    assert ops["silu * mul"].bound(H100) == "memory"
    unfused = {op.name for op in transformer_layer_ops(8, 2048, 4096, 32, 11008, flash=False)}
    assert "softmax" in unfused
    rows = h100_layer_report()
    assert all(t > 0 for *_, t in rows)


def test_profile_block_on_cpu():
    prof = profile_block(batch=1, seq=16, d_model=64, n_heads=4)
    names = {e.key for e in prof.key_averages()}
    assert "block_forward" in names and "aten::mm" in names


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU")
def test_benchmark_cuda():
    x = torch.randn(1024, 1024, device="cuda")
    timing = benchmark(lambda: x @ x, device="cuda")
    assert timing.median_ms > 0
