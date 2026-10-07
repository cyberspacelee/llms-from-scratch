import pytest
import torch

from llms_from_scratch.gpu.fusion import (
    bias_gelu_dropout_costs,
    bias_gelu_dropout_eager,
    bias_gelu_dropout_fused,
    estimated_time_us,
    graph_break_report,
    inductor_source,
)


def test_cost_model():
    n, d = 8 * 2048 * 4096, 4096
    unfused = bias_gelu_dropout_costs(n, d)
    fused = bias_gelu_dropout_costs(n, d, fused=True)
    assert len(unfused) == 3 and len(fused) == 1
    total_unfused = sum(c.bytes for c in unfused)
    total_fused = sum(c.bytes for c in fused)
    # BF16：未融合 ≈ 6 次读写 × 2 字节 + 1 字节掩码 = 13n；融合 ≈ 2×2 + 1 = 5n
    assert total_unfused == pytest.approx(13 * n, rel=1e-3)
    assert total_fused == pytest.approx(5 * n, rel=1e-3)
    no_mask = bias_gelu_dropout_costs(n, d, fused=True, store_mask=False)
    assert sum(c.bytes for c in no_mask) == pytest.approx(4 * n, rel=1e-3)
    assert estimated_time_us(unfused, 3.35) > 2.5 * estimated_time_us(fused, 3.35)


def test_fused_matches_eager():
    x, b = torch.randn(64, 96), torch.randn(96)
    eager = bias_gelu_dropout_eager(x, b, 0.1, torch.Generator().manual_seed(0))
    fused = bias_gelu_dropout_fused(x, b, 0.1, torch.Generator().manual_seed(0), chunk=1000)
    torch.testing.assert_close(fused, eager)


def test_graph_break_is_reported():
    def f(x):
        y = torch.sin(x)
        print("side effect")  # Python 副作用：Dynamo 无法放进图里
        return torch.cos(y)

    report = graph_break_report(f, torch.randn(8))
    assert report.graph_count == 2 and report.graph_break_count == 1


def test_inductor_fuses_bias_gelu_on_cpu():
    """在 CPU 上真的调用一次 Inductor（约 10 秒）：两个逐元素算子被融合成一个循环。"""
    def f(x, b):
        return torch.nn.functional.gelu(x + b)

    x, b = torch.randn(16, 32), torch.randn(32)
    code = inductor_source(f, x, b)
    assert "cpp_fused_add_gelu" in code
    torch.testing.assert_close(torch.compile(f)(x, b), f(x, b))
