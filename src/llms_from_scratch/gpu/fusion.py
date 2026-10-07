"""算子融合：带宽受限算子的字节成本模型，手写融合与 torch.compile 的观察工具。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# region cost_model
@dataclass(frozen=True)
class KernelCost:
    name: str
    bytes_read: int
    bytes_written: int

    @property
    def bytes(self) -> int:
        return self.bytes_read + self.bytes_written


def bias_gelu_dropout_costs(
    n: int, d: int, elem_bytes: int = 2, fused: bool = False, store_mask: bool = True
) -> list[KernelCost]:
    """``dropout(gelu(x + b))`` 的前向 HBM 读写量。x 有 n 个元素，偏置 b 有 d 个。

    未融合：三个 kernel，每个都把整个张量读一遍、写一遍；dropout 另写 1 字节/元素的掩码。
    融合：一个 kernel 读 x 一次、写结果一次；掩码可以不存，反向时用同一随机种子重新生成。
    """
    mask = n if store_mask else 0
    if fused:
        return [KernelCost("fused_bias_gelu_dropout", n * elem_bytes + d * elem_bytes,
                           n * elem_bytes + mask)]
    return [
        KernelCost("add_bias", n * elem_bytes + d * elem_bytes, n * elem_bytes),
        KernelCost("gelu", n * elem_bytes, n * elem_bytes),
        KernelCost("dropout", n * elem_bytes, n * elem_bytes + mask),
    ]


def estimated_time_us(
    costs: list[KernelCost], bandwidth_tb_s: float, launch_us: float = 0.0
) -> float:
    """带宽受限算子的时间下界：总字节 / 带宽 + 每个 kernel 的固定开销。"""
    total = sum(c.bytes for c in costs)
    return total / (bandwidth_tb_s * 1e12) * 1e6 + launch_us * len(costs)


# endregion cost_model


# region eager_vs_fused
def bias_gelu_dropout_eager(
    x: torch.Tensor, b: torch.Tensor, p: float, generator: torch.Generator
) -> torch.Tensor:
    """逐个算子执行：每一步都产生一个完整的中间张量。"""
    y = x + b
    g = F.gelu(y)
    keep = torch.rand(g.shape, generator=generator) >= p
    return g * keep / (1 - p)


def bias_gelu_dropout_fused(
    x: torch.Tensor, b: torch.Tensor, p: float, generator: torch.Generator, chunk: int = 4096
) -> torch.Tensor:
    """“融合 kernel”的 CPU 模拟：按块处理，每块的中间结果只存在于局部变量（寄存器）中。

    为了与 eager 版本逐位一致，先一次性生成同样的随机数；真实 kernel 会在块内用
    Philox 计数器按 (seed, offset) 生成，不需要这张随机数表。
    """
    rand = torch.rand(x.shape, generator=generator).reshape(-1)
    flat = x.reshape(-1)
    out = torch.empty_like(flat)
    d = b.numel()
    for start in range(0, flat.numel(), chunk):
        idx = torch.arange(start, min(start + chunk, flat.numel()))
        v = flat[idx] + b[idx % d]  # 读 x 一次
        v = 0.5 * v * (1 + torch.erf(v / math.sqrt(2)))  # GELU（erf 形式）
        v = torch.where(rand[idx] >= p, v / (1 - p), torch.zeros_like(v))
        out[idx] = v  # 写一次
    return out.reshape(x.shape)


# endregion eager_vs_fused


# region compile_tools
def inductor_source(fn, *args) -> str:
    """用 torch.compile 编译 ``fn`` 并返回 Inductor 生成的 Python 包装代码（含 kernel 源码）。

    CPU 上生成 C++/OpenMP 向量化循环，GPU 上生成 Triton kernel。
    """
    from torch._inductor.utils import run_and_get_code

    torch._dynamo.reset()
    _, sources = run_and_get_code(torch.compile(fn), *args)
    return "\n".join(sources)


def graph_break_report(fn, *args):
    """用 Dynamo 的 explain 统计图的个数与 graph break 原因（不调用 Inductor，很快）。"""
    torch._dynamo.reset()
    return torch._dynamo.explain(fn)(*args)


# endregion compile_tools
