"""沿最后一维进行 LayerNorm/RMSNorm；统计使用 fp32 或 fp64。"""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    """x / sqrt(mean(x²)+eps) * weight；任意前缀 Shape，最后一维 D。

    不减均值、无 bias；只有 D 个参数，计算/临时空间随输入元素数线性增长。
    同一实现用于 Block、QK-Norm 和 MLA latent，但各自参数互不共享。
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = x if x.dtype == torch.float64 else x.float()
        return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps)).to(
            x.dtype
        ) * self.weight


class LayerNorm(nn.Module):
    """(x-mean(x)) / sqrt(var(x)+eps) * weight + bias，共 2D 个参数。

    方差使用总体方差 mean(centered²)，与 torch.nn.LayerNorm 一致。
    Norm 不持有序列状态，训练、prefill、decode 都逐 token 处理最后一维。
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = x if x.dtype == torch.float64 else x.float()
        centered = work - work.mean(-1, keepdim=True)
        return (centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + self.eps)).to(
            x.dtype
        ) * self.weight + self.bias


def make_norm(kind: str, dim: int) -> RMSNorm | LayerNorm:
    return {"rms": RMSNorm, "layer": LayerNorm}[kind](dim)
