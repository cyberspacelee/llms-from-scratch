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
        """根据尺寸和配置创建参数/子层。

        Args:
            dim: 最后一维的归一化宽度 D。
            eps: 归一化分母中的稳定常数。

        Returns:
            None；参数与子层注册在 self 中。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行本模块的前向计算；输入与输出 shape 约定如下。

        Args:
            x: float [...,D]，例如 [B,T,D]。

        Returns:
            normalized tensor，同输入 shape；统计 [...,1]。
        """
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
        """根据尺寸和配置创建参数/子层。

        Args:
            dim: 最后一维的归一化宽度 D。
            eps: 归一化分母中的稳定常数。

        Returns:
            None；参数与子层注册在 self 中。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行本模块的前向计算；输入与输出 shape 约定如下。

        Args:
            x: float [...,D]，例如 [B,T,D]。

        Returns:
            normalized tensor，同输入 shape；mean/variance [...,1]。
        """
        work = x if x.dtype == torch.float64 else x.float()
        centered = work - work.mean(-1, keepdim=True)
        return (centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + self.eps)).to(
            x.dtype
        ) * self.weight + self.bias


def make_norm(kind: str, dim: int) -> RMSNorm | LayerNorm:
    """按名称构造 LayerNorm/RMSNorm。

    Args:
        kind: 组件类型名，必须为该函数支持的实现。
        dim: 最后一维的归一化宽度 D。

    Returns:
        新 LayerNorm 或 RMSNorm 实例。
    """
    return {"rms": RMSNorm, "layer": LayerNorm}[kind](dim)
