"""Absolute PE and adjacent-pair RoPE, all phases evaluated in fp32/fp64."""

from __future__ import annotations

import math

import torch

from ..config import PositionConfig


def rotary_frequencies(
    dim: int,
    config: PositionConfig,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """生成 dim/2 个角频率 omega_i=base^(-2i/dim)。

    linear 全部除 factor；固定 NTK 调整 base；YaRN 按频段混合原频率
    与插值频率。factor 在整次实验中固定，避免 decode 随长度改变频率后
    新 Q 与已缓存 K 使用不同坐标系。参数量为零，不缓存 token 状态。
    """
    if dim < 2 or dim % 2:
        raise ValueError("RoPE needs a positive even dimension")
    base = config.base
    if config.scaling == "ntk":
        if dim <= 2:
            raise ValueError("NTK base scaling needs rotary dim > 2")
        base *= config.factor ** (dim / (dim - 2))
    freq = base ** (-torch.arange(0, dim, 2, device=device, dtype=dtype) / dim)
    if config.scaling == "linear":
        freq = freq / config.factor
    elif config.scaling == "yarn":

        def correction(rotations: float) -> float:
            return (
                dim
                * math.log(config.original_length / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(0, math.floor(correction(config.beta_fast)))
        high = min(dim - 1, math.ceil(correction(config.beta_slow)))
        ramp = (
            (torch.arange(dim // 2, device=device, dtype=dtype) - low) / max(high - low, 0.001)
        ).clamp(0, 1)
        freq = freq * (1 - ramp) + freq / config.factor * ramp
    return freq


def sinusoidal(
    positions: torch.Tensor, dim: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """PE(p,2i)=sin(p*omega_i)，PE(p,2i+1)=cos(p*omega_i)，返回 [...,D]。"""
    if dim < 1:
        raise ValueError("encoding dimension must be positive")
    # Also supports odd d_model (the last cosine is omitted).
    frequencies = 10000.0 ** (-torch.arange(0, dim, 2, device=positions.device, dtype=dtype) / dim)
    phase = positions.to(dtype)[..., None] * frequencies
    return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)[..., :dim]


def apply_rope(
    x: torch.Tensor, positions: torch.Tensor, config: PositionConfig | None = None
) -> torch.Tensor:
    """相邻维度 (a,b) 旋转为 (a*cos-b*sin, a*sin+b*cos)。

    x [B,H,T,D]，positions [T] -> 相同形状。旋转保持范数，
    R(p)q 与 R(s)k 的点积取决于相对位置 p-s。只用于 Q/K，不旋转 V。
    phase 在 fp32/fp64 计算，输出恢复输入 dtype，设备从 x 派生。
    """
    if x.ndim != 4 or positions.ndim != 1 or positions.numel() != x.shape[-2]:
        raise ValueError("expected [B,H,T,D] and positions [T]")
    config = config or PositionConfig()
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    phase = positions.to(device=x.device, dtype=dtype)[:, None] * rotary_frequencies(
        x.shape[-1], config, x.device, dtype
    )
    a, b = x.to(dtype)[..., 0::2], x.to(dtype)[..., 1::2]
    return (
        torch.stack((a * phase.cos() - b * phase.sin(), a * phase.sin() + b * phase.cos()), -1)
        .flatten(-2)
        .to(x.dtype)
    )
