"""Absolute PE and adjacent-pair RoPE, all phases evaluated in fp32/fp64."""

from __future__ import annotations

import math

import torch

from .config import PositionConfig


def rotary_frequencies(
    dim: int,
    config: PositionConfig,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
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
    if dim < 1:
        raise ValueError("encoding dimension must be positive")
    # Also supports odd d_model (the last cosine is omitted).
    frequencies = 10000.0 ** (-torch.arange(0, dim, 2, device=positions.device, dtype=dtype) / dim)
    phase = positions.to(dtype)[..., None] * frequencies
    return torch.stack((phase.sin(), phase.cos()), -1).flatten(-2)[..., :dim]


def apply_rope(
    x: torch.Tensor, positions: torch.Tensor, config: PositionConfig | None = None
) -> torch.Tensor:
    """[B,H,T,D], positions [T]; R(p) preserves norms and relative dot products."""
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
