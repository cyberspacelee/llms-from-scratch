"""True means visible. Absolute positions keep chunked causal masks correct."""

from __future__ import annotations

import torch

from .config import AttentionConfig


def attention_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    config: AttentionConfig,
    causal: bool = False,
    key_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    q, k = query_positions[:, None], key_positions[None, :]
    visible = torch.ones((q.numel(), k.numel()), device=q.device, dtype=torch.bool)
    if config.pattern == "sliding":
        visible &= (
            (q - k < config.window) & (q - k >= 0) if causal else (q - k).abs() < config.window
        )
    elif config.pattern == "local":
        visible &= q // config.window == k // config.window
    elif config.pattern == "block_sparse":
        qb, kb = q // config.block_size, k // config.block_size
        visible &= ((qb - kb).abs() <= 1) | (kb % config.global_stride == 0)
    elif config.pattern == "token_sparse":
        visible &= ((q - k).abs() < config.window) | (k % config.global_stride == 0)
    if causal:
        visible &= k <= q
    visible = visible[None, None]  # [1,1,Q,K]
    if key_valid is not None:
        if (
            key_valid.dtype != torch.bool
            or key_valid.ndim != 2
            or key_valid.shape[1] != key_positions.numel()
            or key_valid.device != q.device
        ):
            raise ValueError("key_valid must be boolean [B,K] on the input device")
        visible = visible & key_valid[:, None, None, :]
    return visible


def masked_softmax(scores: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """Zero for a fully masked row; finite gradients, including padded queries."""
    dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    masked = scores.to(dtype).masked_fill(~visible, -torch.inf)
    has_key = visible.any(-1, keepdim=True)
    masked = torch.where(has_key, masked, torch.zeros_like(masked))
    return (masked.softmax(-1) * has_key).to(scores.dtype)
