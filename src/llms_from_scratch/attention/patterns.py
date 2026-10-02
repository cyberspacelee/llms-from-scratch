"""True means visible. Absolute positions keep chunked causal masks correct."""

from __future__ import annotations

import torch

from ..config import AttentionConfig


def attention_mask(
    query_positions: torch.Tensor,
    key_positions: torch.Tensor,
    config: AttentionConfig,
    causal: bool = False,
    key_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """根据绝对 q/k 位置生成 [1或B,1,Q,T] 可见性，随后叠加 padding。

    causal 使用 k<=q；decode 的 q 通常从 S-1 开始，不能用一个局部 tril
    代替此条件。pattern 只改变可见关系，本函数仍生成密集 mask，
    不会自动减少 Q*S 的 score 分配；真实稀疏计算见 gathered_attention。

    Args:
        query_positions: long [T_q] query 绝对位置。
        key_positions: long [T_kv] key 绝对位置。
        config: 本模块的显式配置对象。
        causal: 是否限制 key 绝对位置不超过 query。
        key_valid: 可选 bool [B,T_kv]，True 表示有效，含完整历史。

    Returns:
        bool visible[1或B,1,T_q,T_kv]，可广播到 Attention scores。
    """
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
            raise ValueError("key_valid must be boolean [B,T_kv] on the input device")
        visible = visible & key_valid[:, None, None, :]
    return visible


def masked_softmax(scores: torch.Tensor, visible: torch.Tensor) -> torch.Tensor:
    """Zero for a fully masked row; finite gradients, including padded queries.

    Args:
        scores: float [B,H_q,T_q,T_kv] 或 [...,K]。
        visible: bool [...,K] mask，可广播到 scores；True=可见。

    Returns:
        概率 tensor，同 scores shape/dtype；全遮挡行返回零。
    """
    dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
    masked = scores.to(dtype).masked_fill(~visible, -torch.inf)
    has_key = visible.any(-1, keepdim=True)
    masked = torch.where(has_key, masked, torch.zeros_like(masked))
    return (masked.softmax(-1) * has_key).to(scores.dtype)
