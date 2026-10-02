"""Teacher Forcing、Next-Token 和 MTP 联合损失；显式处理标签对齐。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn import functional as F

from ..layers import Routing

if TYPE_CHECKING:
    from ..models import ModelOutput, Transformer


def teacher_forcing(
    targets: torch.Tensor, bos_id: int, pad_id: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Targets [B,S] include EOS; output [BOS,y0,...,y(T-2)], same targets.

    Args:
        targets: long [B,S]，真实监督 token IDs。
        bos_id: 非负 BOS token ID。
        pad_id: 可选 PAD ID，用于生成有效监督 mask。

    Returns:
        tuple: long inputs[B,S]、targets[B,S]、bool valid[B,S]，inputs 为 BOS+targets[:,:-1]。
    """
    if (
        targets.ndim != 2
        or targets.dtype != torch.long
        or targets.numel() == 0
        or type(bos_id) is not int
        or bos_id < 0
    ):
        raise ValueError("expected nonempty targets and nonnegative BOS")
    inputs = torch.cat((torch.full_like(targets[:, :1], bos_id), targets[:, :-1]), 1)
    valid = torch.ones_like(targets, dtype=torch.bool) if pad_id is None else targets != pad_id
    return inputs, targets, valid


def token_loss(
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """只在有效监督位置计算交叉熵。

    Args:
        logits: float [B,S,V] 词表 logits。
        targets: long [B,S]，真实监督 token IDs。
        valid: 可选 bool [B,S] 监督位置。

    Returns:
        可微 scalar [] 平均 cross-entropy；无有效目标时返回可微零。
    """
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or targets.dtype != torch.long:
        raise ValueError("expected logits [B,S,V] and long targets [B,S]")
    if valid is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    if valid.shape != targets.shape or valid.dtype != torch.bool or valid.device != targets.device:
        raise ValueError("invalid loss mask")
    selected = targets[valid]  # [N_valid]；logits[valid]为[N_valid,V]
    if selected.numel() and (selected.min() < 0 or selected.max() >= logits.shape[-1]):
        raise ValueError("target outside vocabulary")
    if not selected.numel():
        return logits.sum() * 0
    return F.cross_entropy(logits[valid], selected)


def next_token_loss(
    logits: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    """logits[:,t] 监督 tokens[:,t+1]，丢掉没有 next-token 标签的最后一位。

    Args:
        logits: float [B,S,V] 词表 logits。
        tokens: long [B,S] 未右移的真实 token 序列。
        valid: 可选 bool [B,S] token有效性，同时检查当前位置和下一位置。

    Returns:
        scalar [] NTP loss，logits[:,:S-1] 对齐 tokens[:,1:S]。
    """
    mask = None if valid is None else valid[:, :-1] & valid[:, 1:]
    return token_loss(logits[:, :-1], tokens[:, 1:], mask)


def language_model_loss(
    model: Transformer,
    output: ModelOutput,
    tokens: torch.Tensor,
    valid: torch.Tensor | None = None,
    mtp_weight: float = 0.3,
    auxiliary_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Routing]:
    """L=NTP + mtp_weight*mean(MTP_j) + auxiliary_weight*MoE_aux。

    返回可 backward 的总损失、各深度诊断损失和显式 routing 记录。
    MTP 的有效 mask 覆盖当前位置到未来标签之间的全部 token，不能跨 padding
    把互不连续的文本当成一个预测链。普通 generate 只使用 NTP 主 head。

    Args:
        model: 参与检查/损失计算的模型。
        output: ModelOutput，logits[B,S,V]、hidden[B,S,D] 和 scalar MoE loss。
        tokens: long [B,S] 未右移的真实 token 序列。
        valid: 可选 bool [B,S] 有效 token mask，True=参与监督。
        mtp_weight: 额外 MTP 平均损失的非负权重。
        auxiliary_weight: MoE 辅助损失的非负权重。

    Returns:
        tuple: scalar 总 loss、dict[str,scalar] 分项 loss、Routing records。
    """
    if mtp_weight < 0 or auxiliary_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    losses = {"ntp": next_token_loss(output.logits, tokens, valid)}
    auxiliary = output.auxiliary_loss
    routing = output.routing
    if model.mtp is not None:
        predictions, aux, stats = model.mtp(
            output.hidden, tokens, model.embedding, model.head, valid
        )
        auxiliary = auxiliary + aux
        routing = routing + stats
        active = torch.ones_like(tokens, dtype=torch.bool) if valid is None else valid
        for depth, logits in enumerate(predictions, 2):
            mask = active[:, : tokens.shape[1] - depth].clone()
            for offset in range(1, depth + 1):
                mask &= active[:, offset : tokens.shape[1] - depth + offset]
            losses[f"mtp_{depth}"] = token_loss(logits, tokens[:, depth:], mask)
    mtp = [value for name, value in losses.items() if name.startswith("mtp_")]
    total = losses["ntp"] + auxiliary_weight * auxiliary
    if mtp:
        total = total + mtp_weight * torch.stack(mtp).mean()
    return total, losses, routing
