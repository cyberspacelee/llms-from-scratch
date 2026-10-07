"""大规模推理 RL 中对 GRPO 的改进：损失聚合方式、clip-higher、动态采样、超长惩罚、
GSPO 的序列级重要性比、截断重要性采样（TIS）与熵。

约定：per_token / logp / mask 的形状为 [N, T]（N 条回复，T 为补齐后的回复长度），
advantages 为 [N]（每条回复一个标量）。
"""

from __future__ import annotations

from typing import Literal

import torch

Aggregation = Literal["seq-mean-token-mean", "token-mean", "seq-mean-token-sum-norm"]


# region aggregate
def aggregate(per_token: torch.Tensor, mask: torch.Tensor, mode: Aggregation,
              max_len: int | None = None) -> torch.Tensor:
    """把逐 token 损失聚合成标量。三种方式只差在每个 token 的权重：

    seq-mean-token-mean      (GRPO)     权重 1 / (N·|o_i|)：短回复的 token 权重大
    token-mean               (DAPO)     权重 1 / Σ_j |o_j|：所有 token 等权
    seq-mean-token-sum-norm  (Dr. GRPO) 权重 1 / (N·L_max)：常数，与长度无关
    """
    mask = mask.to(per_token.dtype)
    summed = (per_token * mask).sum(1)  # [N]
    if mode == "seq-mean-token-mean":
        return (summed / mask.sum(1).clamp_min(1)).mean()
    if mode == "token-mean":
        return summed.sum() / mask.sum().clamp_min(1)
    if mode == "seq-mean-token-sum-norm":
        if max_len is None:
            raise ValueError("Dr. GRPO 需要一个与样本无关的常数 max_len")
        return summed.sum() / (per_token.shape[0] * max_len)
    raise ValueError(f"未知聚合方式 {mode!r}")
# endregion aggregate


# region dapo
def clipped_surrogate(logp: torch.Tensor, old_logp: torch.Tensor, advantages: torch.Tensor,
                      eps_low: float = 0.2, eps_high: float = 0.28) -> torch.Tensor:
    """逐 token 的 −min(ρÂ, clip(ρ, 1−ε_low, 1+ε_high)Â)。ε_high > ε_low 即 DAPO 的 clip-higher。"""
    adv = advantages[:, None]
    ratio = torch.exp(logp - old_logp)
    return -torch.minimum(ratio * adv, ratio.clamp(1 - eps_low, 1 + eps_high) * adv)


def dynamic_sampling_keep(rewards: torch.Tensor) -> torch.Tensor:
    """rewards[P, G] → keep[P]：丢掉组内奖励全相同（全对或全错）的问题，它们的优势全为 0。"""
    return rewards.max(dim=1).values > rewards.min(dim=1).values


def overlong_penalty(lengths: torch.Tensor, max_len: int, cache_len: int) -> torch.Tensor:
    """DAPO 的软超长惩罚：长度 ≤ L_max − L_cache 不罚；之后线性降到 −1；超过 L_max 罚 −1。"""
    lengths = lengths.to(torch.float32)
    soft_start = max_len - cache_len
    ramp = (soft_start - lengths) / cache_len
    return torch.where(lengths <= soft_start, torch.zeros_like(lengths),
                       torch.where(lengths <= max_len, ramp, -torch.ones_like(lengths)))
# endregion dapo


# region gspo
def gspo_loss(logp: torch.Tensor, old_logp: torch.Tensor, advantages: torch.Tensor,
              mask: torch.Tensor, eps_low: float = 3e-4, eps_high: float = 4e-4) -> torch.Tensor:
    """GSPO：重要性比按序列定义 s_i = (π_θ(y_i)/π_old(y_i))^{1/|y_i|}，在序列级裁剪。

    s_i 是逐 token 比值的几何平均：exp(对数比值在有效 token 上的平均)。
    """
    mask = mask.to(logp.dtype)
    mean_log_ratio = ((logp - old_logp) * mask).sum(1) / mask.sum(1).clamp_min(1)
    s = torch.exp(mean_log_ratio)
    surrogate = torch.minimum(s * advantages, s.clamp(1 - eps_low, 1 + eps_high) * advantages)
    return -surrogate.mean()
# endregion gspo


# region tis
def tis_weights(train_logp: torch.Tensor, rollout_logp: torch.Tensor,
                cap: float = 2.0) -> torch.Tensor:
    """截断重要性采样权重 min(π_train / π_rollout, C)，不回传梯度。

    rollout_logp 来自推理引擎（如 vLLM）采样时记录的对数概率，train_logp 由训练引擎
    用同一份权重重新计算；两者因 kernel、精度、批组成不同而不完全相等。
    """
    return torch.exp(train_logp - rollout_logp).clamp(max=cap).detach()
# endregion tis


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """逐位置的策略熵 H = −Σ p log p，形状 logits.shape[:-1]。"""
    logp = logits.log_softmax(-1)
    return -(logp.exp() * logp).sum(-1)


def effective_token_weights(mask: torch.Tensor, mode: Aggregation,
                            max_len: int | None = None) -> torch.Tensor:
    """每个有效 token 在聚合中的权重（= 对该 token 损失的偏导数），用于比较三种方式。"""
    per_token = torch.zeros(mask.shape, requires_grad=True)
    aggregate(per_token, mask, mode, max_len).backward()
    return per_token.grad
