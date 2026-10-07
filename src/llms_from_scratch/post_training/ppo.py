"""RLHF 中的 PPO 组件：token 级 KL 奖励、GAE、裁剪的策略损失与价值损失，以及策略梯度的方差实验。"""

from __future__ import annotations

import torch

from llms_from_scratch.post_training.common import masked_mean


# region kl_rewards
def kl_penalized_rewards(score: torch.Tensor, logp: torch.Tensor, ref_logp: torch.Tensor,
                         mask: torch.Tensor, beta: float) -> torch.Tensor:
    """把序列级奖励模型分数与 token 级 KL 惩罚组合成逐 token 奖励。

    r_t = −β (log π(y_t) − log π_ref(y_t))，最后一个有效 token 再加上 RM 分数 score。
    score[B]；logp、ref_logp、mask 形状均为 [B, T]（只在回复 token 上 mask 为真）。
    """
    rewards = -beta * (logp - ref_logp) * mask
    last = mask.long().cumsum(1).argmax(1)  # 每行最后一个有效位置
    rewards[torch.arange(len(score)), last] += score
    return rewards
# endregion kl_rewards


# region gae
def compute_gae(rewards: torch.Tensor, values: torch.Tensor, mask: torch.Tensor,
                gamma: float = 1.0, lam: float = 0.95) -> tuple[torch.Tensor, torch.Tensor]:
    """广义优势估计。rewards、values、mask 形状 [B, T]；回复结束后的价值视为 0。

    δ_t = r_t + γ V_{t+1} − V_t，  Â_t = δ_t + γλ Â_{t+1}（从后往前递推）。
    返回 (advantages, returns)，returns = Â + V 是价值函数的回归目标。
    """
    B, T = rewards.shape
    mask = mask.to(rewards.dtype)
    advantages = torch.zeros_like(rewards)
    next_value = torch.zeros(B, dtype=rewards.dtype)
    next_adv = torch.zeros(B, dtype=rewards.dtype)
    for t in reversed(range(T)):
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        adv = delta + gamma * lam * next_adv
        advantages[:, t] = adv * mask[:, t]
        # 无效位置不向前传递：下一步看到的仍是最近一个有效位置的值
        next_value = torch.where(mask[:, t] > 0, values[:, t], next_value)
        next_adv = torch.where(mask[:, t] > 0, adv, next_adv)
    return advantages, advantages + values * mask
# endregion gae


# region ppo_loss
def ppo_policy_loss(logp: torch.Tensor, old_logp: torch.Tensor, advantages: torch.Tensor,
                    mask: torch.Tensor, clip_eps: float = 0.2) -> torch.Tensor:
    """L = −E[min(ρ Â, clip(ρ, 1−ε, 1+ε) Â)]，ρ = π_θ / π_old 逐 token 计算。"""
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * advantages
    clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages
    return -masked_mean(torch.minimum(unclipped, clipped), mask)


def ppo_value_loss(values: torch.Tensor, old_values: torch.Tensor, returns: torch.Tensor,
                   mask: torch.Tensor, clip_eps: float = 0.2) -> torch.Tensor:
    """价值函数也裁剪：新价值相对旧价值的变化超过 ε 时用裁剪后的值，取两者中较大的误差。"""
    clipped = old_values + (values - old_values).clamp(-clip_eps, clip_eps)
    loss = torch.maximum((values - returns) ** 2, (clipped - returns) ** 2)
    return 0.5 * masked_mean(loss, mask)
# endregion ppo_loss


def whiten(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """在有效位置上把优势标准化为零均值单位方差（PPO 实现中的常见技巧）。"""
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    return (x - mean) * torch.rsqrt(var + eps) * mask


# region reinforce_variance
def reinforce_gradient_stats(logits: torch.Tensor, rewards: torch.Tensor, baseline: float
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """单步类别策略 π = softmax(logits) 上，单样本 REINFORCE 估计 ĝ = (r(a) − b) ∇log π(a) 的
    精确均值与总方差（对所有动作求和，而不是随机采样）。

    返回 (E[ĝ], Σ_i Var[ĝ_i])。无论 b 取何值均值都等于 ∇ E[r]，方差则依赖 b。
    """
    probs = logits.softmax(-1)
    n = logits.numel()
    # ∇_logits log π(a) = e_a − π
    grad_logp = torch.eye(n, dtype=logits.dtype) - probs
    g = (rewards - baseline)[:, None] * grad_logp  # 第 a 行：采到动作 a 时的估计
    mean = (probs[:, None] * g).sum(0)
    var = (probs[:, None] * (g - mean) ** 2).sum()
    return mean, var
# endregion reinforce_variance
