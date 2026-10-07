"""DPO 及其变体（IPO、SimPO、KTO）的损失函数。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from llms_from_scratch.post_training.common import token_logprobs
from llms_from_scratch.transformer.model import GPT


# region sequence_logprob
def sequence_logprob(logits: torch.Tensor, input_ids: torch.Tensor,
                     response_mask: torch.Tensor, average: bool = False) -> torch.Tensor:
    """log π(y|x) = Σ_t log π(y_t | x, y_<t)，只对回复 token 求和。

    logits[B, T, V] 来自 model(input_ids)；response_mask[B, T] 标出 input_ids 中属于回复的位置。
    位置 t-1 的 logits 预测 token t，所以掩码与目标都取 [:, 1:]。
    average=True 时除以回复长度（SimPO 使用长度归一化的对数概率）。
    """
    lp = token_logprobs(logits[:, :-1], input_ids[:, 1:])
    mask = response_mask[:, 1:].to(lp.dtype)
    total = (lp * mask).sum(-1)
    return total / mask.sum(-1) if average else total


def model_sequence_logprob(model: GPT, input_ids: torch.Tensor, response_mask: torch.Tensor,
                           average: bool = False) -> torch.Tensor:
    return sequence_logprob(model(input_ids), input_ids, response_mask, average)
# endregion sequence_logprob


# region dpo_loss
def dpo_loss(policy_chosen: torch.Tensor, policy_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float = 0.1
             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """L = −log σ(β[(log π_w − log π_ref,w) − (log π_l − log π_ref,l)])。

    输入都是序列对数概率 [B]。返回 (loss, 选中回复的隐式奖励, 被拒回复的隐式奖励)，
    隐式奖励 r̂ = β (log π − log π_ref) 已从计算图中分离，仅用于监控。
    """
    chosen = policy_chosen - ref_chosen.detach()
    rejected = policy_rejected - ref_rejected.detach()
    loss = -F.logsigmoid(beta * (chosen - rejected)).mean()
    return loss, beta * chosen.detach(), beta * rejected.detach()
# endregion dpo_loss


# region variants
def ipo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected, tau: float = 0.1):
    """IPO：把对数似然比之差回归到常数 1/(2τ)，而不是推向无穷。"""
    h = (policy_chosen - ref_chosen) - (policy_rejected - ref_rejected)
    return ((h - 1 / (2 * tau)) ** 2).mean()


def simpo_loss(avg_chosen: torch.Tensor, avg_rejected: torch.Tensor, beta: float = 2.0,
               gamma: float = 1.0) -> torch.Tensor:
    """SimPO：隐式奖励取长度归一化的平均对数概率 β/|y|·log π(y|x)，不需要参考模型，外加间隔 γ。"""
    return -F.logsigmoid(beta * (avg_chosen - avg_rejected) - gamma).mean()


def kto_loss(policy_logp: torch.Tensor, ref_logp: torch.Tensor, desirable: torch.Tensor,
             kl_ref_point: torch.Tensor, beta: float = 0.1, lambda_d: float = 1.0,
             lambda_u: float = 1.0) -> torch.Tensor:
    """KTO：只需要“好/坏”的单条标签，不需要成对数据。

    r = log π/π_ref；参考点 z0 是策略与参考模型 KL 的估计（不回传梯度）。
    好样本的价值 λ_D σ(β(r − z0))，坏样本 λ_U σ(β(z0 − r))，损失为 λ − 价值。
    """
    r = policy_logp - ref_logp
    z0 = kl_ref_point.detach().clamp_min(0)
    value = torch.where(desirable, lambda_d * torch.sigmoid(beta * (r - z0)),
                        lambda_u * torch.sigmoid(beta * (z0 - r)))
    weight = torch.where(desirable, torch.full_like(r, lambda_d), torch.full_like(r, lambda_u))
    return (weight - value).mean()
# endregion variants
