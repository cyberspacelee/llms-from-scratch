"""知识蒸馏：带温度的 token 级 KD、前向/反向 KL 的对比，以及在线策略蒸馏的一步。"""

from __future__ import annotations

import math

import torch

from llms_from_scratch.post_training.common import masked_mean, response_logprobs, sample_responses
from llms_from_scratch.transformer.model import GPT


# region kd_loss
def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 2.0,
            mask: torch.Tensor | None = None) -> torch.Tensor:
    """T² · KL(p_T^teacher ‖ p_T^student)，在 mask 为真的位置上平均。

    p_T = softmax(z / T)。乘以 T² 是为了抵消软目标梯度随 1/T² 缩小，
    使不同温度下这一项与硬标签交叉熵的相对权重保持不变（Hinton 等，2015）。
    """
    t_logp = (teacher_logits.detach() / temperature).log_softmax(-1)
    s_logp = (student_logits / temperature).log_softmax(-1)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
    if mask is None:
        mask = torch.ones_like(kl, dtype=torch.bool)
    return temperature**2 * masked_mean(kl, mask)
# endregion kd_loss


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """离散分布的 KL(p ‖ q)，沿最后一维求和。"""
    p_logp, q_logp = p_logits.log_softmax(-1), q_logits.log_softmax(-1)
    return (p_logp.exp() * (p_logp - q_logp)).sum(-1)


# region gaussian_fit
def bimodal_logpdf(x: torch.Tensor, mu: float = 2.0, sigma: float = 0.6) -> torch.Tensor:
    """0.5·N(−μ, σ²) + 0.5·N(μ, σ²) 的对数密度。"""
    def normal(m: float) -> torch.Tensor:
        return -0.5 * ((x - m) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))
    return torch.logaddexp(normal(-mu), normal(mu)) + math.log(0.5)


def fit_gaussian(direction: str, steps: int = 600, lr: float = 0.1,
                 init: tuple[float, float] = (1.0, 0.5)) -> tuple[float, float]:
    """用单个高斯 q = N(m, s²) 拟合双峰分布 p，返回 (m, s)。

    direction="forward" 最小化 KL(p‖q)（SFT/传统 KD 的方向），
    direction="reverse" 最小化 KL(q‖p)（在线策略蒸馏 / MiniLLM 的方向）。
    在 [−8, 8] 的网格上用黎曼和计算积分。
    """
    x = torch.linspace(-8, 8, 801, dtype=torch.float64)
    dx = x[1] - x[0]
    log_p = bimodal_logpdf(x)
    m = torch.tensor(init[0], dtype=torch.float64, requires_grad=True)
    log_s = torch.tensor(math.log(init[1]), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([m, log_s], lr=lr)
    for _ in range(steps):
        s = log_s.exp()
        log_q = -0.5 * ((x - m) / s) ** 2 - log_s - 0.5 * math.log(2 * math.pi)
        if direction == "forward":
            loss = (log_p.exp() * (log_p - log_q)).sum() * dx
        elif direction == "reverse":
            loss = (log_q.exp() * (log_q - log_p)).sum() * dx
        else:
            raise ValueError(direction)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.item(), log_s.exp().item()
# endregion gaussian_fit


# region on_policy
def on_policy_distill_step(student: GPT, teacher: GPT, prompts: torch.Tensor,
                           optimizer: torch.optim.Optimizer, max_new_tokens: int, eos_id: int,
                           generator: torch.Generator | None = None) -> dict:
    """在线策略蒸馏的一步：学生采样，教师在学生的每个 token 上打分。

    逐 token“奖励” r_t = log π_teacher(y_t) − log π_student(y_t)，即反向 KL 单样本估计的相反数。
    折扣为 0：每个 token 只对自己的 r_t 负责。采样来自当前学生，重要性比为 1，
    损失就是 −E[r_t · log π_student(y_t)]（r_t 不回传梯度）。
    """
    roll = sample_responses(student, prompts, max_new_tokens, eos_id, generator=generator)
    with torch.no_grad():
        teacher_logp = response_logprobs(teacher, roll.sequences)
    student_logp = response_logprobs(student, roll.sequences)
    advantage = (teacher_logp - student_logp).detach()
    loss = -masked_mean(advantage * student_logp, roll.response_mask)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {"loss": loss.item(),
            "reverse_kl_estimate": -masked_mean(advantage, roll.response_mask).item(),
            "rollout": roll}
# endregion on_policy


@torch.no_grad()
def exact_reverse_kl(student: GPT, teacher: GPT, sequences: torch.Tensor,
                     mask: torch.Tensor) -> float:
    """在给定序列的每个回复位置上，用完整词表精确计算 KL(π_student ‖ π_teacher) 并平均。"""
    s_logits = student(sequences)[:, :-1]
    t_logits = teacher(sequences)[:, :-1]
    return masked_mean(kl_divergence(s_logits, t_logits), mask).item()
