"""训练所需的优化组件：AdamW、带预热的余弦学习率、全局梯度范数裁剪。

``AdamW`` 的更新顺序与 ``torch.optim.AdamW``（foreach=False 的单张量实现）相同，
测试逐步核对两者的参数轨迹。更深入的优化器理论见第五部分。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch


# region adamw
class AdamW(torch.optim.Optimizer):
    """Adam + 解耦权重衰减（Loshchilov & Hutter, 2019）。

    对每个参数 θ、梯度 g，第 t 步（t 从 1 开始）：
        θ ← θ − η·λ·θ                           （解耦权重衰减）
        m ← β₁ m + (1 − β₁) g                   （一阶矩：梯度的指数滑动平均）
        v ← β₂ v + (1 − β₂) g²                  （二阶矩：梯度平方的指数滑动平均）
        θ ← θ − η/(1 − β₁ᵗ) · m / (sqrt(v/(1 − β₂ᵗ)) + ε)
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0 or eps < 0 or not (0 <= betas[0] < 1 and 0 <= betas[1] < 1):
            raise ValueError("非法的超参数")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (beta1, beta2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if not state:  # 惰性初始化：第一次见到这个参数时才分配 m、v
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t, m, v = state["step"], state["m"], state["v"]
                p.mul_(1 - lr * wd)
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bias1 = 1 - beta1**t  # 偏差校正：m、v 从 0 开始，早期被低估
                bias2 = 1 - beta2**t
                denom = (v.sqrt() / math.sqrt(bias2)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bias1)
        return loss
# endregion


# region schedule
def cosine_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    """线性预热到 max_lr，再按余弦从 max_lr 退火到 min_lr，之后保持 min_lr。"""
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)
# endregion


# region clip
def clip_grad_norm_(params: Iterable[torch.Tensor], max_norm: float, eps: float = 1e-6) -> float:
    """把所有梯度视为一个长向量，若其 L2 范数超过 max_norm 就整体等比缩小。返回裁剪前的范数。"""
    grads = [p.grad for p in params if p.grad is not None]
    total = math.sqrt(sum(float(g.pow(2).sum()) for g in grads))
    if total > max_norm:
        scale = max_norm / (total + eps)
        for g in grads:
            g.mul_(scale)
    return total
# endregion
