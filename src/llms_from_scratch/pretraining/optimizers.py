"""优化器：从零实现的 AdamW（含 L2 对照）、Newton–Schulz 正交化与 Muon。

Muon 只更新隐藏层的二维权重矩阵；嵌入、输出头、RMSNorm 增益等其余参数交给 AdamW。
`split_muon_params` 与 `HybridOptimizer` 负责这一划分。
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import nn


# region adamw
class AdamW(torch.optim.Optimizer):
    """Adam + 权重衰减。decoupled=True 是 AdamW；False 是把 λθ 加进梯度的 “Adam + L2”。"""

    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01, decoupled: bool = True) -> None:
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                      weight_decay=weight_decay, decoupled=decoupled))

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if not group["decoupled"]:
                    g = g + wd * p                     # L2：衰减项进入矩估计，会被 1/√v 缩放
                state = self.state[p]
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)   # 一阶矩
                    state["v"] = torch.zeros_like(p)   # 二阶（原点）矩
                state["t"] += 1
                t, m, v = state["t"], state["m"], state["v"]
                m.lerp_(g, 1 - b1)                     # m ← β1·m + (1-β1)·g
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1**t)                # 偏差修正
                v_hat = v / (1 - b2**t)
                if group["decoupled"]:
                    p.mul_(1 - lr * wd)                # 解耦：直接把参数向 0 拉，不经过 1/√v
                p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
# endregion adamw


# region newton_schulz
MUON_COEFFS = (3.4445, -4.7750, 2.0315)   # Keller Jordan 调出的五次多项式系数
CUBIC_COEFFS = (1.5, -0.5, 0.0)           # 经典三次 Newton–Schulz：f(σ) = 1.5σ - 0.5σ³


def newton_schulz(G: torch.Tensor, steps: int = 5,
                  coeffs: tuple[float, float, float] = MUON_COEFFS,
                  eps: float = 1e-7) -> torch.Tensor:
    """近似 G 的极分解因子 UVᵀ（G = UΣVᵀ）。

    每步 X ← aX + b(XXᵀ)X + c(XXᵀ)²X，等价于对每个奇异值作用 φ(σ) = aσ + bσ³ + cσ⁵，
    奇异向量不变。先除以 Frobenius 范数，保证所有奇异值落在 (0, 1]。
    """
    a, b, c = coeffs
    X = G.float()
    transposed = X.size(-2) > X.size(-1)
    if transposed:                 # 让 XXᵀ 取较小的那一维，省计算
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.mT if transposed else X


def polar_factor(G: torch.Tensor) -> torch.Tensor:
    """用 SVD 精确计算 UVᵀ，作为 Newton–Schulz 的参照。"""
    U, _, Vh = torch.linalg.svd(G.double(), full_matrices=False)
    return (U @ Vh).to(G.dtype)


def ns_scalar_map(sigma: float, steps: int = 5,
                  coeffs: tuple[float, float, float] = MUON_COEFFS) -> float:
    """单个奇异值经过 steps 次迭代后的值：φ∘φ∘…∘φ(σ)。用于画图与分析。"""
    a, b, c = coeffs
    for _ in range(steps):
        sigma = a * sigma + b * sigma**3 + c * sigma**5
    return sigma
# endregion newton_schulz


# region muon
class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton–Schulz，只接受二维参数。

    update = NS(Nesterov 动量)；再按 rms_scale 调整尺度：
      - "moonlight"：乘 0.2·√max(m, n)，使更新的 RMS ≈ 0.2，与 AdamW 的典型更新 RMS 对齐，
        于是可以直接沿用 AdamW 的学习率与权重衰减（Liu et al. 2025；Kimi K2、DeepSeek-V4 同法）。
      - "keller"：乘 √max(1, m/n)（Keller Jordan 的原始实现）。
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, weight_decay: float = 0.0, rms_scale: str = "moonlight") -> None:
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))
        if rms_scale not in ("moonlight", "keller"):
            raise ValueError("rms_scale 只能是 'moonlight' 或 'keller'")
        self.rms_scale = rms_scale
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError("Muon 只用于二维权重矩阵，其余参数请交给 AdamW")

    def _scale(self, rows: int, cols: int) -> float:
        if self.rms_scale == "moonlight":
            return 0.2 * math.sqrt(max(rows, cols))
        return math.sqrt(max(1.0, rows / cols))

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = torch.zeros_like(p)
                buf = state["buf"]
                buf.mul_(mu).add_(p.grad)                       # M ← μM + G
                m = p.grad + mu * buf if group["nesterov"] else buf
                update = newton_schulz(m, group["ns_steps"]).to(p.dtype)
                p.mul_(1 - lr * wd)                             # 解耦权重衰减
                p.add_(update, alpha=-lr * self._scale(*p.shape))
# endregion muon


# region hybrid
def split_muon_params(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """按 Muon 论文的建议划分：隐藏层二维矩阵 → Muon；嵌入、输出头与一维参数 → AdamW。

    named_parameters() 对共享参数只返回一次，所以权重共享的 embed/lm_head 不会被重复计入。
    """
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_io = name.startswith(("embed", "lm_head"))
        (muon if p.ndim == 2 and not is_io else adamw).append(p)
    return muon, adamw


class HybridOptimizer:
    """把 Muon 与 AdamW 包装成一个对象，训练循环只需调用 step / zero_grad。"""

    def __init__(self, model: nn.Module, lr: float = 3e-3, weight_decay: float = 0.1,
                 betas: tuple[float, float] = (0.9, 0.95), momentum: float = 0.95) -> None:
        muon_params, adamw_params = split_muon_params(model)
        # 采用 RMS 匹配后，两组可以共享同一个学习率
        self.optimizers: list[torch.optim.Optimizer] = [
            Muon(muon_params, lr=lr, momentum=momentum, weight_decay=weight_decay),
            AdamW(adamw_params, lr=lr, betas=betas, weight_decay=weight_decay),
        ]

    @property
    def param_groups(self) -> list[dict]:
        return [g for opt in self.optimizers for g in opt.param_groups]

    def step(self) -> None:
        for opt in self.optimizers:
            opt.step()

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)
# endregion hybrid


# region memory
def optimizer_state_bytes(params: Iterable[torch.Tensor] | int, optimizer: str = "adamw",
                          state_bytes: int = 4) -> int:
    """优化器状态占用的字节数（不含参数与梯度本身）。

    SGD 无状态；带动量的 SGD 与 Muon 每个参数存 1 份；Adam/AdamW 存 m、v 两份。
    """
    n = params if isinstance(params, int) else sum(p.numel() for p in params)
    copies = {"sgd": 0, "momentum": 1, "muon": 1, "adamw": 2}[optimizer]
    return n * copies * state_bytes
# endregion memory
