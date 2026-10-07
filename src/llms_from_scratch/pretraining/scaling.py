"""缩放定律：幂律拟合、Chinchilla 参数化损失、IsoFLOP 分析与计算最优分配。

记号：N 参数量，D 训练 token 数，C ≈ 6ND 训练 FLOPs。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch


# region chinchilla
@dataclass(frozen=True)
class ChinchillaParams:
    """L(N, D) = E + A / N^α + B / D^β。"""
    E: float
    A: float
    B: float
    alpha: float
    beta: float


# Hoffmann et al. 2022 原文（Approach 3）
HOFFMANN = ChinchillaParams(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)
# Besiroglu et al. 2024 重新拟合（修正了原文优化提前终止的问题）
EPOCH_REFIT = ChinchillaParams(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)


def chinchilla_loss(N, D, p: ChinchillaParams = EPOCH_REFIT):
    return p.E + p.A / np.power(N, p.alpha) + p.B / np.power(D, p.beta)


def training_flops(N: float, D: float) -> float:
    """C ≈ 6ND：前向 2ND，反向 4ND（见第二部分的资源核算）。"""
    return 6.0 * N * D


def compute_optimal(C: float, p: ChinchillaParams = EPOCH_REFIT) -> tuple[float, float]:
    """在 6ND = C 约束下最小化 L(N, D) 的解析解。

    N* = G·(C/6)^a，D* = G⁻¹·(C/6)^b，其中 G = (αA / (βB))^{1/(α+β)}，
    a = β/(α+β)，b = α/(α+β)。推导见正文。
    """
    G = (p.alpha * p.A / (p.beta * p.B)) ** (1 / (p.alpha + p.beta))
    a = p.beta / (p.alpha + p.beta)
    b = p.alpha / (p.alpha + p.beta)
    return G * (C / 6) ** a, (C / 6) ** b / G
# endregion chinchilla


# region power_law
def fit_power_law(x, y) -> tuple[float, float]:
    """y = k·x^a 在对数坐标下是直线 log y = log k + a log x，用最小二乘求 (k, a)。"""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    a, logk = np.polyfit(lx, ly, 1)
    return float(np.exp(logk)), float(a)


def fit_chinchilla(N, D, L, delta: float = 1e-3, steps: int = 100) -> ChinchillaParams:
    """Approach 3：在 log 空间用 Huber 损失 + L-BFGS 拟合 (E, A, B, α, β)。

    参数化 A = e^a、B = e^b、E = e^e，模型的对数损失写成 LSE(a - α log N, b - β log D, e)，
    在小网格上多起点优化，取目标最小的一组（与 Hoffmann et al. 的做法相同）。
    """
    logN = torch.tensor(np.log(N), dtype=torch.float64)
    logD = torch.tensor(np.log(D), dtype=torch.float64)
    logL = torch.tensor(np.log(L), dtype=torch.float64)

    def objective(theta: torch.Tensor) -> torch.Tensor:
        a, b, e, alpha, beta = theta
        pred = torch.logsumexp(torch.stack([a - alpha * logN, b - beta * logD,
                                            e.expand_as(logN)]), dim=0)
        return torch.nn.functional.huber_loss(pred, logL, delta=delta, reduction="sum")

    best, best_val = None, math.inf
    for a0 in (2.0, 8.0):
        for b0 in (2.0, 8.0):
            for alpha0 in (0.3,):
                theta = torch.tensor([a0, b0, 0.5, alpha0, alpha0], dtype=torch.float64,
                                     requires_grad=True)
                opt = torch.optim.LBFGS([theta], max_iter=steps, tolerance_grad=1e-12,
                                        tolerance_change=1e-14, line_search_fn="strong_wolfe")

                def closure(theta=theta, opt=opt):
                    opt.zero_grad()
                    loss = objective(theta)
                    loss.backward()
                    return loss

                opt.step(closure)
                val = objective(theta).item()
                if math.isfinite(val) and val < best_val:
                    best_val, best = val, theta.detach().clone()
    a, b, e, alpha, beta = best.tolist()
    return ChinchillaParams(E=math.exp(e), A=math.exp(a), B=math.exp(b), alpha=alpha, beta=beta)
# endregion power_law


# region isoflop
def isoflop_minimum(N, L) -> float:
    """Approach 2：同一计算预算下，损失对 log N 拟合抛物线，返回顶点处的 N。"""
    c2, c1, _ = np.polyfit(np.log(N), L, 2)
    if c2 <= 0:
        raise ValueError("损失对 log N 不是凸的，扫描范围可能没有覆盖最优点")
    return float(np.exp(-c1 / (2 * c2)))


def isoflop_analysis(budgets, p: ChinchillaParams, points: int = 9, span: float = 1.5,
                     noise: float = 0.0, seed: int = 0) -> tuple[list[float], tuple[float, float]]:
    """对每个预算 C，在 N*（解析）附近的对数网格上“训练”模型（用 L(N, C/6N) 代替），
    找到 IsoFLOP 曲线最低点，再把 N_opt 对 C 拟合幂律 N_opt = k·C^a。"""
    rng = np.random.default_rng(seed)
    n_opts = []
    for C in budgets:
        center, _ = compute_optimal(C, p)
        Ns = center * np.exp(np.linspace(-span, span, points))
        Ls = chinchilla_loss(Ns, C / (6 * Ns), p) + noise * rng.standard_normal(points)
        n_opts.append(isoflop_minimum(Ns, Ls))
    return n_opts, fit_power_law(budgets, n_opts)
# endregion isoflop


# region data_constrained
def effective_data(unique_tokens: float, epochs: float, r_star: float = 15.4) -> float:
    """Muennighoff et al. 2023：重复 R = epochs - 1 轮后的“等效独特 token 数”。

    D' = U + U·R*·(1 - e^{-R/R*})。重复数据的价值指数衰减，R* ≈ 15.4 是衰减常数。
    """
    repeats = max(0.0, epochs - 1)
    return unique_tokens * (1 + r_star * (1 - math.exp(-repeats / r_star)))


def tokens_per_param(N: float, D: float) -> float:
    return D / N
# endregion data_constrained
