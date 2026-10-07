"""特征分解、SVD 与低秩近似。

对应《特征分解、SVD 与低秩近似》一章：幂迭代、截断 SVD 与 Eckart–Young 误差、
条件数、谱范数，以及 Muon 用到的极分解 UVᵀ 与 Newton–Schulz 迭代。
"""

from __future__ import annotations

import torch
from torch import Tensor


# region power-iteration
def power_iteration(a: Tensor, steps: int = 100, seed: int = 0) -> tuple[float, Tensor]:
    """对称矩阵 A 的主特征对：反复计算 v ← Av / ‖Av‖。

    把初始向量按特征向量展开，每乘一次 A，第 i 个分量就乘上 λ_i，
    最大的 |λ_1| 分量增长最快，方向收敛到 v_1，速度由 |λ_2/λ_1| 决定。
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(a.shape[1], generator=g, dtype=a.dtype)
    v = v / v.norm()
    for _ in range(steps):
        w = a @ v
        v = w / w.norm()
    return (v @ a @ v).item(), v  # Rayleigh 商 vᵀAv 给出特征值


def spectral_norm(a: Tensor, steps: int = 100, seed: int = 0) -> float:
    """‖A‖₂ = σ_max(A) = sqrt(λ_max(AᵀA))，对 AᵀA 做幂迭代，不需要完整 SVD。"""
    lam, _ = power_iteration(a.T @ a, steps, seed)
    return lam**0.5


# endregion


# region low-rank
def low_rank_approx(a: Tensor, k: int) -> Tensor:
    """截断 SVD：A_k = Σ_{i≤k} σ_i u_i v_iᵀ，是所有秩 ≤ k 矩阵中离 A 最近的（Eckart–Young）。"""
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    return (u[:, :k] * s[:k]) @ vh[:k]


def eckart_young_error(a: Tensor, k: int) -> tuple[float, float]:
    """最优秩 k 近似的误差：Frobenius 范数 sqrt(Σ_{i>k} σ_i²)，谱范数 σ_{k+1}。"""
    s = torch.linalg.svdvals(a)
    fro = s[k:].square().sum().sqrt().item()
    spec = s[k].item() if k < len(s) else 0.0
    return fro, spec


def condition_number(a: Tensor) -> float:
    """κ(A) = σ_max / σ_min：输入的相对扰动最多被放大 κ 倍。"""
    s = torch.linalg.svdvals(a)
    return (s[0] / s[-1]).item()


# endregion


# region polar
def polar_factor(g: Tensor) -> Tensor:
    """G = U Σ Vᵀ 的极分解正交因子 U Vᵀ：把所有奇异值换成 1，保留奇异向量。"""
    u, _, vh = torch.linalg.svd(g, full_matrices=False)
    return u @ vh


def newton_schulz(g: Tensor, steps: int = 30) -> Tensor:
    """只用矩阵乘近似 U Vᵀ：X ← 1.5 X − 0.5 X Xᵀ X。

    先除以 Frobenius 范数使所有 σ ≤ 1；每一步把每个奇异值做 σ ↦ 1.5σ − 0.5σ³，
    这个三次多项式在 (0, √3) 上的不动点是 1，于是奇异值被推向 1，奇异向量不变。
    """
    x = g / g.norm()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T  # 让 X Xᵀ 是较小的那个方阵
    for _ in range(steps):
        x = 1.5 * x - 0.5 * (x @ x.T) @ x
    return x.T if transposed else x


# endregion
