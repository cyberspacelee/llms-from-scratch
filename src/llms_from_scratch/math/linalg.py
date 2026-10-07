"""向量、矩阵与张量：矩阵乘法的四种看法、广播规则与一层神经网络。

本模块对应《向量、矩阵与张量》一章。每个函数都刻意用最直白的循环或
逐项构造来实现，再在测试中与 ``A @ B``、``torch.einsum`` 对照。
"""

from __future__ import annotations

import torch
from torch import Tensor


# region cosine
def cosine_similarity(u: Tensor, v: Tensor) -> Tensor:
    """两个向量夹角的余弦：cos θ = u·v / (‖u‖‖v‖)。"""
    return (u @ v) / (u.norm() * v.norm())


# endregion


# region four-views
def matmul_dots(a: Tensor, b: Tensor) -> Tensor:
    """看法一：C[i, j] 是 A 的第 i 行与 B 的第 j 列的点积。"""
    m, k = a.shape
    k2, n = b.shape
    assert k == k2, f"内维不匹配：{a.shape} @ {b.shape}"
    c = torch.empty(m, n, dtype=a.dtype)
    for i in range(m):
        for j in range(n):
            c[i, j] = a[i, :] @ b[:, j]
    return c


def matmul_columns(a: Tensor, b: Tensor) -> Tensor:
    """看法二：C 的第 j 列是 A 的各列以 B[:, j] 为系数的线性组合。"""
    columns = []
    for j in range(b.shape[1]):
        col = sum(b[p, j] * a[:, p] for p in range(a.shape[1]))
        columns.append(col)
    return torch.stack(columns, dim=1)


def matmul_outer(a: Tensor, b: Tensor) -> Tensor:
    """看法三：C 是 k 个秩 1 矩阵（A 的第 p 列 ⊗ B 的第 p 行）之和。"""
    c = torch.zeros(a.shape[0], b.shape[1], dtype=a.dtype)
    for p in range(a.shape[1]):
        c += torch.outer(a[:, p], b[p, :])
    return c


def matmul_rows(a: Tensor, b: Tensor) -> Tensor:
    """看法四（线性变换）：C 的第 i 行是把 A 的第 i 行这个向量送进 x ↦ xB 的结果。"""
    return torch.stack([a[i, :] @ b for i in range(a.shape[0])], dim=0)


# endregion


# region broadcast
def broadcast_shape(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """按 NumPy/PyTorch 规则计算广播后的形状。

    从最右边的维度开始逐维对齐；缺失的维度视为 1；
    两个维度相等，或其中一个为 1，才能广播，结果取较大者。
    """
    ndim = max(len(s) for s in shapes)
    padded = [(1,) * (ndim - len(s)) + tuple(s) for s in shapes]
    out = []
    for dims in zip(*padded, strict=True):
        sizes = {d for d in dims if d != 1}
        if len(sizes) > 1:
            raise ValueError(f"无法广播的形状：{shapes}")
        out.append(sizes.pop() if sizes else 1)
    return tuple(out)


# endregion


# region layer
def linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """与 ``torch.nn.functional.linear`` 相同：y = x Wᵀ + b。

    x: (..., d_in)，weight: (d_out, d_in)，bias: (d_out,)。
    """
    y = x @ weight.T
    if bias is not None:
        y = y + bias  # (…, d_out) + (d_out,)：沿前面所有维度广播
    return y


def two_layer_mlp(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor) -> Tensor:
    """一层网络 = 矩阵乘 + 非线性；两层叠起来就是最简单的 MLP。"""
    h = torch.relu(linear(x, w1, b1))
    return linear(h, w2, b2)


# endregion


# region einsum
def attention_scores_einsum(q: Tensor, k: Tensor) -> Tensor:
    """q, k: (B, H, T, D) → 分数 (B, H, T, T)，S[b,h,i,j] = Σ_d q[b,h,i,d]·k[b,h,j,d]。"""
    return torch.einsum("bhid,bhjd->bhij", q, k)


def attention_scores_matmul(q: Tensor, k: Tensor) -> Tensor:
    """同一个计算的批量矩阵乘写法：最后两维做矩阵乘，前面的维度当作批。"""
    return q @ k.transpose(-2, -1)


# endregion
