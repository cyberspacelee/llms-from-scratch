"""链式法则与计算图：用对偶数做前向模式，用手写反向做一个小网络。

对应《链式法则与计算图》一章。
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


# region dual
@dataclass(frozen=True)
class Dual:
    """对偶数 a + ḃε（ε² = 0）。``val`` 是函数值，``dot`` 是沿某个方向的导数。

    每个运算同时更新值与导数，这就是前向模式自动微分：
    导数随着计算一起“向前”流动。
    """

    val: float
    dot: float = 0.0

    def __add__(self, other: Dual | float) -> Dual:
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.val + o.val, self.dot + o.dot)

    __radd__ = __add__

    def __sub__(self, other: Dual | float) -> Dual:
        o = other if isinstance(other, Dual) else Dual(other)
        return Dual(self.val - o.val, self.dot - o.dot)

    def __rsub__(self, other: float) -> Dual:
        return Dual(other) - self

    def __mul__(self, other: Dual | float) -> Dual:
        o = other if isinstance(other, Dual) else Dual(other)
        # 乘积法则：(uv)' = u'v + uv'
        return Dual(self.val * o.val, self.dot * o.val + self.val * o.dot)

    __rmul__ = __mul__


def d_sin(x: Dual) -> Dual:
    return Dual(math.sin(x.val), math.cos(x.val) * x.dot)


def d_exp(x: Dual) -> Dual:
    e = math.exp(x.val)
    return Dual(e, e * x.dot)


def d_tanh(x: Dual) -> Dual:
    t = math.tanh(x.val)
    return Dual(t, (1 - t * t) * x.dot)


# endregion


# region forward-mode
def jvp(f: Callable[[Sequence[Dual]], Dual], x: Sequence[float], v: Sequence[float]) -> float:
    """一次前向传递求方向导数 ∇f(x)·v（雅可比-向量积）。"""
    return f([Dual(xi, vi) for xi, vi in zip(x, v, strict=True)]).dot


def forward_mode_gradient(f: Callable[[Sequence[Dual]], Dual], x: Sequence[float]) -> list[float]:
    """前向模式求完整梯度：每个输入坐标要单独跑一遍，共 n 遍。"""
    n = len(x)
    basis = [[1.0 if j == i else 0.0 for j in range(n)] for i in range(n)]
    return [jvp(f, x, e) for e in basis]


# endregion


# region tiny-network
def tiny_network(x: Tensor, w1: Tensor, b1: Tensor, w2: Tensor, b2: Tensor, t: float) -> dict:
    """两层网络的完整前向与手写反向：z = W1 x + b1，h = relu(z)，y = w2·h + b2，ℓ = ½(y − t)²。

    x: (2,)，w1: (2, 2)，b1: (2,)，w2: (2,)，b2: 标量。返回所有中间量与梯度，便于逐项核对。
    """
    # 前向：从输入到损失，记住每个中间结果
    z = w1 @ x + b1
    h = torch.relu(z)
    y = w2 @ h + b2
    loss = 0.5 * (y - t) ** 2

    # 反向：从 dℓ/dℓ = 1 出发，每一步乘上局部导数
    dy = y - t  # dℓ/dy
    dw2 = dy * h  # y = w2·h + b2 ⇒ ∂y/∂w2 = h
    db2 = dy
    dh = dy * w2  # ∂y/∂h = w2
    dz = dh * (z > 0).to(z.dtype)  # relu 的导数：z>0 处为 1，否则为 0
    dw1 = torch.outer(dz, x)  # z_i = Σ_j W1[i,j] x_j + b1_i ⇒ ∂z_i/∂W1[i,j] = x_j
    db1 = dz
    dx = w1.T @ dz  # x_j 影响每一个 z_i，所有路径相加
    return {
        "z": z, "h": h, "y": y, "loss": loss,
        "dy": dy, "dw2": dw2, "db2": db2, "dh": dh, "dz": dz,
        "dw1": dw1, "db1": db1, "dx": dx,
    }


# endregion
