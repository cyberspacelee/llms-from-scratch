"""导数、偏导数与梯度：数值差分、方向导数与第一个梯度下降。

对应《导数、偏导数与梯度》一章。数值梯度在后面几章反复用作“标准答案”，
因此这里实现得尽量朴素：一次只扰动一个坐标。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor


# region difference
def forward_difference(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """前向差分 (f(x+h) − f(x)) / h，截断误差 O(h)。"""
    return (f(x + h) - f(x)) / h


def central_difference(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    """中心差分 (f(x+h) − f(x−h)) / 2h，截断误差 O(h²)。"""
    return (f(x + h) - f(x - h)) / (2 * h)


# endregion


# region numerical-gradient
def numerical_gradient(f: Callable[[Tensor], Tensor], x: Tensor, h: float = 1e-6) -> Tensor:
    """标量函数 f 在 x 处的梯度，逐坐标中心差分。x 可以是任意形状。

    建议使用 float64：h 太小会被舍入误差淹没，float32 下要取 h≈1e-3。
    """
    x = x.detach().clone()
    grad = torch.zeros_like(x)
    flat_x, flat_g = x.view(-1), grad.view(-1)
    for i in range(flat_x.numel()):
        old = flat_x[i].item()
        flat_x[i] = old + h
        plus = f(x).item()
        flat_x[i] = old - h
        minus = f(x).item()
        flat_x[i] = old
        flat_g[i] = (plus - minus) / (2 * h)
    return grad


def directional_derivative(
    f: Callable[[Tensor], Tensor], x: Tensor, u: Tensor, h: float = 1e-6
) -> float:
    """沿单位方向 u 的方向导数 D_u f(x) = lim (f(x+hu) − f(x−hu)) / 2h。"""
    u = u / u.norm()
    return ((f(x + h * u) - f(x - h * u)) / (2 * h)).item()


# endregion


# region fit-line
@dataclass
class LineFit:
    w: float
    b: float
    losses: list[float]


def line_loss(w: float, b: float, x: Tensor, y: Tensor) -> float:
    """均方误差 L(w, b) = mean((w x + b − y)²) / 2。"""
    return 0.5 * ((w * x + b - y) ** 2).mean().item()


def line_gradient(w: float, b: float, x: Tensor, y: Tensor) -> tuple[float, float]:
    """手推的梯度：∂L/∂w = mean(r·x)，∂L/∂b = mean(r)，其中残差 r = w x + b − y。"""
    r = w * x + b - y
    return (r * x).mean().item(), r.mean().item()


def fit_line(x: Tensor, y: Tensor, lr: float, steps: int, w: float = 0.0, b: float = 0.0) -> LineFit:
    """用最朴素的梯度下降拟合 y ≈ w x + b，记录每一步的损失。"""
    losses = [line_loss(w, b, x, y)]
    for _ in range(steps):
        gw, gb = line_gradient(w, b, x, y)
        w, b = w - lr * gw, b - lr * gb  # 沿负梯度走一步
        losses.append(line_loss(w, b, x, y))
    return LineFit(w, b, losses)


# endregion


# region quadratic
def descend_quadratic(curvature: float, lr: float, x0: float, steps: int) -> list[float]:
    """在 f(x) = c x² / 2 上做梯度下降，返回轨迹。

    更新 x ← x − η c x = (1 − η c) x：|1 − η c| < 1 即 η < 2/c 时收敛。
    """
    xs = [x0]
    for _ in range(steps):
        xs.append(xs[-1] - lr * curvature * xs[-1])
    return xs


# endregion
