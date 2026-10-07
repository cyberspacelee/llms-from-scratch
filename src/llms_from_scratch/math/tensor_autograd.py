"""基于 NumPy 的张量自动微分：把标量引擎推广到数组。

对应《从零实现自动微分》一章的后半部分。与标量版本相比，关键差别只有两点：

1. 每个运算的局部反向是一个 VJP（矩阵乘的反向是两次矩阵乘）；
2. 广播在前向时“复制”了数据，反向时就要把梯度沿被复制的维度求和（``sum_to_shape``）。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


# region sum-to-shape
def sum_to_shape(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """把广播后形状的梯度归约回原形状：广播的伴随运算是求和。

    1. 前面补出来的维度：整维求和；
    2. 原来大小为 1、被拉伸的维度：沿该维求和并保留维度。
    """
    extra = grad.ndim - len(shape)
    if extra > 0:
        grad = grad.sum(axis=tuple(range(extra)))
    stretched = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if stretched:
        grad = grad.sum(axis=stretched, keepdims=True)
    return grad


# endregion


class Tensor:
    def __init__(
        self,
        data: np.ndarray | float,
        parents: tuple[Tensor, ...] = (),
        backward: Callable[[np.ndarray], tuple[np.ndarray, ...]] | None = None,
    ) -> None:
        self.data = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray | None = None
        self._parents = parents
        # 给定 dL/d(out)，返回对每个父节点的梯度（与 parents 一一对应）
        self._vjp = backward

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def __add__(self, other: Tensor) -> Tensor:
        a, b = self, other

        def vjp(g: np.ndarray) -> tuple[np.ndarray, ...]:
            return sum_to_shape(g, a.shape), sum_to_shape(g, b.shape)

        return Tensor(a.data + b.data, (a, b), vjp)

    def __mul__(self, other: Tensor) -> Tensor:
        a, b = self, other

        def vjp(g: np.ndarray) -> tuple[np.ndarray, ...]:
            return sum_to_shape(g * b.data, a.shape), sum_to_shape(g * a.data, b.shape)

        return Tensor(a.data * b.data, (a, b), vjp)

    def __matmul__(self, other: Tensor) -> Tensor:
        """二维矩阵乘 C = A B：dA = G Bᵀ，dB = Aᵀ G。"""
        a, b = self, other

        def vjp(g: np.ndarray) -> tuple[np.ndarray, ...]:
            return g @ b.data.T, a.data.T @ g

        return Tensor(a.data @ b.data, (a, b), vjp)

    @property
    def T(self) -> Tensor:  # noqa: N802
        return Tensor(self.data.T, (self,), lambda g: (g.T,))

    def relu(self) -> Tensor:
        mask = self.data > 0
        return Tensor(self.data * mask, (self,), lambda g: (g * mask,))

    def sum(self) -> Tensor:
        shape = self.shape
        return Tensor(self.data.sum(), (self,), lambda g: (np.broadcast_to(g, shape).copy(),))

    def mean(self) -> Tensor:
        n = self.data.size
        shape = self.shape
        return Tensor(self.data.mean(), (self,), lambda g: (np.broadcast_to(g / n, shape).copy(),))

    def log_softmax(self) -> Tensor:
        """沿最后一维：y = z − logsumexp(z)。VJP：g − softmax(z) · Σ g。"""
        z = self.data
        m = z.max(axis=-1, keepdims=True)
        lse = m + np.log(np.exp(z - m).sum(axis=-1, keepdims=True))
        out = z - lse
        p = np.exp(out)

        def vjp(g: np.ndarray) -> tuple[np.ndarray, ...]:
            return (g - p * g.sum(axis=-1, keepdims=True),)

        return Tensor(out, (self,), vjp)

    def pick(self, index: np.ndarray) -> Tensor:
        """每一行取出一个元素：out[n] = self[n, index[n]]。反向把梯度放回原位置。"""
        rows = np.arange(self.shape[0])
        shape = self.shape

        def vjp(g: np.ndarray) -> tuple[np.ndarray, ...]:
            full = np.zeros(shape)
            full[rows, index] = g
            return (full,)

        return Tensor(self.data[rows, index], (self,), vjp)

    def backward(self) -> None:
        order: list[Tensor] = []
        seen: set[int] = set()

        def visit(t: Tensor) -> None:
            if id(t) not in seen:
                seen.add(id(t))
                for p in t._parents:
                    visit(p)
                order.append(t)

        visit(self)
        for t in order:
            t.grad = np.zeros_like(t.data)
        self.grad = np.ones_like(self.data)
        for t in reversed(order):
            if t._vjp is None:
                continue
            for parent, g in zip(t._parents, t._vjp(t.grad), strict=True):
                parent.grad = parent.grad + g  # 多条路径汇合：梯度相加


# region cross-entropy
def cross_entropy(logits: Tensor, targets: np.ndarray) -> Tensor:
    """批平均交叉熵：−mean_n log_softmax(z_n)[y_n]，完全由上面的原语组合而成。"""
    picked = logits.log_softmax().pick(targets)
    return picked.mean() * Tensor(-1.0)


# endregion
