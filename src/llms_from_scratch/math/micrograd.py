"""标量自动微分引擎（micrograd 风格）与一个小 MLP。

对应《从零实现自动微分》一章的前半部分。每个 ``Value`` 记住：

* ``data``：前向计算出的值；
* ``grad``：最终输出对它的偏导数，反向传播时累加；
* ``_parents`` 与 ``_backward``：它由哪些节点算出，以及如何把自己的梯度分给它们。
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable


class Value:
    def __init__(self, data: float, parents: tuple[Value, ...] = (), op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._parents = parents
        self._backward = lambda: None
        self.op = op

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4g}, grad={self.grad:.4g})"

    @staticmethod
    def wrap(x: Value | float) -> Value:
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other: Value | float) -> Value:
        other = Value.wrap(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # ∂(a+b)/∂a = ∂(a+b)/∂b = 1；用 += 是因为同一节点可能被多处使用
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = Value.wrap(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k: float) -> Value:
        out = Value(self.data**k, (self,), f"**{k}")

        def _backward() -> None:
            self.grad += k * self.data ** (k - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(max(0.0, self.data), (self,), "relu")

        def _backward() -> None:
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    # 其余运算都由上面几种组合而成
    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other: Value | float) -> Value:
        return self + (-Value.wrap(other))

    def __truediv__(self, other: Value | float) -> Value:
        return self * Value.wrap(other) ** -1

    __radd__ = __add__
    __rmul__ = __mul__

    def __rsub__(self, other: float) -> Value:
        return Value(other) - self

    def __rtruediv__(self, other: float) -> Value:
        return Value(other) / self

    def backward(self) -> None:
        """从本节点出发反向传播：先拓扑排序，再按逆序调用每个节点的局部反向。"""
        order: list[Value] = []
        visited: set[int] = set()

        def visit(v: Value) -> None:
            if id(v) in visited:
                return
            visited.add(id(v))
            for p in v._parents:
                visit(p)
            order.append(v)  # 所有父节点都已在它之前入列

        visit(self)
        self.grad = 1.0  # dL/dL = 1
        for v in reversed(order):
            v._backward()


# region mlp
class Neuron:
    def __init__(self, n_in: int, nonlinear: bool, rng: random.Random) -> None:
        bound = 1 / math.sqrt(n_in)
        self.w = [Value(rng.uniform(-bound, bound)) for _ in range(n_in)]
        self.b = Value(0.0)
        self.nonlinear = nonlinear

    def __call__(self, x: list[Value]) -> Value:
        act = sum((wi * xi for wi, xi in zip(self.w, x, strict=True)), self.b)
        return act.tanh() if self.nonlinear else act

    def parameters(self) -> list[Value]:
        return [*self.w, self.b]


class MLP:
    """全连接网络：隐藏层用 tanh，最后一层线性输出。"""

    def __init__(self, sizes: list[int], seed: int = 0) -> None:
        rng = random.Random(seed)
        self.layers = [
            [Neuron(sizes[i], i < len(sizes) - 2, rng) for _ in range(sizes[i + 1])]
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x: Iterable[float | Value]) -> list[Value]:
        h = [Value.wrap(v) for v in x]
        for layer in self.layers:
            h = [neuron(h) for neuron in layer]
        return h

    def parameters(self) -> list[Value]:
        return [p for layer in self.layers for n in layer for p in n.parameters()]


def train_mlp(
    model: MLP, xs: list[list[float]], ys: list[float], lr: float = 0.1, steps: int = 100
) -> list[float]:
    """全批量梯度下降最小化均方误差，返回损失曲线。"""
    history = []
    for _ in range(steps):
        preds = [model(x)[0] for x in xs]
        loss = sum(((p - y) ** 2 for p, y in zip(preds, ys, strict=True)), Value(0.0)) / len(xs)
        for p in model.parameters():
            p.grad = 0.0  # 梯度是累加的，每步之前必须清零
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        history.append(loss.data)
    return history


# endregion
