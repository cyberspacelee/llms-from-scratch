"""M6/M8：形状、几何、手工梯度、autograd 和中心差分验证。

运行：.venv/bin/python examples/math/linear_algebra.py
依赖：numpy、torch。固定种子，CPU，float64。
"""

import numpy as np
import torch


def check_geometry():
    x = np.array([3.0, 4.0], dtype=np.float64)
    y = np.array([2.0, 0.0], dtype=np.float64)
    projection = (y @ x) / (y @ y) * y
    np.testing.assert_allclose(projection, [3.0, 0.0])
    np.testing.assert_allclose((x - projection) @ y, 0.0)
    np.testing.assert_allclose(np.linalg.norm(x), 5.0)
    Q = np.array([[0.0, -1.0], [1.0, 0.0]])
    np.testing.assert_allclose(Q.T @ Q, np.eye(2))
    np.testing.assert_allclose((Q @ x) @ (Q @ y), x @ y)
    np.testing.assert_allclose(np.linalg.norm(Q @ x), np.linalg.norm(x))
    X = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.0]])
    W = np.array([[1.0, -1.0], [2.0, 0.0]])
    b = np.array([0.5, -1.0])
    np.testing.assert_allclose(X @ W.T + b, [[-0.5, 1.0], [-0.5, 5.0], [-0.5, -3.0]])
    np.testing.assert_allclose(X @ W.T + b, np.stack([W @ row + b for row in X]))


def central_difference(loss_fn, parameter, h=1e-6):
    """逐元素扰动独立副本，适用于本讲义的小型平滑函数。"""
    numerical = np.empty_like(parameter)
    for index in np.ndindex(parameter.shape):
        plus, minus = parameter.copy(), parameter.copy()
        plus[index] += h
        minus[index] -= h
        numerical[index] = (loss_fn(plus) - loss_fn(minus)) / (2 * h)
    return numerical


def check_gradients():
    rng = np.random.default_rng(42)
    B, d, m = 4, 3, 2
    X = rng.normal(size=(B, d))
    W = rng.normal(size=(m, d))
    b = rng.normal(size=m)
    Y = rng.normal(size=(B, m))

    def loss(x, w, bias):
        error = x @ w.T + bias - Y
        return np.sum(error**2) / (2 * B)

    G = (X @ W.T + b - Y) / B
    manual = (G @ W, G.T @ X, G.sum(axis=0))
    tensors = tuple(torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in (X, W, b))
    xt, wt, bt = tensors
    error = xt @ wt.T + bt - torch.tensor(Y, dtype=torch.float64)
    objective = error.square().sum() / (2 * B)
    objective.backward()
    numerical = (
        central_difference(lambda a: loss(a, W, b), X),
        central_difference(lambda a: loss(X, a, b), W),
        central_difference(lambda a: loss(X, W, a), b),
    )
    for name, tensor, expected, approximation in zip(("X", "W", "b"), tensors, manual, numerical):
        assert tensor.grad.shape == tensor.shape == expected.shape
        np.testing.assert_allclose(tensor.grad.numpy(), expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(approximation, expected, rtol=1e-6, atol=1e-8)
        print(
            f"{name}: autograd 与手工一致；中心差分最大绝对误差 {np.max(np.abs(approximation - expected)):.3e}"
        )

    # 单样本的完整手算例，与正文数字核对。
    x = np.array([1.0, 2.0])
    w = np.array([[1.0, -1.0], [2.0, 0.0]])
    bias = np.array([0.5, -1.0])
    g = w @ x + bias
    np.testing.assert_allclose(0.5 * (g @ g), 0.625)
    np.testing.assert_allclose(np.outer(g, x), [[-0.5, -1.0], [1.0, 2.0]])
    np.testing.assert_allclose(w.T @ g, [1.5, 0.5])

    # region batch_gradient
    # M8's three records share the same W and b as M6.
    chapter_x = np.array([[1.0, 2.0], [3.0, 4.0], [-1.0, 0.0]])
    chapter_error = chapter_x @ w.T + bias
    chapter_g = chapter_error / len(chapter_x)
    np.testing.assert_allclose(np.sum(chapter_error**2) / (2 * len(chapter_x)), 143 / 24)
    np.testing.assert_allclose(chapter_g.T @ chapter_x, [[-0.5, -1.0], [19 / 3, 22 / 3]])
    np.testing.assert_allclose(chapter_g.sum(axis=0), [-0.5, 1.0])
    tx = torch.tensor(chapter_x, requires_grad=True)
    tw = torch.tensor(w, requires_grad=True)
    tb = torch.tensor(bias, requires_grad=True)
    chapter_loss = (tx @ tw.T + tb).square().sum() / (2 * len(tx))
    chapter_loss.backward()
    np.testing.assert_allclose(tw.grad, chapter_g.T @ chapter_x)
    np.testing.assert_allclose(tb.grad, chapter_g.sum(axis=0))
    np.testing.assert_allclose(tx.grad, chapter_g @ w)
    # endregion batch_gradient

    # 正文的两个样本：逐样本外积求平均，与矩阵公式严格对应。
    batch_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    residual = batch_x @ w.T + bias
    batch_g = residual / len(batch_x)
    individual = np.stack([np.outer(e, row) for e, row in zip(residual, batch_x)])
    np.testing.assert_allclose(individual.mean(axis=0), batch_g.T @ batch_x)
    np.testing.assert_allclose(batch_g.T @ batch_x, [[-1.0, -1.5], [8.0, 11.0]])
    np.testing.assert_allclose(batch_g.sum(axis=0), [-0.5, 3.0])
    repeated_x = np.repeat(batch_x, 2, axis=0)
    repeated_g = (repeated_x @ w.T + bias) / len(repeated_x)
    np.testing.assert_allclose(repeated_g.T @ repeated_x, batch_g.T @ batch_x)

    # 非方阵和多个输出数避免转置、平均系数错误被巧合掩盖。
    for outputs in (1, 3):
        weights = torch.randn(
            (outputs, d),
            generator=torch.Generator().manual_seed(42),
            dtype=torch.float64,
            requires_grad=True,
        )
        prediction = torch.tensor(X) @ weights.T
        target = torch.zeros_like(prediction)
        half_batch = (prediction - target).square().sum() / (2 * B)
        mse = torch.nn.functional.mse_loss(prediction, target)
        (grad_half,) = torch.autograd.grad(half_batch, weights, retain_graph=True)
        (grad_mse,) = torch.autograd.grad(mse, weights)
        torch.testing.assert_close(grad_mse, grad_half * (2 / outputs))
    print("均方误差归约系数验证通过")


if __name__ == "__main__":
    check_geometry()
    check_gradients()
    print("M6/M8 全部验证通过")
