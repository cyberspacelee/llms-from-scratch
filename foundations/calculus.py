"""数学 04/06：用 float64 校验导数、梯度下降与反向传播。"""
import numpy as np
import torch


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_default_dtype(torch.float64)

    # 导数与均匀分布积分的确定性核对。
    grid = np.array([0.2, 1.0, 2.0], dtype=np.float64)
    eps = 1e-5
    sigmoid = lambda value: 1 / (1 + np.exp(-value))
    for function, derivative in [
        (np.exp, np.exp), (np.log, lambda value: 1 / value),
        (sigmoid, lambda value: sigmoid(value) * (1 - sigmoid(value))),
    ]:
        numeric = (function(grid + eps) - function(grid - eps)) / (2 * eps)
        np.testing.assert_allclose(numeric, derivative(grid), rtol=1e-8, atol=1e-9)
    n = 1000
    u = (np.arange(n, dtype=np.float64) + 0.5) / n
    np.testing.assert_allclose(np.mean(u ** 2) - np.mean(u) ** 2,
                               1 / 12 - 1 / (12 * n ** 2))

    # 中心差分：验证偏导，而非用它替代训练时的自动微分。
    def loss(v):
        return 0.5 * (v[0] ** 2 + 4 * v[1] ** 2)

    x = np.array([2.0, 1.0])
    eps = 1e-5
    basis = np.eye(2)
    numeric = np.array([(loss(x + eps * e) - loss(x - eps * e)) / (2 * eps)
                        for e in basis])
    np.testing.assert_allclose(numeric, [2.0, 4.0], rtol=1e-9)
    direction = np.array([3.0, 4.0]) / 5.0
    directional = (loss(x + eps * direction) - loss(x - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(directional, numeric @ direction, rtol=1e-9)
    for _ in range(30):
        old = loss(x)
        x -= 0.2 * np.array([x[0], 4 * x[1]])
        assert loss(x) < old
    assert loss(x) < 1e-5

    # Taylor 二阶项精确补足二次损失的变化。
    point = np.array([2.0, 1.0])
    delta = np.array([0.01, -0.02])
    hessian = np.diag([1.0, 4.0])
    predicted = loss(point) + (hessian @ point) @ delta + 0.5 * delta @ hessian @ delta
    np.testing.assert_allclose(predicted, loss(point + delta))

    # 一元链与共享节点的贡献累加。
    u = torch.tensor(2.0, requires_grad=True)
    a = u ** 2
    ell = a + 3 * a
    ell.backward()
    torch.testing.assert_close(u.grad, torch.tensor(16.0))

    # 手算线性层 + 平方损失；向量默认列、W 的布局为 (out, in)。
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    W = torch.tensor([[2.0, 1.0], [-1.0, 3.0]], requires_grad=True)
    b = torch.tensor([0.5, -0.5], requires_grad=True)
    y = torch.tensor([1.0, -1.0])
    z = W @ x + b
    ell = 0.5 * ((z - y) ** 2).sum()
    g = (z - y).detach()
    expected_W = g[:, None] @ x.detach()[None, :]
    expected_x = W.detach().T @ g
    ell.backward()
    torch.testing.assert_close(W.grad, expected_W)
    torch.testing.assert_close(b.grad, g)
    torch.testing.assert_close(x.grad, expected_x)

    # 新建计算图后第二次 backward 会累加已有 .grad。
    (0.5 * ((W @ x + b - y) ** 2).sum()).backward()
    torch.testing.assert_close(W.grad, 2 * expected_W)
    W.grad = None
    b.grad = None
    x.grad = None
    (0.5 * ((W @ x + b - y) ** 2).sum()).backward()
    torch.testing.assert_close(W.grad, expected_W)

    # 两层网络逐层核对梯度。
    x2 = torch.tensor([1.0, -2.0], requires_grad=True)
    W1 = torch.tensor([[2.0, 1.0], [-1.0, 3.0]], requires_grad=True)
    b1 = torch.tensor([0.5, -0.5], requires_grad=True)
    W2 = torch.tensor([[1.0, -2.0]], requires_grad=True)
    b2 = torch.tensor([0.5], requires_grad=True)
    hidden = W1 @ x2 + b1
    hidden.retain_grad()
    prediction = W2 @ hidden + b2
    (0.5 * ((prediction - 1) ** 2).sum()).backward()
    torch.testing.assert_close(hidden.grad, torch.tensor([15.0, -30.0]))
    torch.testing.assert_close(W1.grad, torch.tensor([[15.0, -30.0], [-30.0, 60.0]]))
    torch.testing.assert_close(W2.grad, torch.tensor([[7.5, -112.5]]))
    torch.testing.assert_close(x2.grad, torch.tensor([60.0, -75.0]))

    # 不等长微批的平均必须按样本数加权。
    inputs = torch.tensor([1.0, 2.0, 3.0])
    targets = torch.tensor([0.0, 1.0, 1.0])
    weight = torch.tensor(0.5, requires_grad=True)
    (0.5 * ((weight * inputs - targets) ** 2).mean()).backward()
    reference = weight.grad.clone()
    weight.grad = None
    for indices in [slice(0, 2), slice(2, 3)]:
        xb, yb = inputs[indices], targets[indices]
        micro_loss = 0.5 * ((weight * xb - yb) ** 2).mean()
        (micro_loss * len(xb) / len(inputs)).backward()
    torch.testing.assert_close(weight.grad, reference)
    torch.testing.assert_close(weight.grad, torch.tensor(2 / 3))

    # 种子与求导边界。
    point = torch.tensor([2.0, 1.0], requires_grad=True)
    output = torch.stack([point[0] ** 2, 3 * point[1]])
    output.backward(torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(point.grad, torch.tensor([4.0, 6.0]))
    leaf = torch.tensor(2.0, requires_grad=True)
    squared = leaf ** 2
    with torch.no_grad():
        frozen = leaf ** 2
    assert not frozen.requires_grad
    (squared + squared.detach()).backward()
    torch.testing.assert_close(leaf.grad, torch.tensor(4.0))

    zero = torch.tensor(0.0, requires_grad=True)
    torch.relu(zero).backward()
    torch.testing.assert_close(zero.grad, torch.tensor(0.0))
    print("math-04/06: derivative, descent, branch sum, VJP, accumulation, ReLU checks passed")


if __name__ == "__main__":
    main()
