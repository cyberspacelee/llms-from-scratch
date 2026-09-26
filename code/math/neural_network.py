"""数学 07：CPU 上核对两层网络梯度，并训练 XOR。"""
import argparse

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def verify_gradients():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(4, 2))
    target = np.array([0, 1, 1, 0])
    W1, b1 = rng.normal(size=(3, 2)), rng.normal(size=3)
    W2, b2 = rng.normal(size=(2, 3)), rng.normal(size=2)
    H = np.tanh(X @ W1.T + b1)
    Z = H @ W2.T + b2
    shifted = Z - Z.max(axis=1, keepdims=True)
    log_p = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    P = np.exp(log_p)
    G2 = (P - np.eye(2)[target]) / len(X)
    G1 = (G2 @ W2) * (1 - H**2)
    manual = [G1.T @ X, G1.sum(axis=0), G2.T @ H, G2.sum(axis=0)]
    params = [torch.tensor(a, dtype=torch.float64, requires_grad=True)
              for a in (W1, b1, W2, b2)]
    w1, c1, w2, c2 = params
    logits = torch.tanh(torch.tensor(X) @ w1.T + c1) @ w2.T + c2
    loss = F.cross_entropy(logits, torch.tensor(target))
    loss.backward()
    np.testing.assert_allclose(loss.item(), -log_p[np.arange(4), target].mean())
    for expected, param in zip(manual, params):
        np.testing.assert_allclose(expected, param.grad.numpy(), rtol=1e-10, atol=1e-12)
    print("两层网络：4 组手工梯度与 autograd 一致")


def train_xor():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float64)
    y = torch.tensor([0, 1, 1, 0])
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2)).double()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    losses = []
    for _ in range(1500):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(X), y)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_loss = F.cross_entropy(model(X), y).item()
        predictions = model(X).argmax(dim=1)
    assert torch.equal(predictions, y)
    assert final_loss < 0.02 and final_loss < losses[0]
    print(f"XOR：loss {losses[0]:.6f} → {final_loss:.6f}；预测 {predictions.tolist()}")
    return model, losses


def plot(model, losses):
    from plot_style import plt, configure, save, GREEN, ORANGE, BLUE, MUTED
    configure()
    x = np.linspace(-4, 4, 401)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    sigmoid = 1 / (1 + np.exp(-x))
    for label, value, derivative, color in [
        ("sigmoid", sigmoid, sigmoid * (1 - sigmoid), GREEN),
        ("tanh", np.tanh(x), 1 - np.tanh(x)**2, BLUE),
        ("ReLU", np.maximum(x, 0), (x > 0).astype(float), ORANGE),
    ]:
        axes[0].plot(x, value, label=label, color=color)
        axes[1].plot(x, derivative, label=label, color=color)
    for ax, title in zip(axes, ["激活函数：输入如何变成表示", "局部导数：梯度如何被缩放"]):
        ax.set(xlabel="输入 z", title=title)
        ax.axhline(0, color=MUTED, linewidth=.6)
        ax.grid(alpha=.2)
        ax.legend()
    axes[0].set_ylabel("激活输出")
    axes[1].set_ylabel("导数值")
    save(fig, "neural-activations.svg")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    axes[0].plot(losses, color=GREEN)
    axes[0].set(xlabel="参数更新次数", ylabel="批平均交叉熵", title="异或问题：训练损失")
    axes[0].grid(alpha=.2)
    grid = np.linspace(-.5, 1.5, 160)
    xx, yy = np.meshgrid(grid, grid)
    with torch.no_grad():
        probs = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()])).softmax(1)[:, 1].numpy()
    mesh = axes[1].contourf(xx, yy, probs.reshape(xx.shape), levels=np.linspace(0, 1, 11), cmap="Blues", vmin=0, vmax=1)
    axes[1].contour(xx, yy, probs.reshape(xx.shape), levels=[.5], colors=[ORANGE])
    axes[1].scatter([0, 0, 1, 1], [0, 1, 0, 1], c=[0, 1, 1, 0], cmap="Blues", vmin=0, vmax=1, edgecolors="black", s=70)
    axes[1].set(xlabel=r"输入 $x_1$", ylabel=r"输入 $x_2$", title="类别 1 概率；橙线为 0.5 边界", aspect="equal")
    fig.colorbar(mesh, ax=axes[1])
    save(fig, "neural-training.svg")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true", help="同时重建讲义函数图与训练图")
    args = parser.parse_args()
    verify_gradients()
    network, history = train_xor()
    if args.plot:
        plot(network, history)
