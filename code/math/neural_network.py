"""数学 07：核对同一个两单元 ReLU 网络的首步梯度与 XOR 训练。"""
import argparse

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]], dtype=torch.float64)
TARGET = torch.tensor([0, 1, 1, 0])


def make_model():
    model = nn.Sequential(nn.Linear(2, 2), nn.ReLU(), nn.Linear(2, 2)).double()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[1., -1.], [-1., 1.]]))
        model[0].bias.zero_()
        model[2].weight.copy_(torch.tensor([[0., 0.], [-1., -1.]]))
        model[2].bias.zero_()
    return model


def verify_first_step():
    model = make_model()
    logits = model(X)
    loss = F.cross_entropy(logits, TARGET)
    q = 1 / (1 + np.e)
    a = (1 - q) / 4
    np.testing.assert_allclose(loss.item(), (np.log(2) + np.log1p(np.e)) / 2)
    loss.backward()
    expected = [
        [[a, 0], [0, a]], [a, a],
        [[a, a], [-a, -a]], [2 * a - .25, .25 - 2 * a],
    ]
    for param, gradient in zip(model.parameters(), expected):
        np.testing.assert_allclose(param.grad.numpy(), gradient, rtol=1e-12, atol=1e-12)
    optimizer = torch.optim.SGD(model.parameters(), lr=.3)
    optimizer.step()
    np.testing.assert_allclose(F.cross_entropy(model(X), TARGET).item(), .9236409233667829)
    assert model(X).argmax(1).tolist() == [1, 0, 0, 1]
    print("首步：4 组梯度与 autograd 一致；loss 1.003204 → 0.923641")


def train_xor():
    torch.set_num_threads(1)
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.3)
    losses = []
    for _ in range(1500):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(X), TARGET)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_loss = F.cross_entropy(model(X), TARGET).item()
        predictions = model(X).argmax(dim=1)
    assert torch.equal(predictions, TARGET)
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
    verify_first_step()
    network, history = train_xor()
    if args.plot:
        plot(network, history)
