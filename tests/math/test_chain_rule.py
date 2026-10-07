import math

import torch

from llms_from_scratch.math.chain_rule import (
    Dual,
    d_exp,
    d_sin,
    d_tanh,
    forward_mode_gradient,
    jvp,
    tiny_network,
)


def test_dual_numbers_follow_the_chain_rule():
    x = Dual(1.3, 1.0)
    # f(x) = sin(x²)，f'(x) = 2x cos(x²)
    assert math.isclose(d_sin(x * x).dot, 2 * 1.3 * math.cos(1.3**2))
    # 两条路径：g(x) = x · exp(x)，g'(x) = exp(x) + x exp(x)
    assert math.isclose((x * d_exp(x)).dot, math.exp(1.3) * (1 + 1.3))


def test_forward_mode_gradient_matches_autograd():
    def f(v):
        x, y, z = v
        return d_tanh(x * y + z) * x + d_sin(z)

    point = [0.3, -1.2, 0.8]
    grad = forward_mode_gradient(f, point)
    t = torch.tensor(point, dtype=torch.float64, requires_grad=True)
    (torch.tanh(t[0] * t[1] + t[2]) * t[0] + torch.sin(t[2])).backward()
    torch.testing.assert_close(torch.tensor(grad, dtype=torch.float64), t.grad)
    v = [1.0, 2.0, -1.0]  # 一次 JVP = 梯度与方向的点积
    assert math.isclose(jvp(f, point, v), (t.grad @ torch.tensor(v, dtype=torch.float64)).item())


def params():
    return dict(
        x=torch.tensor([1.0, 2.0]),
        w1=torch.tensor([[1.0, -1.0], [0.5, 1.0]]),
        b1=torch.tensor([0.5, -1.0]),
        w2=torch.tensor([2.0, -1.0]),
        b2=torch.tensor(0.5),
    )


def test_tiny_network_hand_values():
    out = tiny_network(**params(), t=1.0)
    torch.testing.assert_close(out["z"], torch.tensor([-0.5, 1.5]))
    torch.testing.assert_close(out["loss"], torch.tensor(2.0))
    torch.testing.assert_close(out["dw2"], torch.tensor([0.0, -3.0]))
    torch.testing.assert_close(out["dw1"], torch.tensor([[0.0, 0.0], [2.0, 4.0]]))
    torch.testing.assert_close(out["dx"], torch.tensor([1.0, 2.0]))


def test_tiny_network_matches_autograd():
    p = {k: v.clone().requires_grad_() for k, v in params().items()}
    out = tiny_network(**{k: v.detach() for k, v in p.items()}, t=1.0)
    loss = 0.5 * (p["w2"] @ torch.relu(p["w1"] @ p["x"] + p["b1"]) + p["b2"] - 1.0) ** 2
    loss.backward()
    for name in ("x", "w1", "b1", "w2", "b2"):
        torch.testing.assert_close(out["d" + name], p[name].grad)
