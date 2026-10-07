import math

import torch

from llms_from_scratch.math.micrograd import MLP, Value, train_mlp


def test_reused_node_accumulates_gradient():
    a = Value(3.0)
    b = a * a + a  # a 被用了三次
    b.backward()
    assert a.grad == 2 * 3.0 + 1


def test_expression_matches_torch():
    a, b = Value(-4.0), Value(2.0)
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g = g + (a * 0.1).tanh() + (b * 0.3).exp().log()
    g.backward()

    ta = torch.tensor(-4.0, dtype=torch.float64, requires_grad=True)
    tb = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    c = ta + tb
    d = ta * tb + tb**3
    c = c + c + 1
    c = c + 1 + c + (-ta)
    d = d + d * 2 + (tb + ta).relu()
    d = d + 3 * d + (tb - ta).relu()
    e = c - d
    f = e**2
    tg = f / 2.0
    tg = tg + 10.0 / f
    tg = tg + torch.tanh(ta * 0.1) + torch.log(torch.exp(tb * 0.3))
    tg.backward()

    assert math.isclose(g.data, tg.item(), rel_tol=1e-12)
    assert math.isclose(a.grad, ta.grad.item(), rel_tol=1e-9)
    assert math.isclose(b.grad, tb.grad.item(), rel_tol=1e-9)


def test_mlp_gradients_match_torch():
    model = MLP([3, 4, 1], seed=1)
    x = [0.5, -1.0, 2.0]
    out = model(x)[0]
    out.backward()
    # 用同样的权重在 PyTorch 中重建网络
    (w1, b1), (w2, b2) = [
        (
            torch.tensor([[p.data for p in n.w] for n in layer], dtype=torch.float64, requires_grad=True),
            torch.tensor([n.b.data for n in layer], dtype=torch.float64, requires_grad=True),
        )
        for layer in model.layers
    ]
    tx = torch.tensor(x, dtype=torch.float64)
    ref = (w2 @ torch.tanh(w1 @ tx + b1) + b2)[0]
    ref.backward()
    assert math.isclose(out.data, ref.item(), rel_tol=1e-12)
    mine_w1 = torch.tensor([[p.grad for p in n.w] for n in model.layers[0]], dtype=torch.float64)
    torch.testing.assert_close(mine_w1, w1.grad)
    torch.testing.assert_close(
        torch.tensor([n.b.grad for n in model.layers[1]], dtype=torch.float64), b2.grad
    )


def test_training_reduces_loss():
    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ys = [1.0, -1.0, -1.0, 1.0]
    model = MLP([3, 4, 4, 1], seed=0)
    history = train_mlp(model, xs, ys, lr=0.1, steps=200)
    assert history[-1] < 0.01 < history[0]
