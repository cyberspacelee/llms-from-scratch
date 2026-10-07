import numpy as np
import pytest
import torch
import torch.nn.functional as F

from llms_from_scratch.math.tensor_autograd import Tensor, cross_entropy, sum_to_shape


@pytest.mark.parametrize(
    "small",
    [(4,), (1, 4), (3, 1), (1, 1), (), (2, 1, 4)],
)
def test_sum_to_shape_is_adjoint_of_broadcast(small):
    big = np.broadcast_shapes(small, (2, 3, 4))
    rng = np.random.default_rng(0)
    g = rng.standard_normal(big)
    x = rng.standard_normal(small)
    # ⟨broadcast(x), g⟩ = ⟨x, sum_to_shape(g)⟩：求和是广播的伴随
    lhs = (np.broadcast_to(x, big) * g).sum()
    rhs = (x * sum_to_shape(g, small)).sum()
    assert np.isclose(lhs, rhs)
    assert sum_to_shape(g, small).shape == small


def test_broadcast_add_mul_gradients():
    rng = np.random.default_rng(1)
    a_np, b_np = rng.standard_normal((3, 4)), rng.standard_normal((1, 4))
    a, b = Tensor(a_np), Tensor(b_np)
    ((a + b) * b).sum().backward()
    ta = torch.tensor(a_np, requires_grad=True)
    tb = torch.tensor(b_np, requires_grad=True)
    ((ta + tb) * tb).sum().backward()
    np.testing.assert_allclose(a.grad, ta.grad.numpy())
    np.testing.assert_allclose(b.grad, tb.grad.numpy())


def test_mlp_cross_entropy_matches_torch():
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((8, 5))
    w1_np, b1_np = rng.standard_normal((5, 16)) * 0.5, rng.standard_normal((16,)) * 0.1
    w2_np, b2_np = rng.standard_normal((16, 3)) * 0.5, rng.standard_normal((3,)) * 0.1
    y = rng.integers(0, 3, size=8)

    x, w1, b1, w2, b2 = (Tensor(a) for a in (x_np, w1_np, b1_np, w2_np, b2_np))
    loss = cross_entropy((x @ w1 + b1).relu() @ w2 + b2, y)
    loss.backward()

    tx, tw1, tb1, tw2, tb2 = (
        torch.tensor(a, requires_grad=True) for a in (x_np, w1_np, b1_np, w2_np, b2_np)
    )
    ref = F.cross_entropy(torch.relu(tx @ tw1 + tb1) @ tw2 + tb2, torch.tensor(y))
    ref.backward()

    assert np.isclose(loss.data, ref.item())
    for mine, theirs in zip((x, w1, b1, w2, b2), (tx, tw1, tb1, tw2, tb2)):
        np.testing.assert_allclose(mine.grad, theirs.grad.numpy(), atol=1e-12)


def test_transpose_and_reuse():
    rng = np.random.default_rng(2)
    a_np = rng.standard_normal((3, 3))
    a = Tensor(a_np)
    (a @ a.T).mean().backward()  # a 出现在两条路径上
    ta = torch.tensor(a_np, requires_grad=True)
    (ta @ ta.T).mean().backward()
    np.testing.assert_allclose(a.grad, ta.grad.numpy())
