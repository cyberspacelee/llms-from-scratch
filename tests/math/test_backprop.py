import pytest
import torch
import torch.nn.functional as F

from llms_from_scratch.math import backprop as bp
from llms_from_scratch.math.calculus import numerical_gradient

D = torch.float64


def rand(*shape, seed=0):
    return torch.randn(*shape, dtype=D, generator=torch.Generator().manual_seed(seed))


def test_linear_backward_matches_autograd():
    x, w, b = rand(5, 4, seed=0), rand(3, 4, seed=1), rand(3, seed=2)
    dy = rand(5, 3, seed=3)
    out, cache = bp.linear_forward(x, w, b)
    dx, dw, db = bp.linear_backward(dy, cache)
    xs, ws, bs = (t.clone().requires_grad_() for t in (x, w, b))
    ref = F.linear(xs, ws, bs)
    torch.testing.assert_close(out, ref)
    ref.backward(dy)
    torch.testing.assert_close(dx, xs.grad)
    torch.testing.assert_close(dw, ws.grad)
    torch.testing.assert_close(db, bs.grad)


@pytest.mark.parametrize(
    "fwd,bwd,ref",
    [
        (bp.relu_forward, bp.relu_backward, F.relu),
        (bp.gelu_forward, bp.gelu_backward, F.gelu),
        (bp.silu_forward, bp.silu_backward, F.silu),
    ],
)
def test_activation_backward(fwd, bwd, ref):
    x = rand(6, 5) + 0.01  # 避开 relu 在 0 处的不可导点
    dy = rand(6, 5, seed=1)
    out, cache = fwd(x)
    xs = x.clone().requires_grad_()
    expected = ref(xs)
    torch.testing.assert_close(out, expected)
    expected.backward(dy)
    torch.testing.assert_close(bwd(dy, cache), xs.grad)
    assert torch.autograd.gradcheck(bp.as_autograd(fwd, bwd), (x.clone().requires_grad_(),))


def test_softmax_vjp_matches_explicit_jacobian():
    z = rand(4)
    g = rand(4, seed=1)
    s, cache = bp.softmax_forward(z)
    jac = torch.diag(s) - torch.outer(s, s)
    torch.testing.assert_close(bp.softmax_backward(g, cache), jac.T @ g)
    torch.testing.assert_close(jac, torch.autograd.functional.jacobian(lambda t: torch.softmax(t, 0), z))
    zz = rand(3, 5).requires_grad_()
    assert torch.autograd.gradcheck(bp.as_autograd(bp.softmax_forward, bp.softmax_backward), (zz,))


def test_cross_entropy_gradient_is_p_minus_y():
    logits = rand(4, 6)
    targets = torch.tensor([0, 5, 2, 2])
    loss, cache = bp.cross_entropy_forward(logits, targets)
    torch.testing.assert_close(loss, F.cross_entropy(logits, targets))
    grad = bp.cross_entropy_backward(torch.tensor(1.0, dtype=D), cache)
    p = torch.softmax(logits, -1)
    torch.testing.assert_close(grad, (p - F.one_hot(targets, 6)) / 4)
    numeric = numerical_gradient(lambda z: F.cross_entropy(z, targets), logits)
    torch.testing.assert_close(grad, numeric, atol=1e-7, rtol=1e-6)
    op = bp.as_autograd(
        bp.cross_entropy_forward, lambda g, c: (bp.cross_entropy_backward(g, c), None)
    )
    assert torch.autograd.gradcheck(op, (logits.clone().requires_grad_(), targets))


def test_layernorm_backward():
    x, gamma, beta = rand(2, 3, 8), rand(8, seed=1), rand(8, seed=2)
    dy = rand(2, 3, 8, seed=3)
    out, cache = bp.layernorm_forward(x, gamma, beta)
    xs, gs, bs = (t.clone().requires_grad_() for t in (x, gamma, beta))
    ref = F.layer_norm(xs, (8,), gs, bs, eps=1e-5)
    torch.testing.assert_close(out, ref)
    ref.backward(dy)
    for mine, theirs in zip(bp.layernorm_backward(dy, cache), (xs.grad, gs.grad, bs.grad)):
        torch.testing.assert_close(mine, theirs)
    inputs = tuple(t.clone().requires_grad_() for t in (x, gamma, beta))
    assert torch.autograd.gradcheck(
        bp.as_autograd(bp.layernorm_forward, bp.layernorm_backward), inputs
    )


def test_rmsnorm_backward():
    x, gamma = rand(4, 8), rand(8, seed=1)
    dy = rand(4, 8, seed=2)
    out, cache = bp.rmsnorm_forward(x, gamma)
    xs, gs = x.clone().requires_grad_(), gamma.clone().requires_grad_()
    ref = F.rms_norm(xs, (8,), gs, eps=1e-6)
    torch.testing.assert_close(out, ref)
    ref.backward(dy)
    dx, dgamma = bp.rmsnorm_backward(dy, cache)
    torch.testing.assert_close(dx, xs.grad)
    torch.testing.assert_close(dgamma, gs.grad)
    inputs = (x.clone().requires_grad_(), gamma.clone().requires_grad_())
    assert torch.autograd.gradcheck(bp.as_autograd(bp.rmsnorm_forward, bp.rmsnorm_backward), inputs)


def test_layernorm_input_gradient_is_orthogonal_to_ones_and_xhat():
    """dx 与全 1 向量、x̂ 都正交：平移或缩放输入不改变 LayerNorm 的输出。"""
    x, dy = rand(8), rand(8, seed=1)
    gamma, beta = torch.ones(8, dtype=D), torch.zeros(8, dtype=D)
    out, cache = bp.layernorm_forward(x, gamma, beta, eps=0.0)
    dx, _, _ = bp.layernorm_backward(dy, cache)
    assert abs(dx.sum().item()) < 1e-12
    assert abs((dx * out).sum().item()) < 1e-12
