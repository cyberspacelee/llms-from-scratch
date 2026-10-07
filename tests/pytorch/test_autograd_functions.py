import pytest
import torch
import torch.nn.functional as F

from llms_from_scratch.pytorch.autograd_functions import (
    clamp_grad_hook,
    fake_quantize,
    saved_tensor_bytes,
    silu,
)


def test_silu_matches_torch_and_gradcheck():
    x = torch.randn(16, dtype=torch.float64, requires_grad=True)
    torch.testing.assert_close(silu(x), F.silu(x))
    assert torch.autograd.gradcheck(silu, (x,))
    (g,) = torch.autograd.grad(silu(x).sum(), x)
    (ref,) = torch.autograd.grad(F.silu(x).sum(), x)
    torch.testing.assert_close(g, ref)


def test_custom_silu_saves_less_than_composite():
    x = torch.randn(1024, requires_grad=True)
    assert saved_tensor_bytes(silu, x) == 1024 * 4  # 只存输入 x
    composite = saved_tensor_bytes(lambda t: t * torch.sigmoid(t), x)
    assert composite == 2 * 1024 * 4  # x 与 sigmoid(x)


def test_straight_through_estimator():
    x = torch.tensor([0.12, 0.26, -0.31], requires_grad=True)
    y = fake_quantize(x, 0.1)
    torch.testing.assert_close(y.detach(), torch.tensor([0.1, 0.3, -0.3]))
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones(3))
    z = torch.tensor([0.12], requires_grad=True)
    torch.round(z / 0.1).sum().backward()
    assert z.grad.item() == 0.0  # 不用 STE 时 round 的梯度处处为 0


def test_grad_accumulates_and_leaf_rules():
    w = torch.tensor(2.0, requires_grad=True)
    (w * 3).backward()
    (w * 3).backward()
    assert w.grad.item() == 6.0  # 两次 backward 累加
    h = w * 5
    assert not h.is_leaf and h.grad_fn is not None
    h.retain_grad()
    (h**2).backward()
    assert h.grad.item() == 20.0  # 中间张量只有 retain_grad 才保留梯度
    with torch.no_grad():
        assert not (w * 2).requires_grad
    assert not w.detach().requires_grad


def test_graph_freed_after_backward():
    x = torch.randn(3, requires_grad=True)
    y = (x.exp() * 2).sum()
    y.backward()
    with pytest.raises(RuntimeError):
        y.backward()  # 保存的激活已在第一次反向后释放
    y2 = (x.exp() * 2).sum()
    y2.backward(retain_graph=True)
    y2.backward()


def test_tensor_hook_modifies_gradient():
    x = torch.tensor([1.0, -4.0, 9.0], requires_grad=True)
    h = x * 1.0
    h.register_hook(clamp_grad_hook(2.0))
    (h * torch.tensor([1.0, 5.0, -7.0])).sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([1.0, 2.0, -2.0]))


def test_no_grad_saves_nothing():
    lin = torch.nn.Linear(64, 64)
    x = torch.randn(8, 64)
    params = list(lin.parameters())
    assert saved_tensor_bytes(lambda t: torch.tanh(lin(t)), x, exclude=params) > 0
    with torch.no_grad():
        assert saved_tensor_bytes(lambda t: torch.tanh(lin(t)), x, exclude=params) == 0
