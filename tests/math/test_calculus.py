import math

import torch

from llms_from_scratch.math.calculus import (
    central_difference,
    descend_quadratic,
    directional_derivative,
    fit_line,
    forward_difference,
    line_gradient,
    numerical_gradient,
)


def test_central_difference_is_more_accurate():
    exact = math.cos(1.0)
    fwd = abs(forward_difference(math.sin, 1.0, 1e-3) - exact)
    ctr = abs(central_difference(math.sin, 1.0, 1e-3) - exact)
    assert fwd > 1e-4 and ctr < 1e-6  # O(h) 对 O(h²)


def test_numerical_gradient_of_quadratic():
    a = torch.tensor([[3.0, 1.0], [1.0, 2.0]], dtype=torch.float64)

    def f(x):
        return 0.5 * x @ a @ x

    x = torch.tensor([1.0, -2.0], dtype=torch.float64)
    torch.testing.assert_close(numerical_gradient(f, x), a @ x)


def test_gradient_is_steepest_direction():
    def f(x):
        return x[0] ** 2 + 3 * x[1] ** 2

    x = torch.tensor([1.0, 1.0], dtype=torch.float64)
    grad = numerical_gradient(f, x)
    norm = grad.norm().item()
    for angle in torch.linspace(0, 2 * math.pi, 73, dtype=torch.float64):
        u = torch.stack([torch.cos(angle), torch.sin(angle)])
        d = directional_derivative(f, x, u)
        assert d <= norm + 1e-6
        assert math.isclose(d, (grad @ u).item(), abs_tol=1e-6)  # D_u f = ∇f·u
    assert math.isclose(directional_derivative(f, x, grad), norm, rel_tol=1e-6)


def test_line_gradient_matches_autograd():
    x = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)
    y = torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float64)
    w = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(-0.2, dtype=torch.float64, requires_grad=True)
    (0.5 * ((w * x + b - y) ** 2).mean()).backward()
    gw, gb = line_gradient(0.5, -0.2, x, y)
    assert math.isclose(gw, w.grad.item()) and math.isclose(gb, b.grad.item())


def test_fit_line_recovers_parameters():
    x = torch.linspace(-1, 1, 21, dtype=torch.float64)
    y = 2 * x + 1
    fit = fit_line(x, y, lr=0.5, steps=200)
    assert abs(fit.w - 2) < 1e-4 and abs(fit.b - 1) < 1e-4
    assert all(a >= b for a, b in zip(fit.losses, fit.losses[1:]))


def test_learning_rate_threshold():
    c = 4.0
    assert abs(descend_quadratic(c, 0.4, 1.0, 50)[-1]) < 1e-6  # ηc = 1.6 < 2：振荡收敛
    assert descend_quadratic(c, 0.25, 1.0, 1)[-1] == 0.0  # η = 1/c：一步到位
    assert abs(descend_quadratic(c, 0.6, 1.0, 50)[-1]) > 1e6  # ηc = 2.4 > 2：发散
