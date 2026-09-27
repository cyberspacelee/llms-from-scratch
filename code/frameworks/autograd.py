"""CPU float64 checks for VJP, accumulation, graph lifetime and detach."""
import torch


def main():
    dtype = torch.float64
    x = torch.tensor(2., dtype=dtype, requires_grad=True)
    y = x * x + 3 * x
    assert x.is_leaf and not y.is_leaf and y.grad_fn is not None
    y.backward()
    assert x.grad.item() == 7
    (x * x + 3 * x).backward()
    assert x.grad.item() == 14
    x.grad = None
    (x * x + 3 * x).backward()
    assert x.grad.item() == 7
    try:
        y.backward()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Saved graph values unexpectedly survived the first backward')

    z = torch.tensor([2., -1.], dtype=dtype, requires_grad=True)
    output = torch.stack([z[0] ** 2 + z[1], z[0] * z[1]])
    seed = torch.tensor([2., 3.], dtype=dtype)
    gradient, = torch.autograd.grad(output, z, grad_outputs=seed)
    assert torch.equal(gradient, torch.tensor([5., 8.], dtype=dtype))
    assert z.grad is None
    epsilon = 1e-6
    def scalar(v):
        return 2 * (v[0] ** 2 + v[1]) + 3 * v[0] * v[1]
    for coordinate in range(2):
        delta = torch.zeros_like(z); delta[coordinate] = epsilon
        difference = (scalar(z.detach() + delta) - scalar(z.detach() - delta)) / (2 * epsilon)
        assert torch.allclose(difference, gradient[coordinate], atol=1e-8, rtol=1e-8)

    original = torch.tensor([1., 2.], dtype=dtype, requires_grad=True)
    clone = original.clone()
    assert clone.data_ptr() != original.data_ptr() and not clone.is_leaf
    clone.sum().backward()
    assert torch.equal(original.grad, torch.ones_like(original))
    detached = original.detach()
    assert detached.data_ptr() == original.data_ptr() and not detached.requires_grad
    independent = original.detach().clone().requires_grad_()
    independent.sum().backward()
    assert independent.is_leaf and torch.equal(independent.grad, torch.ones_like(independent))
    assert torch.equal(original.grad, torch.ones_like(original))

    parameter = torch.tensor(2., dtype=dtype, requires_grad=True)
    with torch.no_grad():
        result = parameter * 3
    assert not result.requires_grad and parameter.requires_grad
    with torch.inference_mode():
        inference = parameter * 3
    assert not inference.requires_grad
    model = torch.nn.Linear(1, 1, dtype=dtype).eval()
    assert model.weight.requires_grad and model(torch.ones(1, 1, dtype=dtype)).requires_grad

    value = torch.tensor(2., dtype=dtype, requires_grad=True)
    saved = value * value
    value.detach().add_(1)
    try:
        saved.backward()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Saved value mutation did not trigger version check')
    print(f'autograd: scalar/VJP, accumulation, finite differences, detach and modes OK (torch {torch.__version__})')


if __name__ == '__main__':
    main()
