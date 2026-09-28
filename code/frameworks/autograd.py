"""One branched loss checks PyTorch gradients, graph state and storage."""
import torch


DTYPE = torch.float64
X = torch.tensor([2., -1.], dtype=DTYPE)


def forward(w, b, x=X):
    u = w * x + b
    s = u[0] ** 2 + u[0] * u[1]
    return u, 0.5 * (s - 2) ** 2


def main():
    w = torch.tensor([1., 2.], dtype=DTYPE, requires_grad=True)
    b = torch.tensor(1., dtype=DTYPE, requires_grad=True)
    u, loss = forward(w, b)
    assert u.tolist() == [3., -1.] and loss.item() == 8.
    assert w.is_leaf and b.is_leaf and not u.is_leaf
    assert w.grad is None and b.grad is None
    u.retain_grad()
    loss.backward()
    assert u.grad.tolist() == [20., 12.]
    assert w.grad.tolist() == [40., -12.] and b.grad.item() == 32.
    assert X.grad is None

    _, second_loss = forward(w, b)
    second_loss.backward()
    assert w.grad.tolist() == [80., -24.] and b.grad.item() == 64.
    w.grad = b.grad = None
    _, third_loss = forward(w, b)
    third_loss.backward()
    assert w.grad.tolist() == [40., -12.] and b.grad.item() == 32.
    try:
        loss.backward()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Reusing this released graph should fail')

    w.grad = b.grad = None
    uv, _ = forward(w, b)
    seed = torch.tensor([20., 12.], dtype=DTYPE)
    gw, gb = torch.autograd.grad(uv, (w, b), grad_outputs=seed)
    assert gw.tolist() == [40., -12.] and gb.item() == 32.
    assert w.grad is None and b.grad is None
    _, fresh_loss = forward(w, b)
    gw, gb = torch.autograd.grad(fresh_loss, (w, b))
    assert gw.tolist() == [40., -12.] and gb.item() == 32.
    assert w.grad is None and b.grad is None

    epsilon = 1e-6
    for coordinate, expected in enumerate([40., -12.]):
        delta = torch.zeros_like(w)
        delta[coordinate] = epsilon
        plus = forward(w.detach() + delta, b.detach())[1].item()
        minus = forward(w.detach() - delta, b.detach())[1].item()
        assert abs((plus - minus) / (2 * epsilon) - expected) < 1e-7
    plus = forward(w.detach(), b.detach() + epsilon)[1].item()
    minus = forward(w.detach(), b.detach() - epsilon)[1].item()
    assert abs((plus - minus) / (2 * epsilon) - 32.) < 1e-7

    copy = w.clone()
    assert copy.data_ptr() != w.data_ptr() and copy.grad_fn is not None
    copy.sum().backward()
    assert w.grad.tolist() == [1., 1.]
    shared = w.detach()
    assert shared.data_ptr() == w.data_ptr() and not shared.requires_grad
    independent = w.detach().clone().requires_grad_()
    independent.sum().backward()
    assert independent.is_leaf and independent.grad.tolist() == [1., 1.]
    assert w.grad.tolist() == [1., 1.]

    changed = torch.tensor([1., 2.], dtype=DTYPE, requires_grad=True)
    bias = torch.tensor(1., dtype=DTYPE, requires_grad=True)
    saved_u, saved_loss = forward(changed, bias)
    saved_u.detach().add_(1)
    try:
        saved_loss.backward()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Mutating saved u should fail version checking')

    with torch.no_grad():
        no_history = forward(w, b)[1]
    assert not no_history.requires_grad and w.requires_grad
    with torch.inference_mode():
        inference = forward(w, b)[1]
    assert not inference.requires_grad
    module = torch.nn.Linear(1, 1, dtype=DTYPE).eval()
    assert module.weight.requires_grad
    assert module(torch.ones(1, 1, dtype=DTYPE)).requires_grad
    print(f'autograd: branched loss, VJP, state and modes OK (torch {torch.__version__})')


if __name__ == '__main__':
    main()
