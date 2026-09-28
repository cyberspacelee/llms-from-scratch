"""M6: verify one branched two-layer network and its batch gradient."""

import torch


def make_parameters():
    return [
        torch.tensor([[1., 1.], [2., -1.]], requires_grad=True),
        torch.zeros(2, requires_grad=True),
        torch.tensor([[1., -2.]], requires_grad=True),
        torch.tensor(1., requires_grad=True),
    ]


def losses(x, y, params):
    w1, b1, w2, b2 = params
    h = torch.relu(x @ w1.T + b1)
    prediction = (h @ w2.T).squeeze(-1) + b2 + h[:, 0]
    return 0.5 * (prediction - y).square()


def main():
    torch.set_default_dtype(torch.float64)
    inputs = torch.tensor([[1., 1.], [0., 1.], [1., 0.]])
    targets = torch.tensor([1., 2., 0.])
    params = make_parameters()

    first_input = inputs[:1].clone().requires_grad_()
    first = losses(first_input, targets[:1], params).sum()
    torch.testing.assert_close(first.detach(), torch.tensor(2.))
    first.backward()
    expected = [
        [[4., 4.], [-4., -4.]],
        [4., -4.],
        [[4., 2.]],
        2.,
    ]
    for param, gradient in zip(params, expected):
        torch.testing.assert_close(param.grad, torch.tensor(gradient))
    torch.testing.assert_close(first_input.grad, torch.tensor([[-4., 8.]]))

    for param in params:
        param.grad = None
    losses(inputs, targets, params).mean().backward()
    reference = [param.grad.clone() for param in params]

    for param in params:
        param.grad = None
    total = len(inputs)
    for part in (slice(0, 2), slice(2, 3)):
        count = len(inputs[part])
        (losses(inputs[part], targets[part], params).mean() * count / total).backward()
    for param, full_gradient in zip(params, reference):
        torch.testing.assert_close(param.grad, full_gradient)

    # A negative preactivation blocks only that sample's local ReLU path.
    for param in params:
        param.grad = None
    variant = losses(inputs[1:2], targets[1:2], params).sum()
    variant.backward()
    torch.testing.assert_close(params[0].grad, torch.tensor([[0., 2.], [0., 0.]]))
    print("math-06: branch, ReLU, batch, and microbatch gradients passed")


if __name__ == "__main__":
    main()
