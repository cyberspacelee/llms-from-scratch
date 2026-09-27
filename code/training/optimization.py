"""AdamW arithmetic and valid-token gradient accumulation."""

import torch
import torch.nn.functional as F


def verify():
    torch.manual_seed(11)
    dtype = torch.float64
    initial = torch.tensor([1.0, -2.0], dtype=dtype)
    manual = initial.clone()
    parameter = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.AdamW([parameter], lr=0.1, betas=(0.9, 0.99),
                                 eps=1e-8, weight_decay=0.2, foreach=False)
    first, second = torch.zeros_like(manual), torch.zeros_like(manual)
    for step, gradient in enumerate(([0.5, -0.25], [-0.2, 0.4]), start=1):
        g = torch.tensor(gradient, dtype=dtype)
        first = 0.9 * first + 0.1 * g
        second = 0.99 * second + 0.01 * g.square()
        first_hat = first / (1 - 0.9 ** step)
        second_hat = second / (1 - 0.99 ** step)
        manual = (1 - 0.1 * 0.2) * manual - 0.1 * first_hat / (second_hat.sqrt() + 1e-8)
        parameter.grad = g.clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.testing.assert_close(parameter, manual, atol=1e-12, rtol=1e-12)
        print("AdamW step", step, manual.tolist())

    x = torch.randn(5, 3, dtype=dtype)
    y = torch.tensor([0, 1, 0, 1, 1])
    weight = torch.randn(2, 3, dtype=dtype, requires_grad=True)
    F.cross_entropy(x @ weight.T, y).backward()
    complete_gradient = weight.grad.clone()
    weight.grad = None
    for begin, end in [(0, 2), (2, 5)]:
        # Sum each micro-batch, divide by the complete valid-token count.
        (F.cross_entropy(x[begin:end] @ weight.T, y[begin:end], reduction="sum") / 5).backward()
    torch.testing.assert_close(weight.grad, complete_gradient, atol=1e-12, rtol=1e-12)
    weight.grad = None
    for begin, end in [(0, 2), (2, 5)]:
        (F.cross_entropy(x[begin:end] @ weight.T, y[begin:end]) / 2).backward()
    assert not torch.allclose(weight.grad, complete_gradient)
    p = torch.nn.Parameter(torch.zeros(2, dtype=dtype))
    p.grad = torch.tensor([3.0, 4.0], dtype=dtype)
    norm = torch.nn.utils.clip_grad_norm_([p], max_norm=2.0)
    assert norm.item() == 5.0
    torch.testing.assert_close(p.grad, torch.tensor([1.2, 1.6], dtype=dtype), atol=1e-6, rtol=1e-6)
    print("optimization: unequal micro-batches reproduce complete gradient; clipping verified")


if __name__ == "__main__":
    verify()
