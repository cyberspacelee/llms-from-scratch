"""One four-target update, unequal micro-batches, and optimizer state."""

import torch
from torch.utils.checkpoint import checkpoint

DTYPE = torch.float64
TARGETS = torch.tensor([[0.0, -2.0], [0.0, -2.0], [0.0, -2.0], [2.0, -1.0]], dtype=DTYPE)
INITIAL = torch.tensor([1.0, -2.0], dtype=DTYPE)


def token_losses(weight, targets):
    return 0.5 * (weight - targets).square().sum(dim=1)


def manual_adamw(weight, gradient, first, second, step):
    first = 0.9 * first + 0.1 * gradient
    second = 0.99 * second + 0.01 * gradient.square()
    first_hat = first / (1 - 0.9**step)
    second_hat = second / (1 - 0.99**step)
    weight = 0.98 * weight - 0.1 * first_hat / (second_hat.sqrt() + 1e-8)
    return weight, first, second


def verify_update():
    weight = torch.nn.Parameter(INITIAL.clone())
    optimizer = torch.optim.AdamW(
        [weight], lr=0.1, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.2, foreach=False
    )
    manual = INITIAL.clone()
    first = torch.zeros_like(manual)
    second = torch.zeros_like(manual)

    for step in (1, 2):
        optimizer.zero_grad(set_to_none=True)
        losses = token_losses(weight, TARGETS)
        full_loss = losses.mean()
        full_loss.backward()
        full_gradient = weight.grad.clone()
        expected_gradient = weight.detach() - TARGETS.mean(dim=0)
        torch.testing.assert_close(full_gradient, expected_gradient)
        if step == 1:
            assert abs(full_loss.item() - 0.625) < 1e-12
            torch.testing.assert_close(full_gradient, torch.tensor([0.5, -0.25], dtype=DTYPE))
            weight.grad = None
            for selected in (slice(0, 1), slice(1, 4)):
                (token_losses(weight, TARGETS[selected]).sum() / 4).backward()
            torch.testing.assert_close(weight.grad, full_gradient)
            weight.grad = None
            for selected in (slice(0, 1), slice(1, 4)):
                (token_losses(weight, TARGETS[selected]).mean() / 2).backward()
            torch.testing.assert_close(weight.grad, torch.tensor([2 / 3, -1 / 6], dtype=DTYPE))
            assert not torch.allclose(weight.grad, full_gradient)
            weight.grad = full_gradient.clone()

        manual, first, second = manual_adamw(manual, full_gradient, first, second, step)
        if step == 2:
            fresh_weight = torch.nn.Parameter(weight.detach().clone())
            fresh_optimizer = torch.optim.AdamW(
                [fresh_weight], lr=0.1, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.2, foreach=False
            )
            fresh_weight.grad = full_gradient.clone()
            fresh_optimizer.step()
        optimizer.step()
        torch.testing.assert_close(weight, manual, atol=1e-12, rtol=1e-12)
        if step == 1:
            torch.testing.assert_close(
                weight, torch.tensor([0.88, -1.86], dtype=DTYPE), atol=1e-8, rtol=1e-8
            )
            assert abs(token_losses(weight, TARGETS).mean().item() - 0.547) < 1e-8
        else:
            assert not torch.allclose(weight, fresh_weight)
        print(
            f"AdamW step {step}: {weight.detach().tolist()}, mean loss {token_losses(weight, TARGETS).mean().item():.6f}"
        )


def verify_clipping_and_precision():
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=DTYPE))
    parameter.grad = torch.tensor([0.5, -0.25], dtype=DTYPE)
    old_norm = torch.nn.utils.clip_grad_norm_([parameter], max_norm=0.25)
    assert abs(old_norm.item() - 5**0.5 / 4) < 1e-12
    torch.testing.assert_close(
        parameter.grad,
        torch.tensor([0.5 / 5**0.5, -0.25 / 5**0.5], dtype=DTYPE),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.tensor(1e-8, dtype=torch.float16).item() == 0
    assert torch.tensor(1e-8 * 1024, dtype=torch.float16).item() > 0
    assert torch.tensor(1 + 2**-8, dtype=torch.bfloat16).item() == 1


def saved_activation_elements(run, parameters):
    weights = {p.data_ptr() for p in parameters}
    count = [0]

    def pack(tensor):
        if tensor.data_ptr() not in weights:
            count[0] += tensor.numel()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = run()
    return loss, count[0]


def verify_checkpoint():
    torch.manual_seed(5)
    block = torch.nn.Sequential(
        torch.nn.Linear(8, 32), torch.nn.GELU(), torch.nn.Linear(32, 8)
    ).double()
    inputs = torch.randn(16, 8, dtype=DTYPE)
    outcomes = []
    for recompute in (False, True):
        block.zero_grad(set_to_none=True)

        def run(recompute=recompute):
            branch = checkpoint(block, inputs, use_reentrant=False) if recompute else block(inputs)
            return (inputs + branch).square().sum()

        loss, saved = saved_activation_elements(run, block.parameters())
        loss.backward()
        outcomes.append((saved, [p.grad.clone() for p in block.parameters()]))
    assert outcomes[1][0] < outcomes[0][0]
    for plain, recomputed in zip(outcomes[0][1], outcomes[1][1]):
        torch.testing.assert_close(plain, recomputed, atol=1e-12, rtol=1e-12)
    print(f"checkpoint saved elements: {outcomes[0][0]} -> {outcomes[1][0]}")


if __name__ == "__main__":
    verify_update()
    verify_clipping_and_precision()
    verify_checkpoint()
