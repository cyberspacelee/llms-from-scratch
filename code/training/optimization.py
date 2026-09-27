"""AdamW arithmetic and valid-token gradient accumulation."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def saved_activation_elements(run, parameters):
    """Elements autograd saves for backward during run(), excluding parameter storage."""
    weights = {p.data_ptr() for p in parameters}
    count = [0]

    def pack(tensor):
        if tensor.data_ptr() not in weights:
            count[0] += tensor.numel()
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = run()
    return loss, count[0]


def verify_scale_precision_memory():
    dtype = torch.float64
    generator = torch.Generator().manual_seed(3)
    # Fan-in initialization: Var(Wx) = d_in Var(w) E[x^2], so std 1/sqrt(d_in) keeps variance near 1.
    x = torch.randn(4096, 256, dtype=dtype, generator=generator)
    scaled = torch.randn(256, 256, dtype=dtype, generator=generator) / 256 ** 0.5
    assert abs((x @ scaled.T).var().item() - 1) < 0.05
    assert abs((x @ (scaled * 256 ** 0.5).T).var().item() / 256 - 1) < 0.05
    # Residual stream: 24 independent unit-variance writes add variance; 1/sqrt(24) scaling bounds it.
    writes = torch.randn(24, 4096, 64, dtype=dtype, generator=generator)
    stream = torch.randn(4096, 64, dtype=dtype, generator=generator)
    assert abs((stream + writes.sum(0)).var().item() - 25) < 1.0
    assert abs((stream + writes.sum(0) / 24 ** 0.5).var().item() - 2) < 0.1

    # FP16 flushes 1e-8 to zero; scaling by 1024 survives, BF16 keeps the range but not 1 + 2^-8.
    assert torch.tensor(1e-8, dtype=torch.float16).item() == 0
    unscaled = torch.tensor(1e-8 * 1024, dtype=torch.float16).float().item() / 1024
    assert abs(unscaled - 1e-8) / 1e-8 < 2e-3
    assert torch.tensor(1e-8, dtype=torch.bfloat16).item() > 0
    assert torch.tensor(1 + 2 ** -8, dtype=torch.bfloat16).item() == 1
    assert torch.tensor(1 + 2 ** -8, dtype=torch.float16).item() == 1 + 2 ** -8

    # Checkpointing keeps block inputs only and recomputes the rest; gradients are unchanged.
    torch.manual_seed(5)
    blocks = torch.nn.ModuleList(torch.nn.Sequential(
        torch.nn.Linear(8, 32), torch.nn.GELU(), torch.nn.Linear(32, 8)) for _ in range(4)).double()
    inputs = torch.randn(16, 8, dtype=dtype)

    def run(use_checkpoint):
        h = inputs
        for block in blocks:
            h = h + (checkpoint(block, h, use_reentrant=False) if use_checkpoint else block(h))
        return h.square().sum()

    results = []
    for use_checkpoint in (False, True):
        blocks.zero_grad(set_to_none=True)
        loss, saved = saved_activation_elements(lambda: run(use_checkpoint), blocks.parameters())
        loss.backward()
        results.append((saved, [p.grad.clone() for p in blocks.parameters()]))
    # Per block: Linear input 16x8, GELU input 16x32, Linear input 16x32; plus the final square.
    assert results[0][0] == 4 * (128 + 512 + 512) + 128 == 4736
    assert results[1][0] == 4 * 128 + 128 == 640
    for plain, recomputed in zip(results[0][1], results[1][1]):
        torch.testing.assert_close(plain, recomputed, atol=1e-12, rtol=1e-12)
    print("optimization: fan-in and residual variance, FP16/BF16 range, checkpoint 4736 -> 640 saved")


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
    verify_scale_precision_memory()
