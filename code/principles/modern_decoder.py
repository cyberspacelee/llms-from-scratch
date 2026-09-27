"""RMSNorm, SwiGLU, grouped-query attention, and shared embedding checks."""

import math

import torch
from torch.nn import functional as F


def rms_norm(x, weight, epsilon=1e-6):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + epsilon) * weight


def verify():
    torch.manual_seed(7)
    x = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64)
    scale = torch.ones(4, dtype=torch.float64)
    torch.testing.assert_close(rms_norm(x, scale), x / math.sqrt(7.5 + 1e-6))
    assert not torch.allclose(rms_norm(x + 1, scale), rms_norm(x, scale))
    gate, up, down = [torch.randn(*shape, dtype=torch.float64) for shape in [(6, 4), (6, 4), (4, 6)]]
    y = (F.silu(x @ gate.T) * (x @ up.T)) @ down.T
    assert y.shape == x.shape
    q = torch.randn(2, 4, 5, 2, dtype=torch.float64)
    k, v = [torch.randn(2, 2, 5, 2, dtype=torch.float64) for _ in range(2)]
    expanded_k, expanded_v = k.repeat_interleave(2, 1), v.repeat_interleave(2, 1)
    mask = torch.ones(5, 5, dtype=torch.bool).tril()
    result = (q @ expanded_k.transpose(-1, -2) / math.sqrt(2)).masked_fill(~mask, -torch.inf).softmax(-1) @ expanded_v
    for head in range(4):
        shared = head // 2
        ref = (q[:, head] @ k[:, shared].transpose(-1, -2) / math.sqrt(2)).masked_fill(~mask, -torch.inf).softmax(-1) @ v[:, shared]
        torch.testing.assert_close(result[:, head], ref)
    embedding = torch.randn(8, 4, dtype=torch.float64, requires_grad=True)
    ids = torch.tensor([1, 2, 1])
    hidden = embedding[ids]
    logits = hidden @ embedding.T
    logits.sum().backward()
    assert embedding.grad is not None and embedding.grad[0].abs().sum() > 0
    print("PASS: RMSNorm hand case, non-shift-invariance, SwiGLU shapes")
    print("PASS: GQA group mapping, tied embedding output gradients")


if __name__ == "__main__":
    verify()
