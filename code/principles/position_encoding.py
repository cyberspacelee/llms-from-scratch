"""Small, explicit positional encodings. Run this file for numerical proofs."""

import math

import torch
from torch import nn


def angles(positions, width, base=10000.0):
    if width <= 0 or width % 2:
        raise ValueError("width must be positive and even")
    if base <= 0:
        raise ValueError("base must be positive")
    dtype = torch.float64 if positions.dtype == torch.float64 else torch.float32
    frequencies = base ** (-torch.arange(0, width, 2, device=positions.device,
                                         dtype=dtype) / width)
    return positions.to(dtype)[..., None] * frequencies


def sinusoidal_pe(positions, width):
    phase = angles(positions, width)
    # Stack each sin/cos pair, then flatten only the pair dimensions.
    return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)


class OriginalTransformerInput(nn.Module):
    def __init__(self, vocab_size, width, dropout=0.1):
        super().__init__()
        if width <= 0 or width % 2:
            raise ValueError("width must be positive and even")
        self.embedding = nn.Embedding(vocab_size, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids):
        x = self.embedding(token_ids)  # [B, T, d_model]
        positions = torch.arange(x.shape[1], device=x.device)
        pe = sinusoidal_pe(positions, x.shape[-1]).to(x.dtype)
        return self.dropout(math.sqrt(x.shape[-1]) * x + pe)


def apply_rope(x, positions, base=10000.0):
    """x: [B,H,T,D]; positions: [T]. Adjacent-pair convention, no padding."""
    if x.ndim != 4 or positions.ndim != 1 or x.shape[-2] != positions.numel():
        raise ValueError("expected x=[B,H,T,D] and positions=[T]")
    if not x.is_floating_point():
        raise ValueError("x must be floating point")
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    phase = angles(positions.to(device=x.device, dtype=dtype), x.shape[-1], base)
    a, b = x.to(dtype)[..., 0::2], x.to(dtype)[..., 1::2]
    c, s = phase.cos(), phase.sin()  # [T,D/2] broadcasts over B,H
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2).to(x.dtype)


def attention(q, k, v, mask=None):
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1) @ v


def verify():
    torch.manual_seed(7)
    dtype = torch.float64
    close = torch.testing.assert_close
    x = torch.randn(5, 8, dtype=dtype)
    wq, wk, wv = [torch.randn(8, 4, dtype=dtype) for _ in range(3)]
    order = torch.tensor([2, 0, 4, 1, 3])
    f = lambda z: attention(z @ wq, z @ wk, z @ wv)
    close(f(x[order]), f(x)[order])

    p = torch.arange(5, dtype=dtype)
    pe = sinusoidal_pe(p, 8)
    phi = angles(torch.tensor(3.0, dtype=dtype), 8)
    sin, cos = pe[:, 0::2], pe[:, 1::2]
    shifted = torch.stack((sin * phi.cos() + cos * phi.sin(),
                           -sin * phi.sin() + cos * phi.cos()), -1).flatten(-2)
    close(shifted, sinusoidal_pe(p + 3, 8))
    close(pe @ pe.T, angles(p[:, None] - p[None, :], 8).cos().sum(-1))
    close(pe.square().sum(-1), torch.full((5,), 4.0, dtype=dtype))

    q = torch.randn(2, 3, 5, 8, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q)
    qr, kr = apply_rope(q, p), apply_rope(k, p)
    scores = qr @ kr.transpose(-1, -2)
    close(scores, apply_rope(q, p + 9) @ apply_rope(k, p + 9).transpose(-1, -2))
    close(qr.square().sum(-1), q.square().sum(-1))
    scores.square().mean().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert OriginalTransformerInput(20, 8)(torch.tensor([[1, 2, 3]])).shape == (1, 3, 8)
    for width in (0, 3):
        try:
            sinusoidal_pe(p, width)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid width accepted")
    print("PASS: permutation equivariance; sin/cos translation, kernel and norm")
    print("PASS: RoPE common shift, norm, gradients; embedding shape; invalid widths")


if __name__ == "__main__":
    verify()
