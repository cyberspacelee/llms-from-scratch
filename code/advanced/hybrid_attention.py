"""Gated delta recurrence with explicit affine-map and hybrid-layer references."""
import math
import torch


def delta_step(state, q, k, v, decay, update):
    decayed = decay * state
    state = decayed + update * torch.outer(v - decayed @ k, k)
    return state, state @ q


def verify():
    dtype = torch.float64
    state = torch.zeros(2, 2, dtype=dtype)
    q = k = torch.tensor([1., 0.], dtype=dtype)
    state, first = delta_step(state, q, k, torch.tensor([2., 1.], dtype=dtype), 1., .5)
    torch.testing.assert_close(first, torch.tensor([1., .5], dtype=dtype))
    q2 = k2 = torch.tensor([0., 1.], dtype=dtype)
    state, second = delta_step(state, q2, k2, torch.tensor([0., 2.], dtype=dtype), .5, 1.)
    torch.testing.assert_close(state, torch.tensor([[.5, 0.], [.25, 2.]], dtype=dtype))
    torch.testing.assert_close(second, torch.tensor([0., 2.], dtype=dtype))
    torch.manual_seed(17)
    q, k, v = [torch.randn(4, 2, dtype=dtype) for _ in range(3)]
    k = k / k.norm(dim=-1, keepdim=True)
    recurrent = torch.zeros(2, 2, dtype=dtype)
    explicit = recurrent.clone()
    outputs = []
    for qi, ki, vi in zip(q, k, v):
        recurrent, yi = delta_step(recurrent, qi, ki, vi, .8, .6)
        explicit = .8 * explicit @ (torch.eye(2, dtype=dtype) - .6 * torch.outer(ki, ki))
        explicit = explicit + .6 * torch.outer(vi, ki)
        torch.testing.assert_close(recurrent, explicit)
        outputs.append(yi)
    hidden = torch.stack(outputs)
    causal = torch.arange(4)[None, :] <= torch.arange(4)[:, None]
    hybrid = (hidden @ hidden.T / math.sqrt(2)).masked_fill(~causal, -torch.inf).softmax(-1) @ hidden
    assert hybrid.shape == (4, 2) and torch.isfinite(hybrid).all()
    assert recurrent.numel() == 4 and 2 * hidden.numel() == 16
    print("Hybrid: delta update = affine reference; recursive state 4 vs full-layer KV 16 elements")


if __name__ == "__main__":
    verify()
