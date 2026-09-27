"""Sparse expert dispatch, dense reference, and routing gradients (CPU)."""
import torch
from torch import nn


def route(logits, top_k):
    if logits.ndim != 2 or not 1 <= top_k <= logits.shape[-1]:
        raise ValueError("expected token-by-expert logits and valid top_k")
    selected, ids = logits.topk(top_k, dim=-1)
    return ids, selected.softmax(-1)


def dispatch(x, experts, ids, weights):
    output = torch.zeros_like(x)
    for expert_id, expert in enumerate(experts):
        rows, slots = (ids == expert_id).nonzero(as_tuple=True)
        if rows.numel():
            output.index_add_(0, rows, expert(x[rows]) * weights[rows, slots, None])
    return output


def verify():
    torch.manual_seed(7)
    x = torch.tensor([[1., 2.], [2., -1.], [-1., 3.]], dtype=torch.float64)
    experts = nn.ModuleList([nn.Linear(2, 2, bias=False).double() for _ in range(4)])
    with torch.no_grad():
        for e, expert in enumerate(experts):
            expert.weight.copy_(torch.eye(2, dtype=x.dtype) * (e + 1))
    logits = torch.tensor([[4., 3., 1., 0.], [0., 1., 4., 3.], [4., 0., 3., 1.]],
                          dtype=x.dtype, requires_grad=True)
    ids, weights = route(logits, 2)
    actual = dispatch(x, experts, ids, weights)
    all_outputs = torch.stack([expert(x) for expert in experts], dim=1)
    selected = all_outputs.gather(1, ids[..., None].expand(-1, -1, 2))
    expected = (selected * weights[..., None]).sum(1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(weights.sum(-1), torch.ones(3, dtype=x.dtype))
    counts = torch.bincount(ids.flatten(), minlength=4)
    assert counts.tolist() == [2, 1, 2, 1]
    actual.square().sum().backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert (logits.grad.gather(1, ids).abs().sum() > 0).item()
    assert sum(p.numel() for p in experts.parameters()) == 16
    print("MoE: sparse dispatch = same-route dense reference; counts", counts.tolist())


if __name__ == "__main__":
    verify()
