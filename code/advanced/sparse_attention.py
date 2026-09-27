"""Gathered attention equals dense attention with exactly the same sparse mask."""
import math
import torch


def selected_attention(q, k, v, ids):
    if ids.ndim != 1 or ids.numel() == 0 or ids.unique().numel() != ids.numel():
        raise ValueError("nonempty unique selected positions required")
    if ids.min() < 0 or ids.max() >= k.shape[0]:
        raise ValueError("selected position outside history")
    weights = (k[ids] @ q / math.sqrt(q.numel())).softmax(0)
    return weights @ v[ids]


def verify():
    torch.manual_seed(19)
    q = torch.randn(4, dtype=torch.float64)
    k, v = torch.randn(8, 4, dtype=q.dtype), torch.randn(8, 3, dtype=q.dtype)
    ids = torch.tensor([0, 3, 7])
    gathered = selected_attention(q, k, v, ids)
    mask = torch.zeros(8, dtype=torch.bool)
    mask[ids] = True
    scores = k @ q / math.sqrt(4)
    dense_masked = scores.masked_fill(~mask, -torch.inf).softmax(0) @ v
    torch.testing.assert_close(gathered, dense_masked)
    full = scores.softmax(0) @ v
    assert not torch.allclose(full, gathered)
    indexer_query, indexer_keys = torch.randn(2, dtype=q.dtype), torch.randn(8, 2, dtype=q.dtype)
    selected = (indexer_keys @ indexer_query).topk(3).indices
    assert selected.unique().numel() == 3
    assert torch.isfinite(selected_attention(q, k, v, selected)).all()
    try:
        selected_attention(q, k, v, torch.tensor([1, 1]))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate positions would double their probability mass")
    print("Sparse attention: gather = dense + same mask; differs from full attention")


if __name__ == "__main__":
    verify()
