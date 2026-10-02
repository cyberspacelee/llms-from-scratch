"""One eight-position example: full, masked, gathered, and reselected attention."""

import math

import torch


def gathered_attention(q, k, v, ids):
    if ids.ndim != 1 or ids.numel() == 0 or ids.unique().numel() != ids.numel():
        raise ValueError("selected positions must be nonempty and unique")
    if ids.min() < 0 or ids.max() >= k.shape[0]:
        raise ValueError("selected position outside visible history")
    scores = k[ids] @ q / math.sqrt(q.numel())
    return scores.softmax(0) @ v[ids]


def verify():
    dtype = torch.float64
    q = torch.ones(1, dtype=dtype)
    weights = torch.tensor([1, 1, 1, 2, 1, 1, 1, 4], dtype=dtype)
    k = weights.log().unsqueeze(1)
    v = torch.arange(0, 80, 10, dtype=dtype).unsqueeze(1)
    scores = k @ q
    full = scores.softmax(0) @ v
    torch.testing.assert_close(full, torch.tensor([130 / 3], dtype=dtype))

    indexer_query = torch.ones(1, dtype=dtype)
    indexer_keys = torch.tensor([8, 0, 0, 7, 0, 0, 0, 9], dtype=dtype).unsqueeze(1)
    ids = (indexer_keys @ indexer_query).topk(3).indices
    assert set(ids.tolist()) == {0, 3, 7}
    gathered = gathered_attention(q, k, v, ids)
    mask = torch.zeros(8, dtype=torch.bool)
    mask[ids] = True
    dense_masked = scores.masked_fill(~mask, -torch.inf).softmax(0) @ v
    torch.testing.assert_close(gathered, dense_masked)
    torch.testing.assert_close(gathered, torch.tensor([340 / 7], dtype=dtype))

    different = gathered_attention(q, k, v, torch.tensor([0, 4, 7]))
    torch.testing.assert_close(different, torch.tensor([160 / 3], dtype=dtype))
    assert not torch.allclose(full, gathered)
    assert not torch.allclose(different, gathered)

    omitted = (~mask).nonzero().flatten()
    omitted_mass = weights[omitted].sum() / weights.sum()
    omitted_output = gathered_attention(q, k, v, omitted)
    torch.testing.assert_close(omitted_mass, torch.tensor(5 / 12, dtype=dtype))
    torch.testing.assert_close(omitted_output, torch.tensor([36.0], dtype=dtype))
    torch.testing.assert_close(full, (1 - omitted_mass) * gathered + omitted_mass * omitted_output)

    for bad in (torch.tensor([3, 3]), torch.tensor([8]), torch.tensor([], dtype=torch.long)):
        try:
            gathered_attention(q, k, v, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid selection accepted: {bad.tolist()}")
    print(f"full={full.item():.6f}, selected={gathered.item():.6f}, changed={different.item():.6f}")


if __name__ == "__main__":
    verify()
