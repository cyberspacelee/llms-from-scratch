"""Numerical checks for the four-position attention example in P3."""

import math

import torch


def causal_attention(q, k, v):
    """Q/K/V have shape [B, heads, T, head_width]."""
    length = q.shape[-2]
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    visible = torch.ones(length, length, dtype=torch.bool).tril()
    weights = scores.masked_fill(~visible, -torch.inf).softmax(-1)
    return weights, weights @ v


def verify():
    torch.set_num_threads(1)
    dtype = torch.float64
    x = torch.eye(4, dtype=dtype)  # BOS, 猫, 吃, 鱼: one position per row.
    wq = torch.tensor([[math.sqrt(2), 0]] * 4, dtype=dtype)
    wk = torch.tensor([[0.2, 0], [1.1, 0], [-0.4, 0], [0.7, 0]], dtype=dtype)
    wv = torch.tensor([[1, 0], [0, 2], [0, 3], [2, 0]], dtype=dtype)
    q, k, v = (x @ w for w in (wq, wk, wv))
    weights, output = causal_attention(q[None, None], k[None, None], v[None, None])
    weights, output = weights[0, 0], output[0, 0]

    for i in range(4):
        scores = k[: i + 1] @ q[i] / math.sqrt(2)
        expected_weights = scores.softmax(0)
        torch.testing.assert_close(weights[i, : i + 1], expected_weights)
        assert torch.count_nonzero(weights[i, i + 1 :]) == 0
        torch.testing.assert_close(output[i], expected_weights @ v[: i + 1])
    torch.testing.assert_close(
        output[1], torch.tensor([0.2890504974, 1.4218990052], dtype=dtype), atol=1e-9, rtol=0
    )
    torch.testing.assert_close(
        output[2], torch.tensor([0.249475, 1.638], dtype=dtype), atol=1e-3, rtol=0
    )

    changed = x.clone()
    changed[3] = 0  # A future input must not affect earlier outputs.
    _, changed_output = causal_attention(
        (changed @ wq)[None, None], (changed @ wk)[None, None], (changed @ wv)[None, None]
    )
    torch.testing.assert_close(changed_output[0, 0, :3], output[:3])

    other = x.flip(0)
    batch = torch.stack((x, other))
    batched = causal_attention((batch @ wq)[:, None], (batch @ wk)[:, None], (batch @ wv)[:, None])[
        1
    ]
    for b in range(2):
        separate = causal_attention(
            (batch[b] @ wq)[None, None], (batch[b] @ wk)[None, None], (batch[b] @ wv)[None, None]
        )[1]
        torch.testing.assert_close(batched[b : b + 1], separate)

    # The second head uses another projection on the same four input rows.
    wq2 = torch.tensor([[1, 0]] * 4, dtype=dtype)
    wk2 = torch.tensor([[1, 0], [0, 1], [0, 1], [0, 1]], dtype=dtype)
    wv2 = wv.flip(0)
    projections = [torch.cat((a, b), -1) for a, b in ((wq, wq2), (wk, wk2), (wv, wv2))]
    qh, kh, vh = [(batch @ w).reshape(2, 4, 2, 2).transpose(1, 2) for w in projections]
    _, heads_output = causal_attention(qh, kh, vh)
    for b in range(2):
        for h in range(2):
            _, one_head = causal_attention(
                qh[b : b + 1, h : h + 1],
                kh[b : b + 1, h : h + 1],
                vh[b : b + 1, h : h + 1],
            )
            torch.testing.assert_close(heads_output[b : b + 1, h : h + 1], one_head)
    merged = heads_output.transpose(1, 2).contiguous().reshape(2, 4, 4)
    torch.testing.assert_close(merged[0, 1, :2], output[1])
    assert not torch.equal((batch @ projections[0]).reshape(2, 2, 4, 2), qh)
    print("PASS: four-position hand calculation, causal mask, batch isolation, head axes")


if __name__ == "__main__":
    verify()
