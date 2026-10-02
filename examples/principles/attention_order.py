"""Check the three-token order and position examples from P5."""

import torch


def attend(x, visible=None):
    scores = x @ x.T
    if visible is not None:
        scores = scores.masked_fill(~visible, float("-inf"))
    return scores.softmax(dim=-1) @ x


def verify():
    x = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
    order = torch.tensor([1, 0, 2])
    causal = torch.ones(3, 3, dtype=torch.bool).tril()
    position = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float64)
    close = torch.testing.assert_close

    plain = attend(x)
    close(
        plain[:, 0],
        torch.tensor([2.57521038, 2.85093709, 2.94797458], dtype=x.dtype),
        atol=1e-8,
        rtol=0,
    )
    close(attend(x[order]), plain[order])

    masked = attend(x, causal)
    close(
        masked[:, 0], torch.tensor([1.0, 1.88079708, 2.94797458], dtype=x.dtype), atol=1e-8, rtol=0
    )
    close(attend(x[order], causal[order][:, order]), masked[order])
    fixed_mask = attend(x[order], causal)
    close(
        fixed_mask[:, 0],
        torch.tensor([2.0, 1.73105858, 2.94797458], dtype=x.dtype),
        atol=1e-8,
        rtol=0,
    )
    assert not torch.allclose(fixed_mask, masked[order])

    with_position = attend(x + position)
    moved_tokens = attend(x[order] + position)
    close(
        with_position[:, 0],
        torch.tensor([2.87324212, 2.99752432, 2.99752432], dtype=x.dtype),
        atol=1e-8,
        rtol=0,
    )
    close(
        moved_tokens[:, 0],
        torch.tensor([2.78698604, 2.78698604, 2.90944300], dtype=x.dtype),
        atol=1e-8,
        rtol=0,
    )
    close(attend((x + position)[order]), with_position[order])
    assert not torch.allclose(moved_tokens, with_position[order])
    print("PASS: bidirectional permutation; fixed and moved causal mask; fixed position table")


if __name__ == "__main__":
    verify()
