"""Three-token gated delta layer followed by one causal attention layer."""

import math

import torch


def delta_step(state, q, k, v, decay, update):
    decayed = decay * state
    next_state = decayed + update * torch.outer(v - decayed @ k, k)
    return next_state, next_state @ q


def verify():
    dtype = torch.float64
    e1 = torch.tensor([1.0, 0.0], dtype=dtype)
    e2 = torch.tensor([0.0, 1.0], dtype=dtype)
    steps = [
        (e1, torch.tensor([2.0, 1.0], dtype=dtype), 1.0, 1.0),
        (e1, torch.tensor([0.0, 2.0], dtype=dtype), 0.5, 1.0),
        (e2, torch.tensor([1.0, 1.0], dtype=dtype), 1.0, 0.5),
    ]
    expected_states = [
        [[2.0, 0.0], [1.0, 0.0]],
        [[0.0, 0.0], [2.0, 0.0]],
        [[0.0, 0.5], [2.0, 0.5]],
    ]
    expected_outputs = [[2.0, 1.0], [0.0, 2.0], [0.5, 0.5]]

    state = torch.zeros(2, 2, dtype=dtype)
    outputs = []
    for (key, value, decay, update), expected_state, expected_output in zip(
        steps, expected_states, expected_outputs
    ):
        previous = state
        state, output = delta_step(previous, key, key, value, decay, update)
        transition = decay * (torch.eye(2, dtype=dtype) - update * torch.outer(key, key))
        affine_reference = previous @ transition + update * torch.outer(value, key)
        torch.testing.assert_close(state, affine_reference)
        torch.testing.assert_close(state, torch.tensor(expected_state, dtype=dtype))
        torch.testing.assert_close(output, torch.tensor(expected_output, dtype=dtype))
        outputs.append(output)

    hidden = torch.stack(outputs)
    keys = hidden[:, :1]
    queries = keys * math.log(16)
    scores = queries @ keys.T
    causal = torch.arange(3)[None, :] <= torch.arange(3)[:, None]
    attention = scores.masked_fill(~causal, -torch.inf).softmax(-1) @ hidden
    torch.testing.assert_close(attention, torch.tensor(
        [[2.0, 1.0], [1.0, 1.5], [33.0 / 19.0, 1.0]], dtype=dtype
    ))
    assert state.numel() == 4 and 3 * (1 + 2) == 9
    print("delta outputs: (2, 1), (0, 2), (0.5, 0.5)")
    print("hybrid final output: (33/19, 1); cached elements: 4 + 9")


if __name__ == "__main__":
    verify()
