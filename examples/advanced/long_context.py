"""RoPE interpolation and equivalent full-window/ring-cache attention."""

import math

import torch


def rotate(x, positions, frequency):
    angles = positions.to(dtype=x.dtype) * frequency
    return torch.stack(
        (
            x[..., 0] * angles.cos() - x[..., 1] * angles.sin(),
            x[..., 0] * angles.sin() + x[..., 1] * angles.cos(),
        ),
        dim=-1,
    )


def verify():
    torch.manual_seed(3)
    length, window, width = 9, 3, 2
    positions = torch.arange(length)
    frequency = math.pi / 8  # pi/4 after position interpolation by 2
    q_raw, k_raw, v = [torch.randn(length, width, dtype=torch.float64) for _ in range(3)]
    q, k = rotate(q_raw, positions, frequency), rotate(k_raw, positions, frequency)

    visible = (positions[None, :] <= positions[:, None]) & (
        positions[None, :] > positions[:, None] - window
    )
    reference = (q @ k.T / math.sqrt(width)).masked_fill(~visible, -torch.inf).softmax(-1) @ v

    # region ring_cache
    ring_k, ring_v = (
        torch.zeros(window, width, dtype=q.dtype),
        torch.zeros(window, width, dtype=q.dtype),
    )
    tags = torch.full((window,), -1)
    outputs = []
    for position in range(length):
        slot = position % window
        ring_k[slot], ring_v[slot], tags[slot] = k[position], v[position], position
        expected = torch.arange(max(0, position - window + 1), position + 1)
        slots = expected % window
        assert torch.equal(tags[slots], expected)
        weights = (ring_k[slots] @ q[position] / math.sqrt(width)).softmax(0)
        outputs.append(weights @ ring_v[slots])
    torch.testing.assert_close(torch.stack(outputs), reference)

    # endregion ring_cache
    numbered_values = torch.arange(length, dtype=torch.float64)
    uniform_scores = torch.zeros(length, length, dtype=torch.float64)
    full_mask = positions[None, :] <= positions[:, None]
    full_means = uniform_scores.masked_fill(~full_mask, -torch.inf).softmax(-1) @ numbered_values
    window_means = uniform_scores.masked_fill(~visible, -torch.inf).softmax(-1) @ numbered_values
    torch.testing.assert_close(full_means[[5, 7]], torch.tensor([2.5, 3.5], dtype=torch.float64))
    torch.testing.assert_close(window_means[[5, 7]], torch.tensor([4.0, 6.0], dtype=torch.float64))

    old_key = rotate(k_raw[4], positions[4], math.pi / 4)
    new_key = rotate(k_raw[4], positions[4], frequency)
    assert not torch.allclose(old_key, new_key)
    print("Nine positions: ring = same-window reference; full and window means differ at 5 and 7")


if __name__ == "__main__":
    verify()
