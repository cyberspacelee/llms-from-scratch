"""Window attention and a bounded ring cache with absolute position tags."""
import math
import torch


def verify():
    torch.manual_seed(3)
    length, window, width = 8, 3, 2
    q, k, v = [torch.randn(length, width, dtype=torch.float64) for _ in range(3)]
    positions = torch.arange(length)
    visible = (positions[None, :] <= positions[:, None]) & (
        positions[None, :] > positions[:, None] - window)
    dense = (q @ k.T / math.sqrt(width)).masked_fill(~visible, -torch.inf).softmax(-1) @ v
    ring_k, ring_v = torch.zeros(window, width, dtype=q.dtype), torch.zeros(window, width, dtype=q.dtype)
    tags = torch.full((window,), -1)
    outputs = []
    for position in range(length):
        slot = position % window
        ring_k[slot], ring_v[slot], tags[slot] = k[position], v[position], position
        expected_positions = torch.arange(max(0, position - window + 1), position + 1)
        slots = expected_positions % window
        assert torch.equal(tags[slots], expected_positions)
        weights = (ring_k[slots] @ q[position] / math.sqrt(width)).softmax(0)
        outputs.append(weights @ ring_v[slots])
    torch.testing.assert_close(torch.stack(outputs), dense)
    full = (q @ k.T / math.sqrt(width)).masked_fill(positions[None, :] > positions[:, None],
                                                   -torch.inf).softmax(-1) @ v
    assert not torch.allclose(full[-1], dense[-1])
    old_frequency, new_frequency = .4, .2
    old_key = torch.tensor([math.cos(5 * old_frequency), math.sin(5 * old_frequency)], dtype=q.dtype)
    rebuilt_key = torch.tensor([math.cos(5 * new_frequency), math.sin(5 * new_frequency)], dtype=q.dtype)
    assert not torch.allclose(old_key, rebuilt_key)
    print("Window: ring cache = same-window dense reference; differs from full attention")


if __name__ == "__main__":
    verify()
