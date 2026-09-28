"""Online attention reference for the six-key example; CPU correctness only."""

import numpy as np


def online_attention(q, k, v, mask, tile=3):
    if tile <= 0 or mask.shape != (len(q), len(k)):
        raise ValueError("invalid tile or mask")
    if not np.all(mask.any(axis=1)):
        raise ValueError("each query needs a visible key")

    out = np.zeros((len(q), v.shape[1]), dtype=np.float64)
    for i, query in enumerate(q):
        maximum, denominator = -np.inf, 0.0
        numerator = np.zeros(v.shape[1], dtype=np.float64)
        for start in range(0, len(k), tile):
            visible = mask[i, start:start + tile]
            if not visible.any():
                continue
            scores = k[start:start + tile] @ query / np.sqrt(q.shape[1])
            scores = np.where(visible, scores, -np.inf)
            new_maximum = max(maximum, scores.max())
            old_scale = np.exp(maximum - new_maximum)
            weights = np.exp(scores - new_maximum)
            denominator = denominator * old_scale + weights.sum()
            numerator = numerator * old_scale + weights @ v[start:start + tile]
            maximum = new_maximum
        out[i] = numerator / denominator
    return out


def dense_attention(q, k, v, mask):
    scores = q @ k.T / np.sqrt(q.shape[1])
    scores = np.where(mask, scores, -np.inf)
    maximum = scores.max(axis=1, keepdims=True)
    weights = np.exp(scores - maximum)
    return (weights @ v) / weights.sum(axis=1, keepdims=True)


def verify():
    query_positions = np.arange(3, 6)
    key_positions = np.arange(6)
    q = np.tile(np.array([[1.0, 0.0]]), (3, 1))
    k = np.column_stack((np.sqrt(2.0) * key_positions, np.zeros(6)))
    v = (key_positions + 1).astype(np.float64)[:, None]
    causal = key_positions[None, :] <= query_positions[:, None]

    expected = dense_attention(q, k, v, causal)
    assert np.isclose(expected[-1, 0], 5.43293276, atol=1e-7)
    for tile in (1, 2, 3, 4, 8):
        assert np.allclose(online_attention(q, k, v, causal, tile), expected)

    # Shift every allowed score by 1000 without changing its softmax.
    shifted_k = k + np.array([1000 * np.sqrt(2.0), 0.0])
    assert np.allclose(dense_attention(q, shifted_k, v, causal), expected)
    for tile in (1, 3, 8):
        assert np.allclose(online_attention(q, shifted_k, v, causal, tile), expected)

    padded = causal & (key_positions[None, :] != 0)
    assert np.allclose(online_attention(q, k, v, padded), dense_attention(q, k, v, padded))
    try:
        online_attention(q, k, v, np.zeros_like(causal))
    except ValueError:
        pass
    else:
        raise AssertionError("all-masked row must be rejected")

    print(f"position 5 output: {expected[-1, 0]:.6f}")
    print("dense/online, tile edges, large scores and masks verified")


if __name__ == "__main__":
    verify()
