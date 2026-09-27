"""Online attention reference; CPU correctness, not a GPU benchmark."""
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


def verify():
    rng = np.random.default_rng(17)
    q, k, v = rng.normal(size=(3, 5)), rng.normal(size=(9, 5)), rng.normal(size=(9, 4))
    mask = np.arange(9)[None, :] <= np.arange(6, 9)[:, None]
    for scale in (1.0, 1000.0):
        scores = q @ (k * scale).T / np.sqrt(5)
        scores = np.where(mask, scores, -np.inf)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        expected = weights @ v / weights.sum(axis=1, keepdims=True)
        for tile in (1, 2, 4, 20):
            assert np.allclose(online_attention(q, k * scale, v, mask, tile), expected)
    try:
        online_attention(q, k, v, np.zeros_like(mask))
    except ValueError:
        pass
    else:
        raise AssertionError("all-masked row must be rejected")
    print("online attention: dense, tiles, extreme scores and masks verified")


if __name__ == "__main__":
    verify()
