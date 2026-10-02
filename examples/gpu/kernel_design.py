"""CPU float64 references for G3; no GPU or optional compiler required."""

import numpy as np


def stable_softmax(x, valid=None):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty-width finite 2D matrix")
    valid = np.ones_like(x, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != x.shape:
        raise ValueError("Mask shape must match logits")
    active = valid.any(axis=1, keepdims=True)
    maximum = np.max(np.where(valid, x, -np.inf), axis=1, keepdims=True)
    maximum = np.where(active, maximum, 0.0)
    numerator = np.exp(np.where(valid, x - maximum, -np.inf))
    denominator = numerator.sum(axis=1, keepdims=True)
    return numerator / np.where(active, denominator, 1.0)


def tiled_gemm(a, b, tile=4):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or tile < 1:
        raise ValueError("Incompatible matrices or tile")
    m, k = a.shape
    n = b.shape[1]
    out = np.zeros((m, n), dtype=np.float64)
    for row in range(0, m, tile):
        for col in range(0, n, tile):
            accumulator = np.zeros((tile, tile), dtype=np.float64)
            for start in range(0, k, tile):
                at = np.zeros((tile, tile), dtype=np.float64)
                bt = np.zeros((tile, tile), dtype=np.float64)
                aa = a[row : row + tile, start : start + tile]
                bb = b[start : start + tile, col : col + tile]
                at[: aa.shape[0], : aa.shape[1]] = aa
                bt[: bb.shape[0], : bb.shape[1]] = bb
                accumulator += at @ bt
            height, width = min(tile, m - row), min(tile, n - col)
            out[row : row + height, col : col + width] = accumulator[:height, :width]
    return out


def tile_intensity(tile, element_bytes):
    """Input-only FLOP/byte of one square output tile per K step: 2t^3 FLOP over 2t^2 loads."""
    if tile < 1 or element_bytes <= 0:
        raise ValueError("Positive tile and element size required")
    return 2 * tile**3 / (element_bytes * 2 * tile**2)


def roofline(intensity, peak_flops, bandwidth):
    """Attainable FLOP/s bound and ridge point; a model, not a measurement."""
    if min(intensity, peak_flops, bandwidth) <= 0:
        raise ValueError("Positive intensity, peak and bandwidth required")
    return min(peak_flops, bandwidth * intensity), peak_flops / bandwidth


def demo():
    rng = np.random.default_rng(7)
    rows = np.arange(5)[:, None]
    inner = np.arange(7)
    cols = np.arange(6)[None, :]
    a = rows + inner + 1
    b = inner[:, None] - cols
    assert sum(a[0, q] * b[q, 0] for q in range(4)) == 20
    assert sum(a[0, q] * b[q, 0] for q in range(4, 7)) == 92
    assert tiled_gemm(a, b, 4)[0, 0] == 112
    np.testing.assert_allclose(tiled_gemm(a, b, 4), a @ b)
    assert 5 * 7 * ((6 + 3) // 4) + 7 * 6 * ((5 + 3) // 4) == 154
    x = np.array([[1000.0, 1001.0, 1002.0], [-2.0, 0.0, 2.0], [1.0, 2.0, 3.0]])
    mask = np.array([[True, True, True], [True, False, True], [False, False, False]])
    y = stable_softmax(x, mask)
    np.testing.assert_allclose(y[0], [0.0900305732, 0.2447284711, 0.6652409558], atol=1e-10)
    np.testing.assert_allclose(y[:2].sum(axis=1), 1.0)
    assert (y[~mask] == 0).all()
    np.testing.assert_allclose(y, stable_softmax(x + 10000, mask), atol=1e-12)
    for m, k, n in [(5, 7, 6), (1, 1, 1), (17, 19, 13)]:
        a, b = rng.normal(size=(m, k)), rng.normal(size=(k, n))
        for tile in (1, 2, 4, 8):
            np.testing.assert_allclose(tiled_gemm(a, b, tile), a @ b, rtol=1e-12, atol=1e-12)
    # Teaching device from G3: 100 TFLOP/s, 2 TB/s, so the ridge is 50 FLOP/byte.
    assert tile_intensity(16, 4) == 4 and tile_intensity(16, 2) == 8
    bound, ridge = roofline(tile_intensity(16, 4), 100e12, 2e12)
    assert ridge == 50 and bound == 8e12
    assert roofline(tile_intensity(256, 2), 100e12, 2e12)[0] == 100e12
    print(
        "G3 CPU checks passed: stable masked softmax, zero empty rows, tiled GEMM edges, tile roofline"
    )


if __name__ == "__main__":
    demo()
