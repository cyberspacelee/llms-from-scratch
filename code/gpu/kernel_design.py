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
                aa = a[row:row + tile, start:start + tile]
                bb = b[start:start + tile, col:col + tile]
                at[:aa.shape[0], :aa.shape[1]] = aa
                bt[:bb.shape[0], :bb.shape[1]] = bb
                accumulator += at @ bt
            height, width = min(tile, m - row), min(tile, n - col)
            out[row:row + height, col:col + width] = accumulator[:height, :width]
    return out


def demo():
    rng = np.random.default_rng(7)
    x = np.array([[1000., 1001., 1002.], [-2., 0., 2.], [1., 2., 3.]])
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
    print("G3 CPU checks passed: stable masked softmax, zero empty rows, tiled GEMM edges")


if __name__ == "__main__":
    demo()
