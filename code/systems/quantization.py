"""Symmetric integer representation and its storage accounting."""
import numpy as np


def quantize(x, bits=8, axis=None):
    if bits < 2 or bits > 8 or not np.isfinite(x).all():
        raise ValueError("finite values and 2..8 bits required")
    limit = 2 ** (bits - 1) - 1
    maximum = np.max(np.abs(x), axis=axis, keepdims=True)
    scale = np.where(maximum == 0, 1.0, maximum / limit)
    integers = np.clip(np.rint(x / scale), -limit, limit).astype(np.int8)
    return integers, scale


def affine_quantize(x, bits=8):
    if bits < 2 or bits > 8 or not np.isfinite(x).all():
        raise ValueError("finite values and 2..8 bits required")
    limit = 2 ** bits - 1
    lower, upper = min(float(x.min()), 0.), max(float(x.max()), 0.)
    scale = (upper - lower) / limit if upper > lower else 1.
    zero = int(np.clip(np.rint(-lower / scale), 0, limit))
    integers = np.clip(np.rint(x / scale) + zero, 0, limit).astype(np.uint8)
    return integers, scale, zero


def verify():
    x = np.array([[-1., -.5, 0., .5, 1.], [-20., -1., 0., 1., 20.]])
    q, scale = quantize(x, 3)
    assert scale.item() == 20 / 3
    assert q.min() >= -3 and q.max() <= 3
    grouped, scales = quantize(x, 3, axis=1)
    assert scales.shape == (2, 1)
    assert np.mean((grouped[0] * scales[0] - x[0]) ** 2) < np.mean((q[0] * scale - x[0]) ** 2)
    zero, zero_scale = quantize(np.zeros((2, 4)), axis=1)
    assert np.all(zero == 0) and np.all(zero_scale == 1)
    # Packed INT4, fp16 scale, group size 128.
    assert 128 * 4 / 8 + 2 == 66
    assert np.rint(np.array([.5, 1.5, 2.5])).tolist() == [0., 2., 2.]
    affine, affine_scale, zero = affine_quantize(np.array([-2., 0., 6.]))
    assert affine.tolist() == [0, 64, 255] and zero == 64
    restored = affine_scale * (affine.astype(np.float64) - zero)
    assert restored[1] == 0 and np.max(np.abs(restored - [-2., 0., 6.])) <= affine_scale / 2
    rng = np.random.default_rng(2)
    activation, weight = rng.normal(size=(3, 4)), rng.normal(size=(4, 5))
    scaling = np.array([.5, 1., 2., 3.])
    assert np.allclose((activation / scaling) @ (weight * scaling[:, None]), activation @ weight)
    print("quantization: affine zero point, ranges, outliers, scaling and metadata verified")


if __name__ == "__main__":
    verify()
