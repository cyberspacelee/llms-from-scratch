"""Reproduce the quantization arithmetic and storage accounts in S7."""

import numpy as np


def symmetric(values, bits=3, axis=None):
    if not 2 <= bits <= 8 or not np.isfinite(values).all():
        raise ValueError("expected finite values and 2..8 bits")
    limit = 2 ** (bits - 1) - 1
    maximum = np.max(np.abs(values), axis=axis, keepdims=True)
    scale = np.where(maximum == 0, 1.0, maximum / limit)
    code = np.clip(np.rint(values / scale), -limit, limit).astype(np.int8)
    return code, scale


def affine(values, bits=3):
    if not 2 <= bits <= 8 or not np.isfinite(values).all():
        raise ValueError("expected finite values and 2..8 bits")
    limit = 2**bits - 1
    lower = min(float(np.min(values)), 0.0)
    upper = max(float(np.max(values)), 0.0)
    scale = (upper - lower) / limit if upper > lower else 1.0
    zero = int(np.clip(np.rint(-lower / scale), 0, limit))
    code = np.clip(np.rint(values / scale) + zero, 0, limit).astype(np.uint8)
    return code, scale, zero


def verify():
    weight = np.array([[-1.0, -0.5, 0.5, 1.0], [-20.0, 20.0, -1.0, 1.0]])
    activation = np.array([0.0, 1.0, 2.0, 3.0])
    exact = weight @ activation
    assert np.allclose(exact, [3.5, 21.0])

    whole, whole_scale = symmetric(weight)
    rows, row_scales = symmetric(weight, axis=1)
    # Shape is output row, contiguous group, elements within group.
    groups, group_scales = symmetric(weight.reshape(2, 2, 2), axis=2)
    restored_whole = whole * whole_scale
    restored_rows = rows * row_scales
    restored_groups = (groups * group_scales).reshape(2, 4)
    assert np.allclose(restored_whole @ activation, [0.0, 20.0])
    assert np.allclose(restored_rows @ activation, [11 / 3, 20.0])
    assert np.allclose(restored_groups @ activation, [11 / 3, 21.0])
    assert np.allclose(group_scales[1, :, 0], [20 / 3, 1 / 3])

    eight, eight_scales = symmetric(weight, bits=8, axis=1)
    assert eight[0, 2] == 64 and eight[1, 2] == -6
    assert np.isclose((eight * eight_scales)[0, 2], 64 / 127)
    assert np.isclose((eight * eight_scales)[1, 2], -120 / 127)

    code, scale, zero = affine(activation)
    assert code.tolist() == [0, 2, 5, 7] and scale == 3 / 7 and zero == 0
    restored_activation = scale * (code.astype(float) - zero)
    assert np.allclose(restored_activation, [0, 6 / 7, 15 / 7, 3])
    assert np.isclose(restored_groups[1] @ restored_activation, 18)
    _, shifted_scale, shifted_zero = affine(np.array([-1.0, 3.0]))
    assert shifted_scale == 4 / 7 and shifted_zero == 2

    zeros, zero_scales = symmetric(np.zeros((2, 4)), axis=1)
    assert np.all(zeros == 0) and np.all(zero_scales == 1)
    clipped = np.clip(np.rint(np.array([0.031, 3.0]) / 0.02), -127, 127) * 0.02
    assert np.allclose(clipped, [0.04, 2.54])
    assert np.rint([0.5, 1.5, 2.5]).tolist() == [0.0, 2.0, 2.0]

    packed_code_bytes = weight.size * 3 / 8
    assert [packed_code_bytes + 2 * scales for scales in (1, 2, 4)] == [5, 7, 11]
    assert whole.nbytes == 8  # NumPy stores each 3-bit teaching code in int8.
    assert 128 * 4 / 8 + 2 == 66
    assert 128 * 4 / 8 + 2 + 1 == 67
    print("exact:", exact, "whole:", restored_whole @ activation)
    print("per row:", restored_rows @ activation, "per group:", restored_groups @ activation)
    print(
        "activation codes:",
        code,
        "quantized second output:",
        restored_groups[1] @ restored_activation,
    )
    print("packed teaching bytes (whole, row, group): 5, 7, 11")


if __name__ == "__main__":
    verify()
