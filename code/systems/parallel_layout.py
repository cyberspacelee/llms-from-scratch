"""Check the small two-rank inference layout used in S9, without a GPU."""

import numpy as np


def verify() -> None:
    x = np.array([[1, 2, 3, 4]], dtype=np.float64)
    w = np.arange(16, dtype=np.float64).reshape(4, 4)
    expected = x @ w

    columns = [x @ shard for shard in np.split(w, 2, axis=1)]
    rows = [a @ b for a, b in zip(np.split(x, 2, axis=1), np.split(w, 2, axis=0))]
    np.testing.assert_array_equal(expected, [[80, 90, 100, 110]])
    np.testing.assert_array_equal(np.concatenate(columns, axis=1), expected)
    np.testing.assert_array_equal(sum(rows), expected)
    np.testing.assert_array_equal(columns[0], [[80, 90]])
    np.testing.assert_array_equal(columns[1], [[100, 110]])
    np.testing.assert_array_equal(rows[0], [[8, 11, 14, 17]])
    np.testing.assert_array_equal(rows[1], [[72, 79, 86, 93]])

    element_bytes = 2
    column_send_per_rank = columns[0].size * element_bytes
    row_send_per_rank = rows[0].size * element_bytes
    assert column_send_per_rank == 4 and row_send_per_rank == 8

    layers, hidden, kv_heads, head_dim = 2, 4, 1, 2
    kv_per_token_per_layer = 2 * kv_heads * head_dim * element_bytes
    assert kv_per_token_per_layer == 8
    assert layers * kv_per_token_per_layer == 16
    assert 5 * hidden * element_bytes == 40  # Five prompt tokens cross one PP boundary.

    stages, microbatches = 4, 12
    elapsed = microbatches + stages - 1
    assert elapsed == 15
    assert (stages - 1) / microbatches == .25
    assert (stages - 1) / elapsed == .20
    assert (stages - 1) / (57 + stages - 1) == .05
    assert (stages - 1) / (58 + stages - 1) < .05

    gib_bytes = 2**30
    seconds = gib_bytes / 50e9
    assert .0214 < seconds < .0215
    print("S9: TP results, per-rank bytes, PP state and schedule verified")


if __name__ == "__main__":
    verify()
