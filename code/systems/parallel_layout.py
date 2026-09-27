"""Tensor-parallel arithmetic and packed metadata, without collectives."""
import numpy as np


def zero_bytes(stage, params, ranks):
    """Per-rank bytes of mixed-precision Adam state (2 param + 2 grad + 12 optimizer) under ZeRO."""
    if stage not in (0, 1, 2, 3) or params <= 0 or ranks <= 0:
        raise ValueError("stage 0-3 and positive sizes required")
    param, grad, optimizer = 2 * params, 2 * params, 12 * params
    return (param / ranks if stage >= 3 else param) + (grad / ranks if stage >= 2 else grad) \
        + (optimizer / ranks if stage >= 1 else optimizer)


def verify_zero():
    n = 8029995008
    assert zero_bytes(0, n, 8) == 16 * n
    assert zero_bytes(1, n, 8) == 4 * n + 12 * n / 8
    assert zero_bytes(2, n, 8) == 2 * n + 14 * n / 8
    assert zero_bytes(3, n, 8) == 2 * n
    assert 44.1e9 < zero_bytes(1, n, 8) < 44.2e9 and 30.1e9 < zero_bytes(2, n, 8) < 30.2e9
    # Reduce-scatter then all-gather equals all-reduce: each rank ends with the summed shard it owns.
    rng = np.random.default_rng(9)
    grads = rng.normal(size=(4, 12))
    shards = [part.sum(axis=0) for part in np.split(grads, 4, axis=1)]
    np.testing.assert_allclose(np.concatenate(shards), grads.sum(axis=0))
    # A rank updates only its shard; gathering the shards rebuilds the full parameter vector.
    weights = rng.normal(size=12)
    updated = [w - 0.1 * g for w, g in zip(np.split(weights, 4), shards)]
    np.testing.assert_allclose(np.concatenate(updated), weights - 0.1 * grads.sum(axis=0))
    print("parallel: ZeRO per-rank bytes for stages 0-3, reduce-scatter/all-gather sharded update verified")


def verify():
    x = np.array([[1., 2., 3., 4.]])
    w = np.arange(16, dtype=np.float64).reshape(4, 4)
    column = np.concatenate([x @ shard for shard in np.split(w, 2, axis=1)], axis=1)
    row = sum(a @ b for a, b in zip(np.split(x, 2, axis=1), np.split(w, 2, axis=0)))
    assert np.array_equal(column, x @ w) and np.array_equal(row, x @ w)
    assert np.array_equal(x @ w, [[80., 90., 100., 110.]])
    lengths, cached = [3, 2, 4], [0, 4, 0]
    cu_q = np.concatenate([[0], np.cumsum(lengths)])
    cu_k = np.concatenate([[0], np.cumsum(np.array(lengths) + cached)])
    positions = np.concatenate([np.arange(past, past + new) for past, new in zip(cached, lengths)])
    assert cu_q.tolist() == [0, 3, 5, 9] and cu_k.tolist() == [0, 3, 9, 13]
    assert positions.tolist() == [0, 1, 2, 4, 5, 0, 1, 2, 3]
    assert (4 - 1) / (12 + 4 - 1) == .2
    assert 3 / (57 + 3) == .05 and 3 / (58 + 3) < .05
    print("parallel: column concatenate, row sum, packed positions and bubble verified")


if __name__ == "__main__":
    verify()
    verify_zero()
