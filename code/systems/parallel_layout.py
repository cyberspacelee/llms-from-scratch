"""Tensor-parallel arithmetic and packed metadata, without collectives."""
import numpy as np


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
