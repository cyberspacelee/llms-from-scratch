"""Independent CPU checks of NumPy array shapes, storage and indexing."""
import numpy as np


def main():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert a.shape == (2, 3) and a.ndim == 2 and a.size == 6
    assert a.itemsize == 8 and a.nbytes == 48
    assert np.asarray(a) is a
    assert not np.shares_memory(a, np.array(a))
    np.testing.assert_array_equal(np.linspace(0, 1, 5), [0, .25, .5, .75, 1])
    assert np.zeros((2, 3)).sum() == 0
    assert np.ones((2, 3)).sum() == 6
    scratch = np.empty((2, 3), dtype=np.float64)
    scratch.fill(7)
    assert scratch.sum() == 42
    assert a[1, -1] == 5
    assert a[1].shape == (3,) and a[1:2].shape == (1, 3)
    assert a[:, 1].shape == (2,) and a[:, 1:2].shape == (2, 1)
    np.testing.assert_array_equal(a[:, ::-1], [[2, 1, 0], [5, 4, 3]])
    view = a[:, 1:]
    copied = a[:, 1:].copy()
    advanced = a[:, [1, 2]]
    assert np.shares_memory(a, view)
    assert not np.shares_memory(a, copied) and not np.shares_memory(a, advanced)
    view[0, 0] = 9
    assert a[0, 1] == 9
    copied[0, 0] = 8
    advanced[0, 0] = 7
    assert a[0, 1] == 9
    a[:, [1, 2]] = [[10, 20], [40, 50]]
    np.testing.assert_array_equal(a, [[0, 10, 20], [3, 40, 50]])
    a[:, [1, 2]][0, 0] = -1
    assert a[0, 1] == 10
    np.testing.assert_array_equal(a[[0, 1], [1, 2]], [10, 50])
    np.testing.assert_array_equal(a[np.ix_([0, 1], [1, 2])], [[10, 20], [40, 50]])
    mask = a > 15
    np.testing.assert_array_equal(a[mask], [20, 40, 50])
    assert not np.shares_memory(a, a[mask])
    a[mask] = -2
    np.testing.assert_array_equal(a, [[0, 10, -2], [3, -2, -2]])
    repeated = np.zeros(3, dtype=np.int64)
    repeated[[0, 0]] += 1
    assert repeated[0] == 1
    np.add.at(repeated, [0, 0], 1)
    assert repeated[0] == 3
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    assert a.strides == (24, 8) and a.T.strides == (8, 24)
    assert np.shares_memory(a, a.T)
    np.testing.assert_array_equal(a.reshape(3, 2), [[0, 1], [2, 3], [4, 5]])
    np.testing.assert_array_equal(a.T, [[0, 3], [1, 4], [2, 5]])
    flattened = a.T.reshape(6)
    assert not np.shares_memory(a, flattened)
    np.testing.assert_array_equal(flattened, [0, 3, 1, 4, 2, 5])
    assert a[None].shape == (1, 2, 3)
    empty = np.zeros((0, 3))
    assert empty.size == 0 and empty.ndim == 2
    print(f"NumPy {np.__version__}: array shape, dtype, views/copies, indexing and layout passed")


if __name__ == '__main__':
    main()
