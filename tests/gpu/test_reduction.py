import math

import numpy as np

from llms_from_scratch.gpu.reduction import (
    block_reduce_sum,
    grid_reduce_sum,
    shfl_down,
    tree_reduce_shared,
    warp_reduce_sum,
)


def test_tree_reduce_steps():
    v = np.arange(256, dtype=np.float32)
    total, active = tree_reduce_shared(v)
    assert total == v.sum()
    assert active == [128, 64, 32, 16, 8, 4, 2, 1]  # log2(256) = 8 步


def test_shfl_down_out_of_range_keeps_own_value():
    lanes = np.arange(32, dtype=np.float32)
    shifted = shfl_down(lanes, 4)
    assert np.array_equal(shifted[:28], lanes[4:])
    assert np.array_equal(shifted[28:], lanes[28:])


def test_warp_reduce():
    lanes = np.random.default_rng(0).standard_normal(32).astype(np.float32)
    assert math.isclose(warp_reduce_sum(lanes)[0], math.fsum(lanes), rel_tol=1e-5)


def test_block_and_grid_reduce():
    rng = np.random.default_rng(1)
    v = rng.standard_normal(256).astype(np.float32)
    assert math.isclose(block_reduce_sum(v), math.fsum(v), rel_tol=1e-4, abs_tol=1e-4)
    x = rng.standard_normal(5000).astype(np.float32)
    assert math.isclose(grid_reduce_sum(x, block_dim=64, grid_dim=8), math.fsum(x), abs_tol=1e-3)


def test_tree_order_is_more_accurate_than_sequential():
    # 大量相同的小数：顺序累加的误差随 n 增长，树形归约只随 log n 增长
    x = np.full(1 << 16, 0.1, dtype=np.float32)
    seq = np.float32(0)
    for value in x:
        seq += value
    tree, _ = tree_reduce_shared(x)
    exact = math.fsum(x.astype(np.float64))
    assert abs(tree - exact) < abs(seq - exact) / 10
