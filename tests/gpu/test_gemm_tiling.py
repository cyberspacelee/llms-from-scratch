import numpy as np
import pytest

from llms_from_scratch.gpu.gemm_tiling import (
    gemm_global_loads,
    gemm_mma_sim,
    gemm_naive_sim,
    gemm_register_tiled_sim,
    gemm_smem_tiled_sim,
    tile_intensity,
)

rng = np.random.default_rng(0)


def rand(*shape):
    return rng.standard_normal(shape).astype(np.float32)


def test_naive():
    a, b = rand(5, 7), rand(7, 3)
    c, stats = gemm_naive_sim(a, b)
    assert np.allclose(c, a @ b, atol=1e-5)
    assert stats.global_loads == 2 * 5 * 3 * 7 and stats.fmas == 5 * 3 * 7


def test_smem_tiled_traffic():
    m = n = k = 32
    a, b = rand(m, k), rand(k, n)
    c, stats = gemm_smem_tiled_sim(a, b, tile=8)
    assert np.allclose(c, a @ b, atol=1e-4)
    assert stats.global_loads == gemm_global_loads(m, n, k, 8, 8)
    assert stats.global_loads * 8 == 2 * m * n * k  # 比朴素少 tile 倍
    assert stats.smem_loads == 2 * stats.fmas  # 每次 FMA 读两次共享内存


def test_register_tiled_traffic():
    m, n, k = 64, 64, 32
    a, b = rand(m, k), rand(k, n)
    c, stats = gemm_register_tiled_sim(a, b, bm=32, bn=32, bk=8, tm=4, tn=4)
    assert np.allclose(c, a @ b, atol=1e-4)
    assert stats.global_loads == gemm_global_loads(m, n, k, 32, 32)
    assert stats.fmas == m * n * k
    assert stats.smem_loads * 2 == stats.fmas  # (4+4)/(4·4) = 0.5 次/FMA


@pytest.mark.parametrize("shape", [(17, 19, 13), (33, 5, 40)])
def test_ragged_shapes(shape):
    m, k, n = shape
    a, b = rand(m, k), rand(k, n)
    assert np.allclose(gemm_smem_tiled_sim(a, b, tile=8)[0], a @ b, atol=1e-4)
    assert np.allclose(gemm_register_tiled_sim(a, b, 16, 16, 4, 4, 4)[0], a @ b, atol=1e-4)


def test_mma_sim():
    a, b = rand(32, 48), rand(48, 16)
    c, count = gemm_mma_sim(a, b)
    assert count == 2 * 1 * 3
    ref = a.astype(np.float16).astype(np.float64) @ b.astype(np.float16).astype(np.float64)
    assert np.allclose(c, ref, atol=1e-4)  # 误差只来自输入舍入到 FP16


def test_tile_intensity():
    assert tile_intensity(1, 1) == 0.25
    assert tile_intensity(128, 128) == 32.0
