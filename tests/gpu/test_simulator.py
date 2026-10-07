import numpy as np

from llms_from_scratch.gpu.simulator import (
    MemStats,
    bank_conflict_degree,
    sectors_per_request,
    thread_map_1d,
    transpose_naive_sim,
    transpose_shared_sim,
    vector_add_sim,
)


def test_thread_map_covers_each_element_once():
    table = thread_map_1d(1000, 256)
    assert table.shape[0] == 4 * 256  # 向上取整到 4 个 block
    active = table[table[:, 3] == 1]
    assert np.array_equal(np.sort(active[:, 2]), np.arange(1000))
    assert (table[table[:, 3] == 0, 2] >= 1000).all()


def test_vector_add_sim():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(300).astype(np.float32)
    b = rng.standard_normal(300).astype(np.float32)
    assert np.array_equal(vector_add_sim(a, b, block_dim=64), a + b)


def test_coalescing_patterns():
    lanes = np.arange(32)
    assert sectors_per_request(lanes * 4) == 4  # 连续 float：128 B = 4 个扇区
    assert sectors_per_request((lanes + 1) * 4) == 5  # 错位一个元素：多跨一个扇区
    assert sectors_per_request(lanes * 2 * 4) == 8  # 步长 2：一半字节浪费
    assert sectors_per_request(lanes * 32 * 4) == 32  # 步长 32：每个线程一个扇区
    assert sectors_per_request(np.zeros(32, dtype=int)) == 1  # 全部读同一地址


def test_bank_conflicts():
    lanes = np.arange(32)
    assert bank_conflict_degree(lanes * 4) == 1
    assert bank_conflict_degree(lanes * 2 * 4) == 2  # 步长 2：两路冲突
    assert bank_conflict_degree(lanes * 32 * 4) == 32  # 同一列：32 路冲突
    assert bank_conflict_degree(lanes * 33 * 4) == 1  # padding 之后无冲突
    assert bank_conflict_degree(np.zeros(32, dtype=int)) == 1  # 广播


def test_transpose_variants_and_traffic():
    x = np.arange(64 * 96, dtype=np.float32).reshape(64, 96)
    naive, tiled, padded = MemStats(), MemStats(), MemStats()
    for out in (
        transpose_naive_sim(x, naive),
        transpose_shared_sim(x, tiled),
        transpose_shared_sim(x, padded, pad=1),
    ):
        assert np.array_equal(out, x.T)
    assert naive.sectors_per_load == 4 and naive.sectors_per_store == 32
    assert tiled.sectors_per_load == 4 and tiled.sectors_per_store == 4
    assert naive.store_sectors == 8 * tiled.store_sectors
    # 共享内存：写入无冲突、按列读出 32 路冲突；padding 后都无冲突
    assert tiled.smem_wavefronts / tiled.smem_requests == (1 + 32) / 2
    assert padded.smem_wavefronts == padded.smem_requests


def test_transpose_ragged_shape():
    x = np.random.default_rng(1).standard_normal((35, 67)).astype(np.float32)
    assert np.array_equal(transpose_shared_sim(x, MemStats(), pad=1), x.T)
    assert np.array_equal(transpose_naive_sim(x, MemStats()), x.T)
