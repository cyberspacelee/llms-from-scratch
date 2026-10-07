"""在 CPU 上按 grid / block / warp / thread 的顺序显式模拟 CUDA kernel，并统计访存。

这里的“硬件模型”刻意简化，只保留决定性能的两条规则：

* 全局内存：一个 warp 的一次访存请求被拆成若干 32 字节扇区（sector）；
  扇区数越少，合并（coalescing）得越好。
* 共享内存：32 个 bank、每个 bank 宽 4 字节；一个 warp 内落在同一 bank
  的不同地址必须串行，串行次数就是冲突度（n-way bank conflict）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

WARP_SIZE = 32
SECTOR_BYTES = 32
NUM_BANKS = 32
BANK_BYTES = 4


# region thread_map
def thread_map_1d(n: int, block_dim: int) -> np.ndarray:
    """返回每个线程的 (blockIdx.x, threadIdx.x, 全局下标 i, 是否越界被 if 挡掉)。

    与 kernel 中的 ``i = blockIdx.x * blockDim.x + threadIdx.x`` 一一对应。
    """
    grid_dim = (n + block_dim - 1) // block_dim  # 向上取整
    rows = []
    for block in range(grid_dim):
        for thread in range(block_dim):
            i = block * block_dim + thread
            rows.append((block, thread, i, int(i < n)))
    return np.array(rows, dtype=np.int64).reshape(-1, 4)


def vector_add_sim(a: np.ndarray, b: np.ndarray, block_dim: int = 256) -> np.ndarray:
    """逐线程执行 vector add：每个线程负责一个元素。"""
    n = a.size
    c = np.zeros_like(a)
    for _block, _thread, i, active in thread_map_1d(n, block_dim):
        if active:  # 对应 kernel 中的 if (i < n)
            c[i] = a[i] + b[i]
    return c


# endregion thread_map


# region coalescing
def sectors_per_request(byte_addresses: np.ndarray) -> int:
    """一个 warp 的一次访存涉及多少个不同的 32 字节扇区。"""
    return int(np.unique(np.asarray(byte_addresses) // SECTOR_BYTES).size)


def bank_conflict_degree(byte_addresses: np.ndarray) -> int:
    """一个 warp 访问共享内存时的串行次数（1 表示无冲突）。

    同一 bank 上的 *不同* 4 字节字才冲突；同一个字被多个线程读是广播，不冲突。
    """
    words = np.unique(np.asarray(byte_addresses) // BANK_BYTES)
    banks = words % NUM_BANKS
    return int(np.bincount(banks, minlength=NUM_BANKS).max())


# endregion coalescing


@dataclass
class MemStats:
    """累计统计：全局内存请求/扇区数与共享内存的串行周期数。"""

    load_requests: int = 0
    load_sectors: int = 0
    store_requests: int = 0
    store_sectors: int = 0
    smem_requests: int = 0
    smem_wavefronts: int = 0  # 共享内存访问的串行次数之和
    notes: list[str] = field(default_factory=list)

    def global_load(self, byte_addresses: np.ndarray) -> None:
        self.load_requests += 1
        self.load_sectors += sectors_per_request(byte_addresses)

    def global_store(self, byte_addresses: np.ndarray) -> None:
        self.store_requests += 1
        self.store_sectors += sectors_per_request(byte_addresses)

    def shared(self, byte_addresses: np.ndarray) -> None:
        self.smem_requests += 1
        self.smem_wavefronts += bank_conflict_degree(byte_addresses)

    @property
    def sectors_per_load(self) -> float:
        return self.load_sectors / max(1, self.load_requests)

    @property
    def sectors_per_store(self) -> float:
        return self.store_sectors / max(1, self.store_requests)


# region transpose_sim
def transpose_naive_sim(x: np.ndarray, stats: MemStats) -> np.ndarray:
    """朴素转置：线程 (tx, ty) 读 in[y, x]，写 out[x, y]。

    一个 warp = 同一 ty 上连续 32 个 tx。读按行连续（合并），写跨行（每个线程一个扇区）。
    """
    rows, cols = x.shape
    out = np.zeros((cols, rows), dtype=x.dtype)
    elem = x.itemsize
    for y in range(rows):
        for x0 in range(0, cols, WARP_SIZE):
            xs = np.arange(x0, min(x0 + WARP_SIZE, cols))
            stats.global_load((y * cols + xs) * elem)
            stats.global_store((xs * rows + y) * elem)
            out[xs, y] = x[y, xs]
    return out


def transpose_shared_sim(
    x: np.ndarray, stats: MemStats, tile: int = 32, pad: int = 0
) -> np.ndarray:
    """共享内存转置：先合并地读入 tile，再从 tile 的“列”读出、合并地写回。

    ``pad=1`` 时共享内存数组是 ``tile[32][33]``，列访问落在 32 个不同 bank。
    """
    rows, cols = x.shape
    out = np.zeros((cols, rows), dtype=x.dtype)
    elem = x.itemsize
    stride = tile + pad  # 共享内存一行的元素数
    for by in range(0, rows, tile):
        for bx in range(0, cols, tile):
            smem = np.zeros((tile, stride), dtype=x.dtype)
            # 阶段 1：warp ty 读入第 by+ty 行的一段，写入 smem[ty][tx]
            for ty in range(min(tile, rows - by)):
                tx = np.arange(min(tile, cols - bx))
                stats.global_load(((by + ty) * cols + bx + tx) * elem)
                stats.shared((ty * stride + tx) * elem)
                smem[ty, tx] = x[by + ty, bx + tx]
            # __syncthreads()
            # 阶段 2：warp ty 写 out 的第 bx+ty 行，数据来自 smem[tx][ty]（一列）
            for ty in range(min(tile, cols - bx)):
                tx = np.arange(min(tile, rows - by))
                stats.shared((tx * stride + ty) * elem)
                stats.global_store(((bx + ty) * rows + by + tx) * elem)
                out[bx + ty, by + tx] = smem[tx, ty]
    return out


# endregion transpose_sim
