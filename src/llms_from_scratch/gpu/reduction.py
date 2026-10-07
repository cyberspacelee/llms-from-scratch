"""并行归约的 CPU 模拟：共享内存树形归约、warp shuffle、block 归约与两阶段网格归约。

每个函数都按 GPU 上的执行顺序做加法（而不是调用 ``sum``），
因此结果的舍入方式与 kernel 一致，步数也可以直接数出来。
"""

from __future__ import annotations

import numpy as np

WARP_SIZE = 32


# region tree_reduce
def tree_reduce_shared(values: np.ndarray) -> tuple[np.float32, list[int]]:
    """一个 block 在共享内存上做“顺序寻址”的树形归约。

    第 s 步只有前 ``stride`` 个线程工作：``smem[t] += smem[t + stride]``，
    stride 从 blockDim/2 每步减半到 1。返回结果与每一步的活跃线程数。
    """
    smem = np.array(values, dtype=np.float32)
    n = smem.size
    assert n & (n - 1) == 0, "blockDim 取 2 的幂"
    active = []
    stride = n // 2
    while stride > 0:
        t = np.arange(stride)  # if (threadIdx.x < stride)
        smem[t] += smem[t + stride]
        active.append(stride)
        stride //= 2  # __syncthreads() 之后进入下一步
    return smem[0], active


# endregion tree_reduce


# region warp_shuffle
def shfl_down(lanes: np.ndarray, offset: int) -> np.ndarray:
    """``__shfl_down_sync(0xffffffff, v, offset)``：lane i 读到 lane i+offset 的值；

    源 lane 越界（>= 32）时返回自己的值。
    """
    src = np.arange(WARP_SIZE) + offset
    src = np.where(src < WARP_SIZE, src, np.arange(WARP_SIZE))
    return lanes[src]


def warp_reduce_sum(lanes: np.ndarray) -> np.ndarray:
    """五轮 shuffle 后 lane 0 持有整个 warp 的和（其余 lane 是部分和）。"""
    v = np.array(lanes, dtype=np.float32)
    assert v.size == WARP_SIZE
    for offset in (16, 8, 4, 2, 1):
        v = v + shfl_down(v, offset)
    return v


# endregion warp_shuffle


# region block_reduce
def block_reduce_sum(values: np.ndarray) -> np.float32:
    """block 级归约：warp 内 shuffle → 每个 warp 的 lane 0 写共享内存 → 第 0 个 warp 再 shuffle。"""
    v = np.array(values, dtype=np.float32)
    n_warps = v.size // WARP_SIZE
    assert v.size == n_warps * WARP_SIZE and n_warps <= WARP_SIZE
    warp_sums = np.zeros(WARP_SIZE, dtype=np.float32)  # __shared__ float s[32]
    for w in range(n_warps):
        warp_sums[w] = warp_reduce_sum(v[w * WARP_SIZE : (w + 1) * WARP_SIZE])[0]
    # __syncthreads(); 第 0 个 warp 读取 s[lane]（lane >= n_warps 的补 0）
    return warp_reduce_sum(warp_sums)[0]


def grid_reduce_sum(x: np.ndarray, block_dim: int = 256, grid_dim: int = 4) -> np.float32:
    """两阶段归约：第一次启动 grid_dim 个 block，每个线程用网格跨步循环累加，

    block 归约得到 grid_dim 个部分和；第二次启动 1 个 block 归约这些部分和。
    """
    x = np.asarray(x, dtype=np.float32)
    partials = np.zeros(grid_dim, dtype=np.float32)
    total_threads = block_dim * grid_dim
    for block in range(grid_dim):
        per_thread = np.zeros(block_dim, dtype=np.float32)
        for t in range(block_dim):
            # for (i = blockIdx.x*blockDim.x + t; i < n; i += blockDim.x*gridDim.x)
            for i in range(block * block_dim + t, x.size, total_threads):
                per_thread[t] += x[i]
        partials[block] = block_reduce_sum(per_thread)
    padded = np.zeros(block_dim, dtype=np.float32)
    padded[:grid_dim] = partials
    return block_reduce_sum(padded)


# endregion block_reduce
