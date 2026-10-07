"""分块 GEMM 的 CPU 模拟器：按 block / k-tile / thread 的顺序计算 C = A @ B，并统计访存。

统计量（单位：元素个数）：
* ``global_loads``：从全局内存（HBM，经 L2）读入的元素数；
* ``smem_loads``：从共享内存读入寄存器的元素数；
* ``fmas``：乘加次数，恒等于 M·N·K。
模拟器把越界部分补 0（对应 kernel 中的边界判断），所以任意形状都能算对。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GemmStats:
    global_loads: int = 0
    smem_loads: int = 0
    fmas: int = 0


# region traffic_model
def gemm_global_loads(m: int, n: int, k: int, bm: int, bn: int) -> int:
    """输出按 BM×BN 分块时，A 被读 N/BN 遍、B 被读 M/BM 遍（假设整除、无缓存命中）。"""
    return m * k * (n // bn) + k * n * (m // bm)


def tile_intensity(bm: int, bn: int, elem_bytes: int = 4) -> float:
    """一个 BM×BN 输出块在每个 k 步上的算术强度（FLOP / 全局字节）。

    每个 k 读入 BM + BN 个元素，做 BM·BN 次乘加 = 2·BM·BN FLOP。
    """
    return 2 * bm * bn / ((bm + bn) * elem_bytes)


# endregion traffic_model


def _tile(x: np.ndarray, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    """取 x[r0:r0+h, c0:c0+w]，越界补 0（kernel 中的 `row < M ? a[...] : 0`）。"""
    out = np.zeros((h, w), dtype=x.dtype)
    sub = x[r0 : r0 + h, c0 : c0 + w]
    out[: sub.shape[0], : sub.shape[1]] = sub
    return out


# region naive
def gemm_naive_sim(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, GemmStats]:
    """每个线程算 C 的一个元素，直接从全局内存读 A 的一行与 B 的一列。"""
    m, k = a.shape
    n = b.shape[1]
    c = np.zeros((m, n), dtype=np.float32)
    stats = GemmStats()
    for i in range(m):
        for j in range(n):
            acc = np.float32(0)
            for p in range(k):
                acc += a[i, p] * b[p, j]
            c[i, j] = acc
            stats.global_loads += 2 * k
            stats.fmas += k
    return c, stats


# endregion naive


# region smem_tiled
def gemm_smem_tiled_sim(
    a: np.ndarray, b: np.ndarray, tile: int = 16
) -> tuple[np.ndarray, GemmStats]:
    """共享内存分块：block 负责 C 的 tile×tile 子块，每个线程一个输出元素。

    沿 K 每次把 A、B 的一个 tile×tile 子块搬进共享内存，block 内所有线程复用它。
    """
    m, k = a.shape
    n = b.shape[1]
    c = np.zeros((m, n), dtype=np.float32)
    stats = GemmStats()
    for bi in range(0, m, tile):  # blockIdx.y
        for bj in range(0, n, tile):  # blockIdx.x
            acc = np.zeros((tile, tile), dtype=np.float32)  # 每线程一个寄存器
            for p0 in range(0, k, tile):
                a_s = _tile(a, bi, p0, tile, tile)  # 每个线程搬一个元素
                b_s = _tile(b, p0, bj, tile, tile)
                stats.global_loads += 2 * tile * tile
                # __syncthreads()
                for ty in range(tile):
                    for tx in range(tile):
                        acc[ty, tx] += a_s[ty, :] @ b_s[:, tx]
                        stats.smem_loads += 2 * tile
                        stats.fmas += tile
                # __syncthreads()
            sub = c[bi : bi + tile, bj : bj + tile]
            sub[...] = acc[: sub.shape[0], : sub.shape[1]]
    return c, stats


# endregion smem_tiled


# region register_tiled
def gemm_register_tiled_sim(
    a: np.ndarray,
    b: np.ndarray,
    bm: int = 32,
    bn: int = 32,
    bk: int = 8,
    tm: int = 4,
    tn: int = 4,
) -> tuple[np.ndarray, GemmStats]:
    """二维寄存器分块：block 算 BM×BN，每个线程算 TM×TN 的小块（thread tile）。

    对每个 k，线程从共享内存读 TM 个 A 元素与 TN 个 B 元素到寄存器，
    做 TM·TN 次乘加（外积）。共享内存读取量因此从 2 次/FMA 降到 (TM+TN)/(TM·TN)。
    """
    m, k = a.shape
    n = b.shape[1]
    c = np.zeros((m, n), dtype=np.float32)
    stats = GemmStats()
    for bi in range(0, m, bm):
        for bj in range(0, n, bn):
            acc = np.zeros((bm // tm, bn // tn, tm, tn), dtype=np.float32)
            for p0 in range(0, k, bk):
                a_s = _tile(a, bi, p0, bm, bk)  # 共享内存 As[BM][BK]
                b_s = _tile(b, p0, bj, bk, bn)  # 共享内存 Bs[BK][BN]
                stats.global_loads += bm * bk + bk * bn
                for ty in range(bm // tm):  # 线程 (ty, tx)
                    for tx in range(bn // tn):
                        for p in range(bk):
                            reg_a = a_s[ty * tm : (ty + 1) * tm, p]  # TM 个寄存器
                            reg_b = b_s[p, tx * tn : (tx + 1) * tn]  # TN 个寄存器
                            acc[ty, tx] += np.outer(reg_a, reg_b)
                            stats.smem_loads += tm + tn
                            stats.fmas += tm * tn
            block = acc.transpose(0, 2, 1, 3).reshape(bm, bn)
            sub = c[bi : bi + bm, bj : bj + bn]
            sub[...] = block[: sub.shape[0], : sub.shape[1]]
    return c, stats


# endregion register_tiled


# region mma_sim
def gemm_mma_sim(a: np.ndarray, b: np.ndarray, shape: int = 16) -> tuple[np.ndarray, int]:
    """Tensor Core 视角：GEMM 被拆成一串 16×16×16 的 mma（D = A·B + C）。

    输入先舍入到 FP16，累加保持 FP32——这正是 WMMA / mma.sync 的常见配置。
    返回结果与 mma 指令条数。
    """
    m, k = a.shape
    n = b.shape[1]
    a16 = a.astype(np.float16)
    b16 = b.astype(np.float16)
    c = np.zeros((m, n), dtype=np.float32)
    count = 0
    for i in range(0, m, shape):
        for j in range(0, n, shape):
            frag_c = np.zeros((shape, shape), dtype=np.float32)  # 累加器片段
            for p in range(0, k, shape):
                frag_a = _tile(a16, i, p, shape, shape).astype(np.float32)
                frag_b = _tile(b16, p, j, shape, shape).astype(np.float32)
                frag_c += frag_a @ frag_b  # 一条 mma
                count += 1
            sub = c[i : i + shape, j : j + shape]
            sub[...] = frag_c[: sub.shape[0], : sub.shape[1]]
    return c, count


# endregion mma_sim
