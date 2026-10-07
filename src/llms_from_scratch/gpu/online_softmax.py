"""softmax 的三种写法与“行 softmax kernel”的 CPU 模拟。

* 三遍：求最大值 → 求指数和 → 归一化，读输入三次。
* 在线（online）：一遍同时维护运行最大值 m 与运行和 ℓ（代码中记作 ell），读输入两次（第二遍写输出）。
* 合并：两段的 (m, ell) 状态可以按 ``merge_states`` 结合，因此可以并行、分块、树形归约。
"""

from __future__ import annotations

import math

import numpy as np


# region three_pass
def softmax_three_pass(x: np.ndarray) -> np.ndarray:
    m = -math.inf
    for v in x:  # 第 1 遍：最大值
        m = max(m, v)
    s = 0.0
    for v in x:  # 第 2 遍：指数和
        s += math.exp(v - m)
    return np.array([math.exp(v - m) / s for v in x])  # 第 3 遍：归一化


# endregion three_pass


# region online
def online_max_sum(x: np.ndarray) -> tuple[float, float]:
    """一遍扫描得到 m = max(x) 与 ell = Σ exp(x - m)。"""
    m, ell = -math.inf, 0.0
    for v in x:
        m_new = max(m, v)
        if m_new == -math.inf:  # 迄今全是 -inf（被掩码），状态不变
            continue
        # 旧的和是相对旧最大值算的，换到新最大值要乘 exp(m - m_new)
        ell = ell * math.exp(m - m_new) + math.exp(v - m_new)
        m = m_new
    return m, ell


def softmax_online(x: np.ndarray) -> np.ndarray:
    m, ell = online_max_sum(x)
    return np.array([math.exp(v - m) / ell for v in x])


def merge_states(m1: float, l1: float, m2: float, l2: float) -> tuple[float, float]:
    """合并两段的状态：(m1, l1) ⊕ (m2, l2)。满足结合律与交换律。"""
    m = max(m1, m2)
    if m == -math.inf:  # 两段都为空
        return m, 0.0
    return m, l1 * math.exp(m1 - m) + l2 * math.exp(m2 - m)


# endregion online


# region row_softmax_sim
def row_softmax_block_sim(x: np.ndarray, block_dim: int = 32) -> np.ndarray:
    """模拟“一个 block 处理一行”的 softmax kernel。

    1. 线程 t 以 block_dim 为步长扫描本行，维护自己的在线状态 (m_t, ell_t)；
    2. 用树形归约（与 warp shuffle 同样的配对方式）合并成整行的 (m, ell)；
    3. 每个线程再扫一遍，写出 exp(x - m) / ell。
    """
    rows, cols = x.shape
    out = np.empty_like(x, dtype=np.float64)
    for r in range(rows):
        row = x[r]
        ms = [-math.inf] * block_dim
        ls = [0.0] * block_dim
        for t in range(block_dim):
            ms[t], ls[t] = online_max_sum(row[t::block_dim])
        offset = block_dim // 2
        while offset > 0:  # 每轮 lane t 与 lane t+offset 合并
            for t in range(offset):
                ms[t], ls[t] = merge_states(ms[t], ls[t], ms[t + offset], ls[t + offset])
            offset //= 2
        m, ell = ms[0], ls[0]
        out[r] = np.exp(row - m) / ell
    return out


# endregion row_softmax_sim
