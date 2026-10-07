"""FlashAttention 的 CPU 分块参考实现：与 GPU kernel 相同的分块顺序、在线 softmax 与重计算反向。

张量布局：``q, k, v`` 形状 ``[N, T, d]``（N 把 batch 与 head 合并）。
前向返回输出 ``O`` 与每行的 logsumexp ``L = m + log ℓ``（ℓ 在代码中记作 ell）；反向只用 ``Q, K, V, O, dO, L``，
不保存 T×T 的注意力矩阵，而是在每个块里重新计算 ``P``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


# region naive_attention
def naive_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False
) -> torch.Tensor:
    """标准注意力：S、P 都是完整的 T×T 矩阵，会被写回并重新读出 HBM。"""
    scale = 1.0 / math.sqrt(q.shape[-1])
    s = q @ k.transpose(-2, -1) * scale
    if causal:
        t_q, t_k = s.shape[-2:]
        mask = torch.ones(t_q, t_k, dtype=torch.bool, device=q.device).triu(1)
        s = s.masked_fill(mask, float("-inf"))
    return torch.softmax(s, dim=-1) @ v


def standard_attention_hbm_elements(t: int, d: int) -> int:
    """标准注意力（三个独立 kernel）的 HBM 读写量，单位：元素。

    S = QKᵀ：读 Q、K（2Td），写 S（T²）；softmax：读 S、写 P（2T²）；
    O = PV：读 P、V（T² + Td），写 O（Td）。合计 4Td + 4T²。
    """
    return 4 * t * d + 4 * t * t


def flash_attention_hbm_elements(t: int, d: int, block_q: int) -> int:
    """FlashAttention-2 前向（非因果）的 HBM 读写量：Q 读一次、O 写一次，

    每个 Q 块都要把 K、V 完整读一遍，共 T/Br 遍。
    """
    n_q_blocks = math.ceil(t / block_q)
    return 2 * t * d + 2 * t * d * n_q_blocks


# endregion naive_attention


@dataclass
class FlashStats:
    kv_tile_loads: int = 0  # 读入 K_j、V_j 的块次数
    skipped_tiles: int = 0  # 因果掩码下整块跳过的次数


# region flash_forward
def flash_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_q: int = 16,
    block_k: int = 16,
    stats: FlashStats | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FlashAttention-2 前向。外层循环 Q 块（GPU 上每个 Q 块是一个 program），内层循环 K/V 块。"""
    n, t_q, d = q.shape
    t_k = k.shape[1]
    scale = 1.0 / math.sqrt(d)
    o = torch.empty_like(q)
    lse = torch.empty(n, t_q, dtype=q.dtype, device=q.device)
    for i0 in range(0, t_q, block_q):
        qi = q[:, i0 : i0 + block_q]  # 从 HBM 读一次 Q_i，留在片上
        rows = torch.arange(i0, i0 + qi.shape[1], device=q.device)
        m = torch.full((n, qi.shape[1]), float("-inf"), dtype=q.dtype, device=q.device)  # 运行最大值
        ell = torch.zeros(n, qi.shape[1], dtype=q.dtype, device=q.device)  # 运行和（未归一化）
        acc = torch.zeros_like(qi)  # 未归一化的输出
        for j0 in range(0, t_k, block_k):
            if causal and j0 > rows[-1]:  # 整块都在对角线右上方：直接跳过
                if stats:
                    stats.skipped_tiles += 1
                continue
            kj, vj = k[:, j0 : j0 + block_k], v[:, j0 : j0 + block_k]
            if stats:
                stats.kv_tile_loads += 1
            s = qi @ kj.transpose(-2, -1) * scale  # [n, Br, Bc]，只在片上
            if causal:
                cols = torch.arange(j0, j0 + kj.shape[1], device=q.device)
                s = s.masked_fill(cols[None, :] > rows[:, None], float("-inf"))
            m_new = torch.maximum(m, s.amax(-1))
            p = torch.exp(s - m_new[..., None])
            alpha = torch.exp(m - m_new)  # 旧状态的缩放因子
            ell = alpha * ell + p.sum(-1)
            acc = alpha[..., None] * acc + p @ vj
            m = m_new
        o[:, i0 : i0 + block_q] = acc / ell[..., None]  # 最后才除以 ℓ（FA2 的改进之一）
        lse[:, i0 : i0 + block_q] = m + torch.log(ell)
    return o, lse


# endregion flash_forward


# region flash_backward
def flash_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    causal: bool = False,
    block_q: int = 16,
    block_k: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """重计算反向。外层循环 K/V 块（dK_j、dV_j 在片上累加），内层循环 Q 块。

    P_ij = exp(S_ij - L_i) 由保存的 logsumexp 直接重算，不需要再求最大值与和。
    """
    n, t_q, d = q.shape
    t_k = k.shape[1]
    scale = 1.0 / math.sqrt(d)
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    delta = (do * o).sum(-1)  # D_i = rowsum(dO ∘ O) = Σ_j P_ij dP_ij
    for j0 in range(0, t_k, block_k):
        kj, vj = k[:, j0 : j0 + block_k], v[:, j0 : j0 + block_k]
        cols = torch.arange(j0, j0 + kj.shape[1], device=q.device)
        dkj = torch.zeros_like(kj)
        dvj = torch.zeros_like(vj)
        for i0 in range(0, t_q, block_q):
            rows = torch.arange(i0, min(i0 + block_q, t_q), device=q.device)
            if causal and j0 > rows[-1]:
                continue
            qi, doi = q[:, i0 : i0 + block_q], do[:, i0 : i0 + block_q]
            s = qi @ kj.transpose(-2, -1) * scale
            if causal:
                s = s.masked_fill(cols[None, :] > rows[:, None], float("-inf"))
            p = torch.exp(s - lse[:, i0 : i0 + block_q, None])  # 重计算 P
            dvj += p.transpose(-2, -1) @ doi
            dp = doi @ vj.transpose(-2, -1)
            ds = p * (dp - delta[:, i0 : i0 + block_q, None])  # softmax 的 VJP
            dq[:, i0 : i0 + block_q] += ds @ kj * scale  # GPU 上用原子加或单独一趟
            dkj += ds.transpose(-2, -1) @ qi * scale
        dk[:, j0 : j0 + block_k] = dkj
        dv[:, j0 : j0 + block_k] = dvj
    return dq, dk, dv


# endregion flash_backward
