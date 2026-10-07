"""稀疏注意力：块级 top-k 选择（NSA 的选择分支）、闪电索引器 + token 级 top-k（DSA），
以及沿序列维压缩 KV 的压缩注意力（DeepSeek-V4 的 HCA/CSA 思路）。

张量布局：q[..., T, d]，k/v[..., S, d]，前导维度（批、头）任意；这里只处理 prefill
（T == S，位置从 0 开始），足以检验选择逻辑与稠密注意力的关系。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """mask 为 True 的位置参与 softmax；mask 可带前导维度。"""
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    return scores.masked_fill(~mask, float("-inf")).softmax(-1) @ v


def causal_mask(T: int, S: int | None = None) -> torch.Tensor:
    S = T if S is None else S
    return torch.ones(T, S, dtype=torch.bool).tril(diagonal=S - T)


# region block_topk
def block_topk_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         block_size: int, top_k: int) -> torch.Tensor:
    """每个查询选 top_k 个键块，只在被选块内做因果注意力。

    块的重要性 = q · mean(块内的键)（NSA 用压缩分支的注意力分数，这里用均值池化代替）。
    当前查询所在的块总是被选中，保证至少能看到自己；未来块分数为 -inf，永不入选。
    """
    T = q.shape[-2]
    n_blocks = math.ceil(T / block_size)
    pad = n_blocks * block_size - T
    k_blocks = F.pad(k, (0, 0, 0, pad)).unflatten(-2, (n_blocks, block_size))
    counts = torch.full((n_blocks,), block_size)
    counts[-1] = block_size - pad
    k_mean = k_blocks.sum(-2) / counts.unsqueeze(-1)  # [..., n_blocks, d]
    block_scores = q @ k_mean.transpose(-1, -2)  # [..., T, n_blocks]
    query_block = torch.arange(T) // block_size
    blocks = torch.arange(n_blocks)
    future = blocks.unsqueeze(0) > query_block.unsqueeze(1)  # [T, n_blocks]
    block_scores = block_scores.masked_fill(future, float("-inf"))
    own = blocks.unsqueeze(0) == query_block.unsqueeze(1)
    block_scores = block_scores.masked_fill(own, float("inf"))
    chosen = torch.topk(block_scores, min(top_k, n_blocks), dim=-1).indices
    block_mask = torch.zeros_like(block_scores, dtype=torch.bool).scatter(-1, chosen, True)
    block_mask &= ~future
    token_mask = block_mask.repeat_interleave(block_size, dim=-1)[..., :T]  # 块 → token
    return attend(q, k, v, token_mask & causal_mask(T))
# endregion


# region indexer
def lightning_indexer(q_idx: torch.Tensor, w_idx: torch.Tensor, k_idx: torch.Tensor) -> torch.Tensor:
    """DSA 的闪电索引器 I_{t,s} = Σ_j w_{t,j} · ReLU(q_{t,j} · k_s)。

    q_idx[T, H_I, d_I] 为少量索引头的查询，w_idx[T, H_I] 为每个索引头的权重，
    k_idx[S, d_I] 为所有索引头共享的键。返回 [T, S] 的索引分数（未加因果掩码）。
    """
    return torch.einsum("th,ths->ts", w_idx, F.relu(torch.einsum("thd,sd->ths", q_idx, k_idx)))


def topk_token_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         index_scores: torch.Tensor, top_k: int) -> torch.Tensor:
    """按索引分数为每个查询选 top_k 个过去的 token，主注意力只在它们上面计算。"""
    T = q.shape[-2]
    causal = causal_mask(T)
    scores = index_scores.masked_fill(~causal, float("-inf"))
    chosen = torch.topk(scores, min(top_k, T), dim=-1).indices
    mask = torch.zeros(T, T, dtype=torch.bool).scatter(-1, chosen, True) & causal
    return attend(q, k, v, mask)
# endregion


# region compress
def compress_kv(c: torch.Tensor, z: torch.Tensor, pos_bias: torch.Tensor, m: int) -> torch.Tensor:
    """把每 m 个 token 的 KV 条目压成一个（DeepSeek-V4 HCA 的不重叠压缩）。

    c[S, d] 是待压缩的 KV 条目，z[S, d] 是压缩权重 logits，pos_bias[m, d] 是块内可学习位置偏置。
    块内按维度做 softmax 得到权重 S_j，再求 Σ_j S_j ⊙ c_j。S 必须是 m 的整数倍。
    """
    S, d = c.shape
    weights = (z.view(S // m, m, d) + pos_bias).softmax(dim=1)
    return (weights * c.view(S // m, m, d)).sum(dim=1)  # [S/m, d]


def compressed_window_attention(q: torch.Tensor, kv_comp: torch.Tensor, kv: torch.Tensor,
                                m: int, window: int) -> torch.Tensor:
    """共享 KV（K = V，MQA 方式）的压缩 + 滑动窗口注意力，两组条目放在同一个 softmax 里。

    q[T, d]；kv_comp[S/m, d] 为压缩条目，第 i 个覆盖位置 [i·m, (i+1)·m)，
    只有当整块都已经过去（(i+1)·m - 1 <= t）时查询 t 才能看到它；
    kv[S, d] 为未压缩条目，查询 t 只看最近 window 个（含自身）。
    """
    T = q.shape[0]
    t = torch.arange(T).unsqueeze(1)
    block_end = (torch.arange(kv_comp.shape[0]) + 1) * m - 1
    comp_mask = block_end.unsqueeze(0) <= t
    j = torch.arange(kv.shape[0]).unsqueeze(0)
    win_mask = (j <= t) & (j > t - window)
    keys = torch.cat([kv_comp, kv], dim=0)
    return attend(q, keys, keys, torch.cat([comp_mask, win_mask], dim=1))
# endregion


def entries_per_query(t: int, m: int, window: int, top_k: int | None = None) -> int:
    """位置 t 的查询要读多少个 KV 条目：完整压缩块数（CSA 再截到 top_k）+ 窗口。"""
    comp = (t + 1) // m
    if top_k is not None:
        comp = min(comp, top_k)
    return comp + min(window, t + 1)
