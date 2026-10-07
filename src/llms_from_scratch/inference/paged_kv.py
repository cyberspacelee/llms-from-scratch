"""分页 KV Cache：块表、槽位映射，以及读取分页缓存的注意力。

物理缓存的布局与 vLLM 相同：每层一对张量 [num_blocks, block_size, n_kv_heads, head_dim]。
一个序列的第 p 个 token 存在物理块 block_table[p // block_size] 的第 p % block_size 个槽里。
"""

from __future__ import annotations

import math

import torch


# region slot
def slot_mapping(block_table: list[int], positions: list[int], block_size: int) -> list[int]:
    """逻辑位置 → 扁平化的物理槽号 block_id · block_size + offset。"""
    return [block_table[p // block_size] * block_size + p % block_size for p in positions]
# endregion slot


def allocate_cache(num_blocks: int, block_size: int, n_kv_heads: int, head_dim: int,
                   dtype: torch.dtype = torch.float32, device=None) -> tuple[torch.Tensor, ...]:
    shape = (num_blocks, block_size, n_kv_heads, head_dim)
    return (torch.zeros(shape, dtype=dtype, device=device),
            torch.zeros(shape, dtype=dtype, device=device))


# region write
def write_kv(k_cache: torch.Tensor, v_cache: torch.Tensor, slots: torch.Tensor,
             k: torch.Tensor, v: torch.Tensor) -> None:
    """把本步 N 个 token 的 K/V（形状 [N, n_kv_heads, head_dim]）写进 slots 指定的槽。"""
    num_blocks, block_size = k_cache.shape[:2]
    k_cache.view(num_blocks * block_size, *k_cache.shape[2:]).index_copy_(0, slots, k)
    v_cache.view(num_blocks * block_size, *v_cache.shape[2:]).index_copy_(0, slots, v)
# endregion write


# region reference
def paged_attention_reference(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                              block_tables: list[list[int]], seq_lens: list[int]) -> torch.Tensor:
    """逐块读取的参考实现（与 vLLM PagedAttention kernel 的循环结构相同）。

    q: [N, n_heads, head_dim]，第 n 行查询能看到它所属序列的前 seq_lens[n] 个键。
    对每一行，沿块表一块一块地取 K/V，用在线 softmax 累积，从不拼出连续的 K/V。
    """
    n_rows, n_heads, head_dim = q.shape
    block_size, n_kv_heads = k_cache.shape[1], k_cache.shape[2]
    group = n_heads // n_kv_heads
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.empty_like(q)
    for n in range(n_rows):
        qn = q[n].float() * scale  # [H, D]
        m = torch.full((n_heads,), -math.inf)  # 每个头到目前为止的最大分数
        denom = torch.zeros(n_heads)  # softmax 分母
        acc = torch.zeros(n_heads, head_dim)  # 未归一化的加权和
        for logical in range(math.ceil(seq_lens[n] / block_size)):
            valid = min(block_size, seq_lens[n] - logical * block_size)
            block = block_tables[n][logical]
            k = k_cache[block, :valid].float().repeat_interleave(group, dim=1)  # [valid, H, D]
            v = v_cache[block, :valid].float().repeat_interleave(group, dim=1)
            scores = torch.einsum("hd,thd->ht", qn, k)
            m_new = torch.maximum(m, scores.max(-1).values)
            correction = torch.exp(m - m_new)
            p = torch.exp(scores - m_new[:, None])
            denom = denom * correction + p.sum(-1)
            acc = acc * correction[:, None] + torch.einsum("ht,thd->hd", p, v)
            m = m_new
        out[n] = (acc / denom[:, None]).to(q.dtype)
    return out
# endregion reference


# region gather
def paged_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                    block_tables: torch.Tensor, seq_lens: torch.Tensor) -> torch.Tensor:
    """张量化的分页注意力：按块表把每行可见的块 gather 出来再做带掩码的注意力。

    q: [N, H, D]；block_tables: [N, max_blocks]（每行一个块表，可重复）；seq_lens: [N]。
    所有形状只依赖 N 与 max_blocks，不依赖数据，因此可以被 CUDA Graph 捕获。
    """
    n_rows, n_heads, head_dim = q.shape
    block_size, n_kv_heads = k_cache.shape[1], k_cache.shape[2]
    max_blocks = block_tables.shape[1]
    k = k_cache[block_tables].reshape(n_rows, max_blocks * block_size, n_kv_heads, head_dim)
    v = v_cache[block_tables].reshape(n_rows, max_blocks * block_size, n_kv_heads, head_dim)
    group = n_heads // n_kv_heads
    if group > 1:
        k = k.repeat_interleave(group, dim=2)
        v = v.repeat_interleave(group, dim=2)
    key_pos = torch.arange(max_blocks * block_size, device=q.device)
    mask = key_pos[None, :] < seq_lens[:, None]  # [N, S]：只看本序列已写入的位置
    out = torch.nn.functional.scaled_dot_product_attention(
        q[:, :, None, :],  # [N, H, 1, D]：每行是一个独立的“批元素”
        k.transpose(1, 2), v.transpose(1, 2),  # [N, H, S, D]
        attn_mask=mask[:, None, None, :],
    )
    return out[:, :, 0, :]
# endregion gather


# region waste
def contiguous_waste(lengths: list[int], max_len: int) -> float:
    """按最大长度为每个请求预留连续空间时，被预留但没用上的槽位比例。"""
    return 1 - sum(lengths) / (len(lengths) * max_len)


def paged_waste(lengths: list[int], block_size: int) -> float:
    """分页时只有每个序列最后一块的空槽是浪费。"""
    reserved = sum(math.ceil(n / block_size) * block_size for n in lengths)
    return 1 - sum(lengths) / reserved
# endregion waste
