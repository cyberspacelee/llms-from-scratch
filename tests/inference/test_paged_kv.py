import torch
import torch.nn.functional as F

from llms_from_scratch.inference.paged_kv import (
    allocate_cache,
    contiguous_waste,
    paged_attention,
    paged_attention_reference,
    paged_waste,
    slot_mapping,
    write_kv,
)


def test_slot_mapping_example():
    # 块大小 4，块表 [5, 7]：位置 4、5 落在第二个逻辑块 → 物理块 7
    assert slot_mapping([5, 7], [0, 3, 4, 5], 4) == [20, 23, 28, 29]


def _scatter_sequences(lengths, block_size, n_kv, d, num_blocks, seed=0):
    """把若干条连续的 K/V 序列按打乱的物理块写进分页缓存。"""
    g = torch.Generator().manual_seed(seed)
    k_cache, v_cache = allocate_cache(num_blocks, block_size, n_kv, d)
    perm = torch.randperm(num_blocks - 1, generator=g).add(1).tolist()  # 块 0 保留
    tables, ks, vs = [], [], []
    for n in lengths:
        need = -(-n // block_size)
        table, perm = perm[:need], perm[need:]
        k = torch.randn(n, n_kv, d, generator=g)
        v = torch.randn(n, n_kv, d, generator=g)
        slots = torch.tensor(slot_mapping(table, list(range(n)), block_size))
        write_kv(k_cache, v_cache, slots, k, v)
        tables.append(table)
        ks.append(k)
        vs.append(v)
    return k_cache, v_cache, tables, ks, vs


def test_paged_attention_matches_contiguous():
    lengths, bs, h, n_kv, d = [5, 1, 12, 8], 4, 4, 2, 8
    k_cache, v_cache, tables, ks, vs = _scatter_sequences(lengths, bs, n_kv, d, num_blocks=16)
    q = torch.randn(len(lengths), h, d)
    expected = []
    for i, n in enumerate(lengths):
        k = ks[i].repeat_interleave(h // n_kv, dim=1).transpose(0, 1)  # [H, n, D]
        v = vs[i].repeat_interleave(h // n_kv, dim=1).transpose(0, 1)
        expected.append(F.scaled_dot_product_attention(q[i][:, None], k, v)[:, 0])
    expected = torch.stack(expected)

    ref = paged_attention_reference(q, k_cache, v_cache, tables, lengths)
    torch.testing.assert_close(ref, expected, atol=1e-5, rtol=1e-5)

    width = max(len(t) for t in tables)
    padded = torch.tensor([t + [0] * (width - len(t)) for t in tables])
    gathered = paged_attention(q, k_cache, v_cache, padded, torch.tensor(lengths))
    torch.testing.assert_close(gathered, expected, atol=1e-5, rtol=1e-5)


def test_paged_attention_causal_rows_of_one_sequence():
    """同一序列的多行查询（prefill）各自只看前缀：等价于因果注意力。"""
    bs, h, d, n = 4, 2, 8, 10
    k_cache, v_cache, tables, ks, vs = _scatter_sequences([n], bs, h, d, num_blocks=8, seed=1)
    q = torch.randn(n, h, d)
    table = torch.tensor(tables[0]).expand(n, -1)
    out = paged_attention(q, k_cache, v_cache, table, torch.arange(1, n + 1))
    expected = F.scaled_dot_product_attention(q.transpose(0, 1), ks[0].transpose(0, 1),
                                              vs[0].transpose(0, 1), is_causal=True)
    torch.testing.assert_close(out, expected.transpose(0, 1), atol=1e-5, rtol=1e-5)


def test_waste():
    lengths = [100, 300, 37, 900]
    assert contiguous_waste(lengths, 2048) > 0.8
    assert paged_waste(lengths, 16) < 0.03
    # 分页浪费的上界：每个序列不超过 block_size - 1 个槽
    reserved = sum(-(-n // 16) * 16 for n in lengths)
    assert reserved - sum(lengths) <= len(lengths) * 15
