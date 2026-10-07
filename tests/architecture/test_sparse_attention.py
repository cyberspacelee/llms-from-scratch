import torch

from llms_from_scratch.architecture.sparse_attention import (
    attend,
    block_topk_attention,
    causal_mask,
    compress_kv,
    compressed_window_attention,
    entries_per_query,
    lightning_indexer,
    topk_token_attention,
)


def qkv(T=13, d=8, heads=2):
    torch.manual_seed(0)
    return (torch.randn(heads, T, d) for _ in range(3))


def test_block_topk_with_all_blocks_is_dense():
    q, k, v = qkv()
    dense = attend(q, k, v, causal_mask(13))
    torch.testing.assert_close(block_topk_attention(q, k, v, block_size=4, top_k=4), dense)


def test_block_topk_restricts_to_selected_blocks():
    q, k, v = qkv(T=16)
    out = block_topk_attention(q, k, v, block_size=4, top_k=1)
    # top_k=1 时只剩查询自己所在的块：等价于“块内因果”注意力
    blockwise = causal_mask(16) & (torch.arange(16)[:, None] // 4 == torch.arange(16)[None] // 4)
    torch.testing.assert_close(out, attend(q, k, v, blockwise))


def test_indexer_topk_all_is_dense():
    q, k, v = qkv(heads=1)
    q, k, v = q[0], k[0], v[0]
    torch.manual_seed(1)
    scores = lightning_indexer(torch.randn(13, 4, 6), torch.rand(13, 4), torch.randn(13, 6))
    assert scores.shape == (13, 13)
    torch.testing.assert_close(topk_token_attention(q, k, v, scores, top_k=13),
                               attend(q, k, v, causal_mask(13)))


def test_indexer_formula():
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # 1 个查询，2 个索引头
    w = torch.tensor([[2.0, 3.0]])
    k = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
    # 键 0：2·ReLU(1) + 3·ReLU(-1) = 2；键 1：2·0.5 + 3·2 = 7
    torch.testing.assert_close(lightning_indexer(q, w, k), torch.tensor([[2.0, 7.0]]))


def test_topk_selects_highest_scoring_past_tokens():
    T = 6
    scores = torch.arange(T, dtype=torch.float).flip(0).expand(T, T).clone()  # 越早的 token 分数越高
    v = torch.eye(T)
    q, k = torch.zeros(T, T), torch.zeros(T, T)  # 主注意力分数全相等：输出 = 被选 token 的平均
    out = topk_token_attention(q, k, v, scores, top_k=2)
    torch.testing.assert_close(out[5], torch.tensor([0.5, 0.5, 0, 0, 0, 0]))


def test_compress_uniform_weights_is_mean_pooling():
    c = torch.randn(12, 4)
    out = compress_kv(c, torch.zeros(12, 4), torch.zeros(4, 4), m=4)
    torch.testing.assert_close(out, c.view(3, 4, 4).mean(1))


def test_compressed_attention_with_m1_and_no_window_is_dense():
    torch.manual_seed(0)
    q, kv = torch.randn(9, 8), torch.randn(9, 8)
    out = compressed_window_attention(q, kv, kv, m=1, window=0)
    torch.testing.assert_close(out, attend(q, kv, kv, causal_mask(9)))


def test_entries_per_query():
    # V4-Pro 的 CSA：m=4、top-k=1024、窗口 128；位置 1M 处每个查询只读 1152 个条目
    assert entries_per_query(1_000_000 - 1, 4, 128, top_k=1024) == 1152
    # HCA：m'=128，没有 top-k，位置 1M 处读 7812 + 128 个条目
    assert entries_per_query(1_000_000 - 1, 128, 128) == 7812 + 128
