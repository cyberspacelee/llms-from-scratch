import pytest

from llms_from_scratch.inference.prefix_cache import (
    BlockPool,
    KVCacheManager,
    RadixCache,
    block_hashes,
)


def test_block_hash_depends_on_prefix():
    a = block_hashes([1, 2, 3, 4, 5, 6, 7, 8], 4)
    b = block_hashes([9, 9, 9, 9, 5, 6, 7, 8], 4)
    assert len(a) == 2 and a[1] != b[1]  # 同样的第二块内容，前缀不同则哈希不同
    assert len(block_hashes([1, 2, 3, 4, 5], 4)) == 1  # 不满的块不哈希


def test_pool_reserves_null_block_and_recycles():
    pool = BlockPool(5)
    assert pool.num_free == 4
    blocks = pool.get_new_blocks(4)
    assert 0 not in [b.block_id for b in blocks]
    with pytest.raises(RuntimeError):
        pool.get_new_blocks(1)
    pool.free_blocks(blocks)
    assert pool.num_free == 4 and all(b.ref_cnt == 0 for b in blocks)


def _run_prefill(m: KVCacheManager, rid: str, tokens: list[int]) -> int:
    hits, n_hit = m.get_computed_blocks(tokens)
    assert m.allocate_slots(rid, len(tokens), hits)
    m.cache_blocks(rid, tokens, len(tokens))
    return n_hit


def test_prefix_hit_shares_blocks_and_refcounts():
    m = KVCacheManager(num_blocks=10, block_size=4)
    system = list(range(8))
    assert _run_prefill(m, "a", system + [100, 101, 102]) == 0
    assert _run_prefill(m, "b", system + [200]) == 8
    assert m.block_ids("a")[:2] == m.block_ids("b")[:2]
    shared = m.req_blocks["a"][0]
    assert shared.ref_cnt == 2
    m.free("a")
    assert shared.ref_cnt == 1
    m.free("b")
    assert shared.ref_cnt == 0
    # 引用归零后块仍留在缓存里，新请求依然能命中
    assert _run_prefill(m, "c", system + [300, 301]) == 8
    m.free("c")
    assert m.num_free_blocks == 9


def test_full_hit_leaves_last_token_to_compute():
    m = KVCacheManager(num_blocks=10, block_size=4)
    tokens = list(range(8))
    _run_prefill(m, "a", tokens)
    m.free("a")
    hits, n_hit = m.get_computed_blocks(tokens)
    assert n_hit == 4  # 第二块包含最后一个 token，不能命中


def test_lru_eviction_of_cached_blocks():
    m = KVCacheManager(num_blocks=5, block_size=2)  # 4 个可用块
    _run_prefill(m, "a", [1, 2, 3, 4])
    m.free("a")
    _run_prefill(m, "b", [5, 6, 7, 8])
    m.free("b")
    # 新请求需要 2 个块：按 LRU 先逐出最早释放的 a 的两个块，b 的缓存不受影响
    assert m.allocate_slots("c", 4)
    assert m.get_computed_blocks([5, 6, 7, 8, 9])[1] == 4  # b 的前缀仍在
    assert m.get_computed_blocks([1, 2, 3, 4, 9])[1] == 0  # a 已被逐出


def test_allocation_failure_is_atomic():
    m = KVCacheManager(num_blocks=3, block_size=4)
    assert m.allocate_slots("a", 8)
    assert not m.allocate_slots("b", 1)
    assert "b" not in m.req_blocks and m.num_free_blocks == 0
    m.free("a")
    assert m.allocate_slots("b", 1)


def test_copy_on_write_after_fork():
    m = KVCacheManager(num_blocks=8, block_size=4)
    assert m.allocate_slots("p", 6)
    m.fork("p", "c")
    first, last = m.req_blocks["p"]
    assert first.ref_cnt == 2 and last.ref_cnt == 2
    # 子序列在位置 6 写新 token：最后一块被共享，必须先复制
    src, dst = m.prepare_write("c", 6)
    assert src == last.block_id and dst != src
    assert last.ref_cnt == 1 and m.req_blocks["c"][1].ref_cnt == 1
    assert m.prepare_write("p", 6) is None  # 父序列现在独占原块
    m.free("p")
    m.free("c")
    assert m.num_free_blocks == 7


def test_radix_cache_split_match_and_evict():
    tree = RadixCache()
    assert tree.insert([1, 2, 3, 4], [10, 11, 12, 13]) == 0
    assert tree.insert([1, 2, 7, 8], [10, 11, 22, 23]) == 2
    slots, node = tree.match_prefix([1, 2, 7, 9])
    assert slots == [10, 11, 22]
    assert tree.total_tokens() == 6  # [1,2] + [3,4] + [7] + [8]
    assert sorted(tree.root.children[1].children) == [3, 7]
    # 锁住 [1,2,7] 这条路径，淘汰只能动 [3,4] 与 [8]
    tree.lock(node)
    freed = tree.evict(100)
    assert sorted(freed) == [12, 13, 23]
    tree.lock(node, -1)
    assert sorted(tree.evict(100)) == [10, 11, 22]
    assert tree.total_tokens() == 0
