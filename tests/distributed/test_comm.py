import torch
import torch.distributed as dist

from llms_from_scratch.distributed.comm import (
    chunk_bounds,
    launch,
    ring_all_reduce,
    ring_all_reduce_time,
    tree_all_reduce_time,
)


def _collectives_worker(rank: int, world_size: int) -> dict:
    """每个 rank 跑一遍各种集合通信与手写环形 all-reduce，把结果交回父进程检查。"""
    out = {}
    x = torch.arange(10, dtype=torch.float64) * (rank + 1)  # 10 个元素，p=3 时不能整除

    ring = x.clone()
    out["ring_sent"] = ring_all_reduce(ring)
    ref = x.clone()
    dist.all_reduce(ref)
    out["ring"], out["all_reduce"] = ring, ref

    b = torch.full((2,), float(rank))
    dist.broadcast(b, src=1)
    out["broadcast"] = b

    r = torch.full((2,), float(rank + 1))
    dist.reduce(r, dst=0)
    out["reduce"] = r

    gathered = [torch.empty(1) for _ in range(world_size)]
    dist.all_gather(gathered, torch.tensor([float(rank)]))
    out["all_gather"] = torch.cat(gathered)

    shard = torch.empty(1)
    dist.reduce_scatter(shard, [torch.tensor([10.0 * rank + j]) for j in range(world_size)])
    out["reduce_scatter"] = shard

    a2a_out = torch.empty(world_size)
    dist.all_to_all_single(a2a_out, torch.tensor([10.0 * rank + j for j in range(world_size)]))
    out["all_to_all"] = a2a_out
    return out


def test_collectives_and_ring_all_reduce():
    p = 3
    results = launch(_collectives_worker, p)
    expected_sum = torch.arange(10, dtype=torch.float64) * sum(range(1, p + 1))
    for rank, out in enumerate(results):
        torch.testing.assert_close(out["ring"], expected_sum)
        torch.testing.assert_close(out["ring"], out["all_reduce"])
        torch.testing.assert_close(out["broadcast"], torch.full((2,), 1.0))
        torch.testing.assert_close(out["all_gather"], torch.arange(p, dtype=torch.float32))
        # rank r 收到所有 rank 第 r 段之和：sum_j (10 j + r)
        assert out["reduce_scatter"].item() == sum(10.0 * j + rank for j in range(p))
        # all-to-all 是转置：rank r 收到第 j 个元素来自 rank j 的第 r 段
        torch.testing.assert_close(out["all_to_all"], torch.tensor([10.0 * j + rank for j in range(p)]))
    torch.testing.assert_close(results[0]["reduce"], torch.full((2,), 6.0))
    sizes = [e - s for s, e in chunk_bounds(10, p)]
    for rank, out in enumerate(results):
        # reduce-scatter 第 k 步发第 (rank-k-1) 段，all-gather 第 k 步发第 (rank-k) 段
        rs = sum(sizes[(rank - k - 1) % p] for k in range(p - 1))
        ag = sum(sizes[(rank - k) % p] for k in range(p - 1))
        assert out["ring_sent"] == rs + ag


def test_ring_volume_when_divisible():
    # n 能被 p 整除时每个 rank 恰好发送 2(p-1)/p · n 个元素
    for p, n in [(2, 8), (4, 12)]:
        sizes = [e - s for s, e in chunk_bounds(n, p)]
        assert sum(sizes[: p - 1]) * 2 == 2 * (p - 1) * n // p


def test_chunk_bounds_cover_buffer():
    bounds = chunk_bounds(10, 3)
    assert bounds == [(0, 4), (4, 7), (7, 10)]


def test_cost_model_crossover():
    alpha, bw = 5e-6, 50e9  # 5 微秒延迟、50 GB/s：跨节点 InfiniBand 的量级
    p = 256
    # 小消息：延迟项主导，树的 2·log2(p)·α 远小于环的 2(p-1)·α
    assert tree_all_reduce_time(1e3, p, alpha, bw) < ring_all_reduce_time(1e3, p, alpha, bw)
    # 卡数多时环的 2(p-1)·α 很大：即使 1 GB 的消息，256 卡上树仍然更快
    assert tree_all_reduce_time(1e9, p, alpha, bw) < ring_all_reduce_time(1e9, p, alpha, bw)
    # 卡数少、消息大：带宽项主导，环的 2(p-1)/p·n/B 优于树的 2n/B
    assert ring_all_reduce_time(1e9, 8, alpha, bw) < tree_all_reduce_time(1e9, 8, alpha, bw)
    # 环的带宽项随 p 趋于 2n/B，与卡数几乎无关
    t8 = ring_all_reduce_time(1e9, 8, 0.0, bw)
    t1024 = ring_all_reduce_time(1e9, 1024, 0.0, bw)
    assert abs(t8 / (2e9 / bw) - 7 / 8) < 1e-12 and t1024 < 2e9 / bw
