import torch
import torch.distributed as dist

from llms_from_scratch.distributed.comm import launch
from llms_from_scratch.distributed.expert_parallel import (
    ExpertParallelMoE,
    MoE,
    all_to_all_bytes,
    dropped_fraction,
    expert_capacity,
    node_limited_topk,
)

D, FF, E, K, N = 8, 16, 8, 2, 12  # 每个 rank 12 个 token


def _setup(world_size: int):
    torch.manual_seed(0)
    moe = MoE(D, FF, E, K).double()
    x = torch.randn(world_size * N, D, dtype=torch.float64)
    upstream = torch.randn(world_size * N, D, dtype=torch.float64)  # 随机的上游梯度
    return moe, x, upstream


def _ep_worker(rank: int, world_size: int) -> dict:
    moe, x, upstream = _setup(world_size)
    ep = ExpertParallelMoE(moe)
    local_x = x[rank * N:(rank + 1) * N].clone().requires_grad_()
    y = ep(local_x)
    (y * upstream[rank * N:(rank + 1) * N]).sum().backward()
    dist.all_reduce(ep.router.weight.grad)  # 路由器是复制的：梯度要在 rank 间求和
    return {
        "y": y.detach(), "x_grad": local_x.grad,
        "router_grad": ep.router.weight.grad,
        "expert_grads": {ep.first + i: [p.grad for p in e.parameters()]
                         for i, e in enumerate(ep.experts)},
        "counts": ep.last_counts,
    }


def test_expert_parallel_matches_single_process():
    world_size = 4
    moe, x, upstream = _setup(world_size)
    x_ref = x.clone().requires_grad_()
    y_ref = moe(x_ref)
    (y_ref * upstream).sum().backward()
    results = launch(_ep_worker, world_size)
    for rank, out in enumerate(results):
        rows = slice(rank * N, (rank + 1) * N)
        torch.testing.assert_close(out["y"], y_ref.detach()[rows])
        torch.testing.assert_close(out["x_grad"], x_ref.grad[rows])
        torch.testing.assert_close(out["router_grad"], moe.router.weight.grad)
        for e, grads in out["expert_grads"].items():
            for got, want in zip(grads, moe.experts[e].parameters()):
                torch.testing.assert_close(got, want.grad)
        send, recv = out["counts"]
        assert sum(send) == N * K  # 每个 token 发出 k 份副本
    # 所有 rank 发出的总行数等于收到的总行数
    assert sum(sum(o["counts"][0]) for o in results) == sum(sum(o["counts"][1]) for o in results)


def test_node_limited_routing():
    torch.manual_seed(0)
    scores = torch.rand(64, 32).sigmoid()
    nodes, per_node = 8, 4
    ids = node_limited_topk(scores, num_nodes=nodes, max_nodes=2, k=4)
    assert ids.shape == (64, 4)
    assert all(len(set((row // per_node).tolist())) <= 2 for row in ids)  # 每个 token 至多 2 个节点
    # 不限制节点数时退化为普通 top-k
    free = node_limited_topk(scores, num_nodes=nodes, max_nodes=nodes, k=8)
    assert torch.equal(free.sort(-1).values, scores.topk(8, dim=-1).indices.sort(-1).values)


def test_capacity_and_volume():
    assert expert_capacity(4096, 64, 8, 1.25) == 640
    ids = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 1]])  # 专家 0 收到 4 份，容量 2 时丢 2 份
    assert dropped_fraction(ids, 4, capacity=2) == 2 / 8
    # DeepSeek-V3 式配置：4096 token、top-8、7168 维、FP8 dispatch + BF16 combine、EP=64
    sent = all_to_all_bytes(4096, 8, 7168, 1, 2, 64)
    assert round(sent / 2**20) == round(4096 * 8 * 63 / 64 * 7168 * 3 / 2**20)
