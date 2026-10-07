import torch
from torch import nn

from llms_from_scratch.distributed.comm import launch
from llms_from_scratch.distributed.ddp import NaiveDDP
from llms_from_scratch.distributed.zero import FSDPLinear, ShardedOptimizer, zero_memory_per_gpu

STEPS, BATCH = 3, 12


def _mlp() -> nn.Sequential:
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(6, 10, bias=False), nn.Tanh(), nn.Linear(10, 3, bias=False))


def _data():
    g = torch.Generator().manual_seed(1)
    return torch.randn(STEPS, BATCH, 6, generator=g), torch.randn(STEPS, BATCH, 3, generator=g)


def _reference() -> list[torch.Tensor]:
    model = _mlp()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    xs, ys = _data()
    for x, y in zip(xs, ys):
        opt.zero_grad()
        ((model(x) - y) ** 2).mean().backward()
        opt.step()
    return [model[0].weight.detach(), model[2].weight.detach()]


def _zero_worker(rank: int, world_size: int) -> dict:
    xs, ys = _data()
    per_rank = BATCH // world_size
    sl = slice(rank * per_rank, (rank + 1) * per_rank)

    # ZeRO-1：DDP 同步梯度 + 优化器状态分片
    model = _mlp()
    ddp = NaiveDDP(model)
    opt = ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=1e-2, weight_decay=0.1)
    for x, y in zip(xs, ys):
        opt.zero_grad()
        ((ddp(x[sl]) - y[sl]) ** 2).mean().backward()
        ddp.sync_gradients()
        opt.step()
    local_state = sum(t.numel() for s in opt.local_optimizer.state.values()
                      for t in s.values() if t.dim() > 0)
    zero1 = [model[0].weight.detach(), model[2].weight.detach()]

    # ZeRO-3：参数、梯度、优化器状态全部只存 1/p
    full = _mlp()
    sharded = nn.Sequential(FSDPLinear(full[0]), nn.Tanh(), FSDPLinear(full[2]))
    opt3 = torch.optim.AdamW(sharded.parameters(), lr=1e-2, weight_decay=0.1)
    for x, y in zip(xs, ys):
        opt3.zero_grad()
        ((sharded(x[sl]) - y[sl]) ** 2).mean().backward()
        opt3.step()  # 每个 rank 只更新自己的分片
    zero3 = [sharded[0].full_weight(), sharded[2].full_weight()]
    shard_numel = sum(p.numel() for p in sharded.parameters())
    return {"zero1": zero1, "zero3": zero3, "local_state": local_state, "shard_numel": shard_numel}


def test_zero1_and_zero3_match_single_process():
    reference = _reference()
    results = launch(_zero_worker, 2)
    for out in results:
        for got, want in zip(out["zero1"], reference):
            torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-5)
        for got, want in zip(out["zero3"], reference):
            torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-5)
        assert out["shard_numel"] == (60 + 30) // 2
    # 两个参数（60 与 30 个元素）分给两个 rank：每个 rank 只为自己的参数保存 Adam 的两个矩
    assert sorted(out["local_state"] for out in results) == [2 * 30, 2 * 60]


def test_zero_memory_formula_matches_paper():
    # ZeRO 论文图 1：Ψ = 7.5B，N_d = 64，K = 12
    gb = 1e9
    psi, n = 7.5e9, 64
    assert zero_memory_per_gpu(psi, n, 0) / gb == 120
    assert round(zero_memory_per_gpu(psi, n, 1) / gb, 2) == 31.41
    assert round(zero_memory_per_gpu(psi, n, 2) / gb, 2) == 16.64
    assert round(zero_memory_per_gpu(psi, n, 3) / gb, 2) == 1.88
