"""ZeRO：把优化器状态、梯度与参数切分到数据并行的各个 rank 上。

- zero_memory_per_gpu：ZeRO 论文中三个阶段的每卡模型状态显存公式；
- ShardedOptimizer：ZeRO-1（优化器状态分片），每个参数只由一个 rank 负责更新，更新后广播；
- FSDPLinear：ZeRO-3 / FSDP 的最小实现，每个 rank 只常驻权重的 1/p，前向与反向时临时
  all-gather 完整权重，反向得到的权重梯度用 reduce-scatter 直接落到各自的分片上。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer

# region memory


def zero_memory_per_gpu(num_params: float, world_size: int, stage: int, k: int = 12) -> float:
    """混合精度 Adam 下每卡模型状态的字节数（ZeRO 论文 3.1 节的记法）。

    参数与梯度各用 2 字节（BF16/FP16），优化器状态 K 字节/参数（FP32 主参数 + 两个矩，K=12）。
    stage 0 是普通 DDP：(2 + 2 + K)Ψ；1 切优化器状态；2 再切梯度；3 再切参数。
    """
    psi, n = num_params, world_size
    if stage == 0:
        return (2 + 2 + k) * psi
    if stage == 1:
        return 2 * psi + 2 * psi + k * psi / n
    if stage == 2:
        return 2 * psi + (2 + k) * psi / n
    if stage == 3:
        return (2 + 2 + k) * psi / n
    raise ValueError("stage 只能是 0、1、2、3")

# endregion


# region sharded_optimizer


class ShardedOptimizer(Optimizer):
    """ZeRO-1：包装任意逐元素优化器，每个 rank 只为自己负责的那部分参数保存状态。

    参数按“当前负载最小的 rank 优先”贪心分配，使各 rank 负责的元素数大致相等。
    step() 中每个 rank 只更新自己的参数，然后由负责的 rank 把新参数 broadcast 给其他 rank。
    调用前梯度必须已经在所有 rank 间平均（例如由 DDP 完成）。
    """

    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs: Any) -> None:
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        self.owner: dict[nn.Parameter, int] = {}
        self._load = [0] * self.world_size
        self._optimizer_cls, self._kwargs = optimizer_cls, kwargs
        self.local_optimizer: Optimizer | None = None
        super().__init__(params, defaults=kwargs)  # 会对每个参数组调用 add_param_group

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        super().add_param_group(param_group)
        mine = []
        for p in param_group["params"]:
            owner = min(range(self.world_size), key=lambda r: self._load[r])
            self.owner[p] = owner
            self._load[owner] += p.numel()
            if owner == self.rank:
                mine.append(p)
        group = {k: v for k, v in param_group.items() if k != "params"}
        if self.local_optimizer is None:
            self.local_optimizer = self._optimizer_cls(mine, **{**self._kwargs, **group})
        else:
            self.local_optimizer.add_param_group({**group, "params": mine})

    @torch.no_grad()
    def step(self, closure=None, **kwargs):
        loss = self.local_optimizer.step(closure, **kwargs)
        for group in self.param_groups:
            for p in group["params"]:
                dist.broadcast(p.data, src=self.owner[p])
        return loss

# endregion


# region fsdp


def _all_gather_flat(shard: torch.Tensor) -> torch.Tensor:
    parts = [torch.empty_like(shard) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, shard.contiguous())
    return torch.cat(parts)


class _ShardedLinearFn(torch.autograd.Function):
    """前向：all-gather 权重 → 线性层 → 丢弃完整权重；反向：再次 all-gather → 求梯度 → reduce-scatter。"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, shard: torch.Tensor, out_features: int, in_features: int):
        weight = _all_gather_flat(shard)[: out_features * in_features].view(out_features, in_features)
        ctx.save_for_backward(x, shard)  # 只保存分片，不保存完整权重（reshard after forward）
        ctx.dims = (out_features, in_features)
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, shard = ctx.saved_tensors
        out_features, in_features = ctx.dims
        p = dist.get_world_size()
        weight = _all_gather_flat(shard)[: out_features * in_features].view(out_features, in_features)
        grad_x = grad_out @ weight
        grad_w = grad_out.reshape(-1, out_features).T @ x.reshape(-1, in_features)
        flat = torch.zeros(shard.numel() * p, dtype=grad_w.dtype)
        flat[: grad_w.numel()] = grad_w.reshape(-1)
        grad_shard = torch.empty_like(shard)
        dist.reduce_scatter(grad_shard, list(flat.chunk(p)))  # 每个 rank 只拿到自己分片的梯度之和
        return grad_x, grad_shard / p, None, None


class FSDPLinear(nn.Module):
    """全分片的无偏置线性层：每个 rank 只常驻展平权重（补零到 p 的倍数）的第 rank 段。"""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        if linear.bias is not None:
            raise ValueError("为保持简洁，FSDPLinear 只支持无偏置线性层")
        rank, p = dist.get_rank(), dist.get_world_size()
        self.out_features, self.in_features = linear.weight.shape
        numel = linear.weight.numel()
        padded = torch.zeros(-(-numel // p) * p)
        padded[:numel] = linear.weight.detach().reshape(-1)
        self.shard = nn.Parameter(padded.chunk(p)[rank].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _ShardedLinearFn.apply(x, self.shard, self.out_features, self.in_features)

    @torch.no_grad()
    def full_weight(self) -> torch.Tensor:
        numel = self.out_features * self.in_features
        return _all_gather_flat(self.shard)[:numel].view(self.out_features, self.in_features)

# endregion
