"""专家并行：MoE 层的 dispatch / combine 用 all-to-all 实现。

- MoE：单进程参考实现（top-k 路由 + SwiGLU 专家），专家循环写法，便于对照；
- ExpertParallelMoE：每个 rank 持有 E/p 个专家与一份复制的路由器，本地 token 经 all-to-all
  发到专家所在的 rank，算完再经 all-to-all 发回并按门控权重合并；
- node_limited_topk：DeepSeek-V3 的节点受限路由，每个 token 最多去 M 个节点；
- 容量与负载统计、all-to-all 通信量估算。
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from torch import nn

from llms_from_scratch.transformer.model import SwiGLU

# region moe


def topk_route(logits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """logits: [N, E] → (weights [N, k], expert_ids [N, k])，权重是 top-k 内重新归一化的 softmax。"""
    weights, ids = logits.softmax(dim=-1).topk(k, dim=-1)
    return weights / weights.sum(dim=-1, keepdim=True), ids


class MoE(nn.Module):
    """y = Σ_{e ∈ topk(x)} g_e(x) · Expert_e(x)，门控 g 是 top-k 内重新归一化的 softmax。"""

    def __init__(self, d_model: int, d_ff: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d_model, d_ff) for _ in range(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, ids = topk_route(self.router(x), self.top_k)
        y = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            token, slot = (ids == e).nonzero(as_tuple=True)
            if token.numel():
                y = y.index_add(0, token, weights[token, slot, None] * expert(x[token]))
        return y

# endregion


# region all_to_all


class _AllToAll(torch.autograd.Function):
    """可求导的变长 all-to-all：发给 rank j 的是 x 中连续的 send_counts[j] 行。

    反向就是把梯度按原路发回：交换 send_counts 与 recv_counts 再做一次 all-to-all。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, send_counts: list[int], recv_counts: list[int]):
        ctx.counts = (send_counts, recv_counts)
        out = x.new_empty((sum(recv_counts), *x.shape[1:]))
        dist.all_to_all_single(out, x.contiguous(), recv_counts, send_counts)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        send_counts, recv_counts = ctx.counts
        out = grad.new_empty((sum(send_counts), *grad.shape[1:]))
        dist.all_to_all_single(out, grad.contiguous(), send_counts, recv_counts)
        return out, None, None


def all_to_all(x: torch.Tensor, send_counts: list[int], recv_counts: list[int]) -> torch.Tensor:
    return _AllToAll.apply(x, send_counts, recv_counts)

# endregion


# region expert_parallel


class ExpertParallelMoE(nn.Module):
    """专家并行的 MoE：rank r 持有第 r·E/p … (r+1)·E/p - 1 号专家。"""

    def __init__(self, moe: MoE) -> None:
        super().__init__()
        rank, p = dist.get_rank(), dist.get_world_size()
        num_experts = len(moe.experts)
        if num_experts % p:
            raise ValueError("专家数必须能被专家并行度整除")
        self.top_k, self.per_rank = moe.top_k, num_experts // p
        self.router = moe.router  # 路由器在每个 rank 上复制（属于数据并行的部分）
        self.experts = nn.ModuleList(moe.experts[rank * self.per_rank:(rank + 1) * self.per_rank])
        self.first = rank * self.per_rank
        self.last_counts: tuple[list[int], list[int]] | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p, N, k = dist.get_world_size(), x.shape[0], self.top_k
        weights, ids = topk_route(self.router(x), k)  # [N, k]
        # 1) 每个 (token, 专家) 副本按目标 rank 排序，相同目标的行连续存放
        flat_ids = ids.reshape(-1)
        dest = flat_ids // self.per_rank
        order = torch.argsort(dest, stable=True)
        token_of = torch.arange(N).repeat_interleave(k)[order]
        send_counts = torch.bincount(dest, minlength=p)
        # 2) 先交换“每个 rank 要发多少行”，接收方才能分配缓冲区
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts)
        send, recv = send_counts.tolist(), recv_counts.tolist()
        self.last_counts = (send, recv)
        # 3) dispatch：token 隐藏状态与专家编号一起发出
        recv_x = all_to_all(x[token_of], send, recv)
        recv_ids = flat_ids.new_empty(sum(recv))
        dist.all_to_all_single(recv_ids, flat_ids[order].contiguous(), recv, send)
        # 4) 本地专家计算
        local_out = torch.zeros_like(recv_x)
        for i, expert in enumerate(self.experts):
            rows = (recv_ids == self.first + i).nonzero(as_tuple=True)[0]
            if rows.numel():
                local_out = local_out.index_copy(0, rows, expert(recv_x[rows]))
        # 5) combine：结果原路发回，恢复排序前的顺序，再按门控权重加权求和
        back = all_to_all(local_out, recv, send)
        unsorted = torch.empty_like(back).index_copy(0, order, back)
        return (unsorted.view(N, k, -1) * weights.unsqueeze(-1)).sum(dim=1)

# endregion


# region routing


def node_limited_topk(scores: torch.Tensor, num_nodes: int, max_nodes: int,
                      k: int) -> torch.Tensor:
    """DeepSeek-V3 的节点受限路由：先选出得分最高的 max_nodes 个节点，再在其中选 top-k 专家。

    scores: [N, E]，专家按节点连续编号（每个节点 E/num_nodes 个）。节点得分是该节点上
    最高的 k/max_nodes 个专家亲和度之和。返回专家编号 [N, k]。
    """
    N, E = scores.shape
    per_node = E // num_nodes
    grouped = scores.view(N, num_nodes, per_node)
    node_score = grouped.topk(k // max_nodes, dim=-1).values.sum(dim=-1)  # [N, num_nodes]
    nodes = node_score.topk(max_nodes, dim=-1).indices
    allowed = torch.zeros(N, num_nodes, dtype=torch.bool).scatter(1, nodes, True)
    masked = scores.masked_fill(~allowed.repeat_interleave(per_node, dim=1), float("-inf"))
    return masked.topk(k, dim=-1).indices


def expert_capacity(num_tokens: int, num_experts: int, k: int, capacity_factor: float) -> int:
    """每个专家每步最多处理的 token 副本数：C = ceil(cf · N·k / E)。超出的副本被丢弃或改道。"""
    return math.ceil(capacity_factor * num_tokens * k / num_experts)


def dropped_fraction(expert_ids: torch.Tensor, num_experts: int, capacity: int) -> float:
    """按到达顺序填满每个专家的容量后，被丢弃的 token 副本占比。"""
    load = torch.bincount(expert_ids.reshape(-1), minlength=num_experts)
    return (load - capacity).clamp(min=0).sum().item() / expert_ids.numel()


def all_to_all_bytes(tokens: int, k: int, d_model: int, dispatch_bytes: float,
                     combine_bytes: float, ep: int) -> float:
    """一个 rank 在一层 MoE 前向中发出的字节数（dispatch + combine），假设专家均匀分布。

    每个 token 有 k 个副本，其中平均 (ep-1)/ep 去往其他 rank（留在本 rank 的不经过网络）。
    """
    remote = tokens * k * (ep - 1) / ep
    return remote * d_model * (dispatch_bytes + combine_bytes)

# endregion
