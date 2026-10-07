"""张量并行（Megatron 式）、序列并行与 Ring Attention。

约定：整个进程组就是一个张量并行组，大小 t = world_size。所有并行模块都由一个完整的
单卡模块构造（from_full），各 rank 只取走属于自己的那一片权重，因此测试可以直接与单卡
模块比较前向输出与梯度。
"""

from __future__ import annotations

import copy
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from llms_from_scratch.transformer.model import RMSNorm, apply_rope

# region conjugate


def _all_reduce(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    dist.all_reduce(x)
    return x


class _CopyToTP(torch.autograd.Function):
    """f 算子：前向恒等（每个 rank 拿到同一份输入），反向 all-reduce（各 rank 的输入梯度是部分和）。"""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return _all_reduce(grad)


class _ReduceFromTP(torch.autograd.Function):
    """g 算子：前向 all-reduce（把行切分的部分和加起来），反向恒等。f 与 g 互为共轭。"""

    @staticmethod
    def forward(ctx, x):
        return _all_reduce(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


def copy_to_tp(x: torch.Tensor) -> torch.Tensor:
    return _CopyToTP.apply(x)


def reduce_from_tp(x: torch.Tensor) -> torch.Tensor:
    return _ReduceFromTP.apply(x)

# endregion


# region sequence_ops


def _gather_along(x: torch.Tensor, dim: int) -> torch.Tensor:
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, x.contiguous())
    return torch.cat(parts, dim=dim)


def _reduce_scatter_along(x: torch.Tensor, dim: int) -> torch.Tensor:
    chunks = [c.contiguous() for c in x.chunk(dist.get_world_size(), dim=dim)]
    out = torch.empty_like(chunks[0])
    dist.reduce_scatter(out, chunks)
    return out


class _GatherSeq(torch.autograd.Function):
    """序列并行进入张量并行区：前向沿序列维 all-gather，反向 reduce-scatter（取代 f）。"""

    @staticmethod
    def forward(ctx, x):
        return _gather_along(x, dim=1)

    @staticmethod
    def backward(ctx, grad):
        return _reduce_scatter_along(grad, dim=1)


class _ReduceScatterSeq(torch.autograd.Function):
    """离开张量并行区：前向沿序列维 reduce-scatter，反向 all-gather（取代 g）。"""

    @staticmethod
    def forward(ctx, x):
        return _reduce_scatter_along(x, dim=1)

    @staticmethod
    def backward(ctx, grad):
        return _gather_along(grad, dim=1)


def gather_sequence(x: torch.Tensor) -> torch.Tensor:
    return _GatherSeq.apply(x)


def reduce_scatter_sequence(x: torch.Tensor) -> torch.Tensor:
    return _ReduceScatterSeq.apply(x)

# endregion


def _shard_rows(weight: torch.Tensor, rank: int, t: int) -> torch.Tensor:
    """nn.Linear 的 weight 形状是 [out, in]：沿输出维（行）切，等价于对 W^T 列切分。"""
    return weight.detach().chunk(t, dim=0)[rank].clone()


def _shard_cols(weight: torch.Tensor, rank: int, t: int) -> torch.Tensor:
    return weight.detach().chunk(t, dim=1)[rank].clone()


# region linear


class ColumnParallelLinear(nn.Module):
    """Y = X Wᵀ，按输出特征切分：rank i 计算 Y 的第 i 段列。输入需完整，输出是分片。"""

    def __init__(self, full: nn.Linear) -> None:
        super().__init__()
        rank, t = dist.get_rank(), dist.get_world_size()
        self.weight = nn.Parameter(_shard_rows(full.weight, rank, t))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(copy_to_tp(x), self.weight)


class RowParallelLinear(nn.Module):
    """Y = X Wᵀ，按输入特征切分：rank i 只持有 X 的第 i 段，算出部分和，g 算子把它们加起来。"""

    def __init__(self, full: nn.Linear, reduce_output: bool = True) -> None:
        super().__init__()
        rank, t = dist.get_rank(), dist.get_world_size()
        self.weight = nn.Parameter(_shard_cols(full.weight, rank, t))
        self.reduce_output = reduce_output

    def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
        partial = F.linear(x_shard, self.weight)
        return reduce_from_tp(partial) if self.reduce_output else partial

# endregion


# region mlp


class TPSwiGLU(nn.Module):
    """W1、W3 列切分，W2 行切分：中间激活 [.., d_ff/t] 全程留在本地，整层只在末尾 all-reduce 一次。"""

    def __init__(self, full: nn.Module, reduce_output: bool = True) -> None:
        super().__init__()
        rank, t = dist.get_rank(), dist.get_world_size()
        self.w1 = nn.Parameter(_shard_rows(full.w1.weight, rank, t))
        self.w3 = nn.Parameter(_shard_rows(full.w3.weight, rank, t))
        self.w2 = nn.Parameter(_shard_cols(full.w2.weight, rank, t))
        self.reduce_output = reduce_output

    def local(self, x: torch.Tensor) -> torch.Tensor:
        """输入完整的 x，返回本 rank 的部分和（尚未跨 rank 相加）。"""
        return F.linear(F.silu(F.linear(x, self.w1)) * F.linear(x, self.w3), self.w2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.local(copy_to_tp(x))
        return reduce_from_tp(out) if self.reduce_output else out

# endregion


# region attention


class TPAttention(nn.Module):
    """按注意力头切分：Q/K/V 投影列切分（每个 rank 拿 h/t 个查询头与 h_kv/t 个 KV 头），O 投影行切分。"""

    def __init__(self, full: nn.Module) -> None:
        super().__init__()
        rank, t = dist.get_rank(), dist.get_world_size()
        if full.n_kv_heads % t:
            raise ValueError("KV 头数必须能被张量并行度整除")
        self.n_heads, self.n_kv_heads = full.n_heads // t, full.n_kv_heads // t
        self.head_dim = full.head_dim
        self.wq = nn.Parameter(_shard_rows(full.q_proj.weight, rank, t))
        self.wk = nn.Parameter(_shard_rows(full.k_proj.weight, rank, t))
        self.wv = nn.Parameter(_shard_rows(full.v_proj.weight, rank, t))
        self.wo = nn.Parameter(_shard_cols(full.o_proj.weight, rank, t))

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        x = copy_to_tp(x)
        q = F.linear(x, self.wq).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = F.linear(x, self.wk).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = F.linear(x, self.wv).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        repeat = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        partial = F.linear(out.transpose(1, 2).reshape(B, T, -1), self.wo)
        return reduce_from_tp(partial)

# endregion


# region vocab_ce


class _VocabParallelCrossEntropy(torch.autograd.Function):
    """logits 按词表切分时的交叉熵：只交换每个 token 的 3 个标量（最大值、指数和、目标 logit）。"""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, target: torch.Tensor, vocab_start: int):
        logits = logits.float()
        local_max = logits.max(dim=-1).values
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX)
        shifted = logits - local_max.unsqueeze(-1)
        exp = shifted.exp()
        sum_exp = exp.sum(dim=-1)
        dist.all_reduce(sum_exp)
        local_target = target - vocab_start
        in_shard = (local_target >= 0) & (local_target < logits.shape[-1])
        safe = local_target.clamp(0, logits.shape[-1] - 1)
        target_logit = shifted.gather(-1, safe.unsqueeze(-1)).squeeze(-1) * in_shard
        dist.all_reduce(target_logit)  # 目标 token 只落在一个 rank 的分片里
        loss = sum_exp.log() - target_logit
        ctx.save_for_backward(exp / sum_exp.unsqueeze(-1), safe, in_shard)
        return loss.mean()

    @staticmethod
    def backward(ctx, grad):
        softmax, safe, in_shard = ctx.saved_tensors
        g = softmax.clone()
        g.scatter_add_(-1, safe.unsqueeze(-1), -in_shard.to(g.dtype).unsqueeze(-1))
        return g * grad / softmax.shape[0], None, None


def vocab_parallel_cross_entropy(logits_shard: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """logits_shard: [N, V/t]，是第 rank 段词表的 logits；target: [N] 全局词表下标。"""
    vocab_start = dist.get_rank() * logits_shard.shape[-1]
    return _VocabParallelCrossEntropy.apply(logits_shard, target, vocab_start)

# endregion


# region sequence_parallel


class SequenceParallelFFN(nn.Module):
    """Pre-Norm 前馈子层 x + FFN(Norm(x)) 的张量 + 序列并行版本。

    输入、输出与残差都沿序列维切分（[B, T/t, d]），RMSNorm 在本地分片上计算；
    进入 FFN 前 all-gather 序列，离开时 reduce-scatter——与张量并行的一次 all-reduce 通信量相同。
    注意 norm.weight 在各 rank 上是复制的，它的梯度只来自本地序列分片，需要再 all-reduce。
    """

    def __init__(self, norm: RMSNorm, ffn: nn.Module) -> None:
        super().__init__()
        self.norm = copy.deepcopy(norm)  # 归一化层的权重在每个 rank 上复制一份
        self.ffn = TPSwiGLU(ffn)

    def forward(self, x_shard: torch.Tensor) -> torch.Tensor:
        h = gather_sequence(self.norm(x_shard))  # [B, T, d]，每个 rank 都是完整序列
        partial = self.ffn.local(h)  # 本 rank 的部分和
        return x_shard + reduce_scatter_sequence(partial)

    def sync_replicated_grads(self) -> None:
        dist.all_reduce(self.norm.weight.grad)

# endregion


# region ring_attention


def _block_attention(q, k, v, mask):
    """一块注意力：返回未归一化合并所需的 (输出, logsumexp)。q,k,v: [B, H, T_blk, d_h]。"""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1, keepdim=True)
    return torch.softmax(scores, dim=-1) @ v, lse


def ring_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """因果 Ring Attention（前向）：rank i 持有第 i 段序列的 Q/K/V，K/V 沿环传递 t-1 次。

    第 s 步本 rank 拿到第 j = (i - s) mod t 段的 K/V：j < i 时整块可见，j = i 时用块内因果掩码，
    j > i 时整块被掩掉（直接跳过计算）。各块结果用在线 softmax 的 logsumexp 公式合并。
    """
    rank, t = dist.get_rank(), dist.get_world_size()
    T = q.shape[-2]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    out, lse = _block_attention(q, k, v, causal)  # 第 0 步：对角块
    for step in range(1, t):
        # 把手里的 K/V 传给右邻居，从左邻居收下一块（真实实现中这一步与本块计算重叠）
        k_next, v_next = torch.empty_like(k), torch.empty_like(v)
        right, left = (rank + 1) % t, (rank - 1) % t
        reqs = [dist.isend(k.contiguous(), right, tag=0), dist.isend(v.contiguous(), right, tag=1)]
        dist.recv(k_next, left, tag=0)
        dist.recv(v_next, left, tag=1)
        for r in reqs:
            r.wait()
        k, v = k_next, v_next
        j = (rank - step) % t
        if j > rank:
            continue  # 未来的 token：因果掩码下整块不可见
        block_out, block_lse = _block_attention(q, k, v, None)
        new_lse = torch.logaddexp(lse, block_lse)
        out = out * (lse - new_lse).exp() + block_out * (block_lse - new_lse).exp()
        lse = new_lse
    return out

# endregion
