"""稀疏混合专家层（MoE）：top-k 路由、负载均衡损失、容量丢弃、共享专家与无辅助损失偏置。

专家就是 ``transformer.model.SwiGLU``；一个 MoE 层 = 路由器 + E 个路由专家 + 若干共享专家。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from llms_from_scratch.transformer.model import SwiGLU


@dataclass
class MoEConfig:
    d_model: int = 32
    d_expert: int = 64  # 每个专家的隐藏宽度（细粒度专家取得更小）
    n_experts: int = 8
    top_k: int = 2
    n_shared: int = 0  # DeepSeekMoE 的共享专家，每个 token 都经过
    score: str = "softmax"  # "softmax"（Switch/Mixtral）或 "sigmoid"（DeepSeek-V3）
    capacity_factor: float | None = None  # None 表示不丢 token


@dataclass
class Routing:
    indices: torch.Tensor  # [N, k] 每个 token 选中的专家
    gates: torch.Tensor  # [N, k] 对应的组合权重
    probs: torch.Tensor  # [N, E] 全部专家的归一化分数（用于负载均衡损失）
    kept: torch.Tensor  # [N, k] 是否在专家容量之内


# region router
class Router(nn.Module):
    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.randn(cfg.n_experts, cfg.d_model) * 0.02)
        # 无辅助损失负载均衡的偏置：只影响“选谁”，不影响“乘多少”，不参与梯度
        self.register_buffer("bias", torch.zeros(cfg.n_experts))

    def forward(self, x: torch.Tensor) -> Routing:
        logits = x @ self.weight.t()  # [N, E]
        if self.cfg.score == "softmax":
            scores = logits.softmax(-1)
        else:
            scores = torch.sigmoid(logits)
        _, indices = torch.topk(scores + self.bias, self.cfg.top_k, dim=-1)
        gates = scores.gather(-1, indices)
        gates = gates / gates.sum(-1, keepdim=True)  # 在选中的 k 个专家内重新归一化
        probs = scores / scores.sum(-1, keepdim=True)
        kept = torch.ones_like(indices, dtype=torch.bool)
        if self.cfg.capacity_factor is not None:
            kept = capacity_mask(indices, self.cfg.n_experts, self.cfg.capacity_factor)
        return Routing(indices, gates, probs, kept)
# endregion


# region capacity
def capacity_mask(indices: torch.Tensor, n_experts: int, capacity_factor: float) -> torch.Tensor:
    """每个专家最多接收 C = ceil(cf · N · k / E) 个分配，按 token 顺序先到先得，超出的被丢弃。"""
    N, k = indices.shape
    capacity = math.ceil(capacity_factor * N * k / n_experts)
    onehot = F.one_hot(indices.reshape(-1), n_experts)  # [N·k, E]，按 (token, 槽位) 顺序排列
    rank = (onehot.cumsum(0) * onehot).sum(-1) - 1  # 该分配是这个专家收到的第几个
    return (rank < capacity).view(N, k)
# endregion


class MoE(nn.Module):
    def __init__(self, cfg: MoEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.router = Router(cfg)
        self.experts = nn.ModuleList(SwiGLU(cfg.d_model, cfg.d_expert) for _ in range(cfg.n_experts))
        self.shared = nn.ModuleList(SwiGLU(cfg.d_model, cfg.d_expert) for _ in range(cfg.n_shared))
        self.last_routing: Routing | None = None

    # region forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x = x.reshape(-1, shape[-1])  # [N, d]，N = B·T
        r = self.router(x)
        self.last_routing = r
        out = torch.zeros_like(x)
        for e, expert in enumerate(self.experts):
            token, slot = torch.where((r.indices == e) & r.kept)  # 分发：选出发给专家 e 的 token
            if token.numel():
                y = expert(x[token]) * r.gates[token, slot].unsqueeze(-1)
                out.index_add_(0, token, y)  # 合并：按门控权重加回原位置
        for expert in self.shared:
            out = out + expert(x)
        return out.view(shape)
    # endregion


def moe_reference(moe: MoE, x: torch.Tensor) -> torch.Tensor:
    """逐 token 的参考实现：y_t = Σ_shared FFN(x_t) + Σ_{i∈TopK} g_{t,i} FFN_i(x_t)。"""
    flat = x.reshape(-1, x.shape[-1])
    r = moe.router(flat)
    rows = []
    for t in range(flat.shape[0]):
        y = sum((s(flat[t]) for s in moe.shared), torch.zeros_like(flat[t]))
        for slot in range(r.indices.shape[1]):
            if r.kept[t, slot]:
                y = y + r.gates[t, slot] * moe.experts[int(r.indices[t, slot])](flat[t])
        rows.append(y)
    return torch.stack(rows).view(x.shape)


# region balance
def load_balance_loss(r: Routing, n_experts: int) -> torch.Tensor:
    """Switch/GShard 式辅助损失 E · Σ_i f_i · P_i。

    f_i：分给专家 i 的分配占比（不可导，来自 top-k 的离散选择）；
    P_i：路由器给专家 i 的平均概率（可导）。完全均衡时损失等于 1。
    """
    counts = torch.bincount(r.indices.reshape(-1), minlength=n_experts).float()
    f = counts / counts.sum()
    P = r.probs.mean(0)
    return n_experts * (f * P).sum()


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """ST-MoE 的 z-loss：惩罚 logsumexp 过大，让路由 logits 保持在数值友好的范围。"""
    return torch.logsumexp(logits, dim=-1).pow(2).mean()


@torch.no_grad()
def update_bias(router: Router, indices: torch.Tensor, gamma: float = 1e-3) -> torch.Tensor:
    """DeepSeek-V3 的无辅助损失均衡：过载专家偏置减 γ，欠载专家加 γ。返回本步各专家负载。"""
    load = torch.bincount(indices.reshape(-1), minlength=router.bias.numel()).float()
    router.bias += gamma * torch.sign(load.mean() - load)
    return load
# endregion


def expert_param_counts(cfg: MoEConfig) -> tuple[int, int]:
    """返回 MoE 层的 (总参数, 每 token 激活参数)，不含路由器。"""
    per_expert = 3 * cfg.d_model * cfg.d_expert
    total = (cfg.n_experts + cfg.n_shared) * per_expert
    active = (cfg.top_k + cfg.n_shared) * per_expert
    return total, active
