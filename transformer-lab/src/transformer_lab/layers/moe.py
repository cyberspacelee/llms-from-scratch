"""Top-K 路由、显式 token dispatch/combine 与共享专家。"""

from __future__ import annotations

import torch
from torch import nn

from ..config import BlockConfig
from .feedforward import FeedForward


class MixtureOfExperts(nn.Module):
    """不丢弃 token 的 Top-K MoE：Router -> Dispatch -> Expert -> Combine。

    输入 [B,S,D] 展平为有效 token [N,D]；router 得 [N,E] 分数，
    top-k ids/weights 为 [N,K]。每个 routed expert 只计算分给自己的行，
    再用 index_add 按权重累加回 [N,D]；shared expert 始终处理全部有效行。
    总专家参数随 E 增长，每 token 激活参数随 K 而非 E 增长。
    此 Python dispatch 是原理实现，FLOPs 理论账本见 analysis.ffn_cost，
    实际耗时还包含索引、不同专家 batch 大小和 kernel 启动。
    """

    def __init__(self, dim: int, config: BlockConfig) -> None:
        """根据尺寸和配置创建参数/子层。

        Args:
            dim: 输入/输出 hidden width D；不要求等于 Attention 投影宽度。
            config: 本模块的显式配置对象。

        Returns:
            None；参数与子层注册在 self 中。
        """
        super().__init__()
        self.config = config
        self.router = nn.Linear(dim, config.experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(dim, config.ff_dim, config.activation) for _ in range(config.experts)]
        )
        self.shared = nn.ModuleList(
            [
                FeedForward(dim, config.ff_dim, config.activation)
                for _ in range(config.shared_experts)
            ]
        )
        self.register_buffer("selection_bias", torch.zeros(config.experts))

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """输入 [N,D]，返回 ids[N,K]、归一化 combine 权重[N,K]、分数[N,E]。

        selection_bias 只改变选中的 expert IDs；combine 仍使用原始分数。
        top-k 的选择是离散操作，选中分数和专家输出保留梯度。

        Args:
            x: float [N,D] 有效 token 行。

        Returns:
            tuple: long ids[N,K]、combine weights[N,K]、router scores[N,E]。
        """
        logits = self.router(x)  # [N,D] -> [N,E]
        work = logits if logits.dtype == torch.float64 else logits.float()
        probabilities = (
            work.softmax(-1) if self.config.router_score == "softmax" else work.sigmoid()
        )
        # selection bias[E]广播；Top-K ids/weights[N,K]。
        ids = (probabilities + self.selection_bias).topk(self.config.top_k, -1).indices
        weights = probabilities.gather(-1, ids)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        return ids, weights.to(x.dtype), probabilities

    def forward(
        self, x: torch.Tensor, valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 y[B,S,D]、标量辅助损失、每个专家的分派次数[E]。

        padding 不进入路由和专家，不消耗激活专家计算。counts 总和为 N*K。
        balance='aux' 使用 E*sum(fraction_e*mean_probability_e)；
        balance='bias' 只返回 counts，由训练循环在参数更新后调整选择偏置。
        推理不更新 bias，prefill/decode 与训练使用相同 dispatch 数学。

        Args:
            x: float [B,S,D]。
            valid: 可选 bool [B,S]；padding不路由。

        Returns:
            tuple: output[B,S,D]、scalar auxiliary loss、long counts[E]。
        """
        flat = x.reshape(-1, x.shape[-1])  # [B,S,D] -> [B*S,D]
        rows = (
            torch.arange(flat.shape[0], device=x.device)
            if valid is None
            else valid.reshape(-1).nonzero().flatten()
        )
        tokens = flat[rows]  # [N,D]，N只计有效token
        if tokens.shape[0] == 0:
            return (
                torch.zeros_like(x),
                x.sum() * 0,
                torch.zeros(len(self.experts), device=x.device, dtype=torch.long),
            )
        ids, weights, probabilities = self.route(tokens)
        result = torch.zeros_like(tokens)
        # ponytail: expert loop is a transparent CPU reference; use grouped GEMM when dispatch dominates.
        # token 指原始有效行，slot 指它的第几个路由选择；同一 token 可进入 K 个专家。
        for expert_id, expert in enumerate(self.experts):
            token, slot = (ids == expert_id).nonzero(as_tuple=True)  # 各为 [N_expert]
            if token.numel():
                result = result.index_add(
                    0, token, expert(tokens[token]) * weights[token, slot, None]
                )
        for expert in self.shared:
            result = result + expert(tokens)
        counts = torch.bincount(ids.flatten(), minlength=len(self.experts))
        fractions = counts.to(probabilities.dtype) / ids.numel()
        auxiliary = len(self.experts) * (fractions * probabilities.mean(0)).sum()
        if self.config.balance != "aux":
            auxiliary = auxiliary * 0
        output = torch.zeros_like(flat).index_copy(0, rows, result).reshape_as(x)
        return output, auxiliary, counts

    @torch.no_grad()
    def update_balance(self, counts: torch.Tensor) -> None:
        """Call once after the optimizer step; sum counts across ranks/microbatches first.

        Args:
            counts: long [E] 专家分派次数，需先汇总各 rank/microbatch。

        Returns:
            None；按 counts 原地调整 selection_bias。
        """
        if self.config.balance == "bias" and counts.sum() > 0:
            self.selection_bias.add_(
                self.config.balance_rate * (counts.float().mean() - counts).sign()
            )
            self.selection_bias.sub_(self.selection_bias.mean())


Routing = tuple[tuple[MixtureOfExperts, torch.Tensor], ...]
