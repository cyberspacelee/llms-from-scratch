"""可组合的 Transformer Block；Attention 只在多实现处进行选择。"""

from __future__ import annotations

import torch
from torch import nn

from ..attention import MultiHeadAttention, MultiHeadLatentAttention, RecurrentAttention
from ..cache import LayerCache
from ..config import AttentionConfig, BlockConfig
from ..layers import FeedForward, MixtureOfExperts, Routing, make_norm


def make_attention(
    dim: int, config: AttentionConfig, dropout: float = 0.0, cross: bool = False
) -> MultiHeadAttention | MultiHeadLatentAttention | RecurrentAttention:
    if config.kind in {"mha", "mqa", "gqa"}:
        return MultiHeadAttention(dim, config, dropout, cross)
    if config.kind == "mla":
        return MultiHeadLatentAttention(dim, config, dropout, cross)
    if cross:
        raise ValueError("cross attention must use softmax")
    return RecurrentAttention(dim, config)


class TransformerBlock(nn.Module):
    """[B,T,D] 经 self-attention、可选 cross-attention、FFN/MoE 保持原 Shape。

    Encoder/Decoder-only 有两条分支，seq2seq Decoder 有三条。
    Pre-Norm: x+F(Norm(x))；Post-Norm: Norm(x+F(x))。
    每条分支有独立 Norm；门控残差用可学习标量 sigmoid(g) 缩放分支。
    参数/FLOPs 是 Attention+FFN+Norm 各项之和；只有 Attention 持有缓存。
    """

    def __init__(
        self,
        dim: int,
        config: BlockConfig,
        cross_attention: AttentionConfig | None = None,
        cross_causal: bool = False,
    ) -> None:
        super().__init__()
        self.config, self.cross_causal = config, cross_causal
        self.attention = make_attention(dim, config.attention, config.dropout)
        self.cross = (
            make_attention(dim, cross_attention, config.dropout, True) if cross_attention else None
        )
        self.ff = (
            MixtureOfExperts(dim, config)
            if config.experts
            else FeedForward(dim, config.ff_dim, config.activation)
        )
        self.norms = nn.ModuleList(
            [make_norm(config.norm, dim) for _ in range(3 if self.cross else 2)]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.residual_gates = (
            nn.Parameter(torch.zeros(len(self.norms))) if config.residual == "gated" else None
        )

    def branch_input(self, x: torch.Tensor, index: int) -> torch.Tensor:
        return self.norms[index](x) if self.config.norm_order == "pre" else x

    def combine(self, x: torch.Tensor, branch: torch.Tensor, index: int) -> torch.Tensor:
        branch = self.dropout(branch)
        if self.residual_gates is not None:
            branch = branch * self.residual_gates[index].sigmoid()
        x = x + branch
        return x if self.config.norm_order == "pre" else self.norms[index](x)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool,
        valid: torch.Tensor,
        cache: LayerCache | None = None,
        use_cache: bool = False,
        offset: int = 0,
        memory: torch.Tensor | None = None,
        memory_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LayerCache | None, torch.Tensor, Routing]:
        """返回 hidden、可选本层 cache、MoE 辅助损失与路由统计。

        x 只含当前 T 个 token；valid 是含历史的 [B,offset+T]，
        Attention 用全部 key validity，MoE 只取当前 chunk 的有效行。
        缓存跨层独立，Cross 状态静态，Self 状态追加或递归更新。
        """
        branch, self_cache = self.attention(
            self.branch_input(x, 0),
            cache=None if cache is None else cache.self_attention,
            use_cache=use_cache,
            query_offset=offset,
            causal=causal,
            key_valid=valid,
        )
        x = self.combine(x, branch, 0)
        cross_cache = None
        if self.cross is not None:
            branch, cross_cache = self.cross(
                self.branch_input(x, 1),
                memory=memory,
                cache=None if cache is None else cache.cross_attention,
                use_cache=use_cache,
                query_offset=offset,
                causal=self.cross_causal,
                key_valid=memory_valid,
            )
            x = self.combine(x, branch, 1)
        index = len(self.norms) - 1
        if isinstance(self.ff, MixtureOfExperts):
            branch, auxiliary, counts = self.ff(
                self.branch_input(x, index), valid[:, offset : offset + x.shape[1]]
            )
            routing = ((self.ff, counts),)
        else:
            branch = self.ff(self.branch_input(x, index))
            auxiliary, routing = x.new_zeros(()), ()
        return (
            self.combine(x, branch, index),
            LayerCache(self_cache, cross_cache) if use_cache else None,
            auxiliary,
            routing,
        )
