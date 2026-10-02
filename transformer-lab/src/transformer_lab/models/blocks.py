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
    """按 Attention 类型构造对应数学实现。

    Args:
        dim: 输入/输出 hidden width D；不要求等于 Attention 投影宽度。
        config: 本模块的显式配置对象。
        dropout: 训练期 dropout 概率，推理期关闭。
        cross: 是否使用 Encoder memory 作为 K/V 来源。

    Returns:
        MultiHeadAttention / MultiHeadLatentAttention / RecurrentAttention 实例。
    """
    if config.kind in {"mha", "mqa", "gqa"}:
        return MultiHeadAttention(dim, config, dropout, cross)
    if config.kind == "mla":
        return MultiHeadLatentAttention(dim, config, dropout, cross)
    if cross:
        raise ValueError("cross attention must use softmax")
    return RecurrentAttention(dim, config)


class TransformerBlock(nn.Module):
    """[B,S,D] 经 self-attention、可选 cross-attention、FFN/MoE 保持原 Shape。

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
        """根据尺寸和配置创建参数/子层。

        Args:
            dim: 输入/输出 hidden width D；不要求等于 Attention 投影宽度。
            config: 本模块的显式配置对象。
            cross_attention: 可选 Cross-Attention 配置。
            cross_causal: 是否使用对齐位置的因果 Cross，可见 k≤q。

        Returns:
            None；参数与子层注册在 self 中。
        """
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
        """为残差分支准备 Pre-Norm 输入。

        Args:
            x: float [B,S,D]。
            index: 残差分支/Norm 的零起点编号。

        Returns:
            float [B,S,D]，Pre-Norm时归一化，否则原输入。
        """
        return self.norms[index](x) if self.config.norm_order == "pre" else x

    def combine(self, x: torch.Tensor, branch: torch.Tensor, index: int) -> torch.Tensor:
        """对分支做 dropout/门控并与残差合并。

        Args:
            x: float [B,S,D] 残差输入。
            branch: float [B,S,D] 分支输出。
            index: 残差分支/Norm 的零起点编号。

        Returns:
            float [B,S,D]，残差相加并按需 Post-Norm。
        """
        branch = self.dropout(branch)
        if self.residual_gates is not None:
            branch = branch * self.residual_gates[index].sigmoid()
        x = x + branch  # 两者[B,S_q,D]，残差相加不改变sequence/hidden维
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

        x 只含当前 S_q 个 token；valid 是含历史的 [B,P+S_q]，
        Attention 用全部 key validity，MoE 只取当前 chunk 的有效行。
        缓存跨层独立，Cross 状态静态，Self 状态追加或递归更新。

        Args:
            x: float [B,S_q,D] 当前 chunk。
            causal: 是否限制 key 绝对位置不超过 query。
            valid: bool [B,P+S_q] 完整 Self prefix。
            cache: 已有单层/模型状态；None 表示没有历史。
            use_cache: 是否返回新状态；不原地修改既有缓存。
            offset: 当前 chunk 的绝对位置起点 P。
            memory: 可选 float [B,S_kv,D] Encoder hidden。
            memory_valid: 可选 bool [B,S_kv] Encoder 有效位置。

        Returns:
            tuple: hidden[B,S_q,D]、LayerCache 或 None、scalar aux loss、Routing。
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
