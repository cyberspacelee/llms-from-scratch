"""顺序 MTP：用真实未来 token 的 embedding 训练额外预测深度。"""

from __future__ import annotations

import torch
from torch import nn

from ..config import BlockConfig
from ..layers import RMSNorm, Routing
from ..models.blocks import TransformerBlock


class MultiTokenPrediction(nn.Module):
    """深度 j 消费真实未来 token x(t+j)，预测 x(t+j+1)，j 从 1 开始。

    h[B,T-j+1,D] 截短一位，与未来 embedding[B,T-j,D] 各自 RMSNorm，
    concat -> [B,T-j,2D] -> Linear -> [B,T-j,D] -> causal Block。
    最后一位无监督标签，再截掉，得到 [B,T-j-1,V] 预测。
    每个深度有独立 Block，embedding/head 从父模型传入并共享参数；
    增量参数与额外训练 FLOPs 是每深度投影、Norm 和 Block 的成本之和。
    此模块用于训练，普通 generate 不调用它；不能直接把训练时的真值
    future embedding 当成推理可用信息。MTP 草稿解码需要另外设计接受/回退。
    """

    def __init__(self, dim: int, config: BlockConfig, depth: int) -> None:
        super().__init__()

        self.hidden_norms = nn.ModuleList([RMSNorm(dim) for _ in range(depth)])
        self.token_norms = nn.ModuleList([RMSNorm(dim) for _ in range(depth)])
        self.projections = nn.ModuleList(
            [nn.Linear(2 * dim, dim, bias=False) for _ in range(depth)]
        )
        self.blocks = nn.ModuleList([TransformerBlock(dim, config) for _ in range(depth)])
        self.output_norms = nn.ModuleList([RMSNorm(dim) for _ in range(depth)])

    def forward(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        embedding: nn.Embedding,
        head: nn.Linear,
        valid: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, Routing]:
        if hidden.shape[:2] != tokens.shape or tokens.ndim != 2:
            raise ValueError("MTP needs aligned hidden states and tokens")
        valid = torch.ones_like(tokens, dtype=torch.bool) if valid is None else valid
        h, outputs, auxiliary, routing = hidden, [], hidden.new_zeros(()), ()
        active = valid
        for i, block in enumerate(self.blocks):
            if h.shape[1] < 3:
                break
            h = h[:, :-1]
            # 第 i 层可见 x(t+i+1)，但因果 Block 不能读取它的目标 x(t+i+2)。
            future = embedding(tokens[:, i + 1 :])
            active = active[:, :-1] & valid[:, i + 1 :]
            h = self.projections[i](
                torch.cat((self.hidden_norms[i](h), self.token_norms[i](future)), -1)
            )
            h, _, aux, stats = block(h, True, active)
            outputs.append(head(self.output_norms[i](h))[:, :-1])
            auxiliary, routing = auxiliary + aux, routing + stats
        return tuple(outputs), auxiliary, routing
