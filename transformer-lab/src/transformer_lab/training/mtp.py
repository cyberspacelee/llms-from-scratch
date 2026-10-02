"""顺序 MTP：用真实未来 token 的 embedding 训练额外预测深度。"""

from __future__ import annotations

import torch
from torch import nn

from ..config import BlockConfig
from ..layers import RMSNorm, Routing
from ..models.blocks import TransformerBlock


class MultiTokenPrediction(nn.Module):
    """深度 j 消费真实未来 token x(t+j)，预测 x(t+j+1)，j 从 1 开始。

    h[B,S-j+1,D] 截短一位，与未来 embedding[B,S-j,D] 各自 RMSNorm，
    concat -> [B,S-j,2D] -> Linear -> [B,S-j,D] -> causal Block。
    最后一位无监督标签，再截掉，得到 [B,S-j-1,V] 预测。
    每个深度有独立 Block，embedding/head 从父模型传入并共享参数；
    增量参数与额外训练 FLOPs 是每深度投影、Norm 和 Block 的成本之和。
    此模块用于训练，普通 generate 不调用它；不能直接把训练时的真值
    future embedding 当成推理可用信息。MTP 草稿解码需要另外设计接受/回退。
    """

    def __init__(self, dim: int, config: BlockConfig, depth: int) -> None:
        """根据尺寸和配置创建参数/子层。

        Args:
            dim: 输入/输出 hidden width D；不要求等于 Attention 投影宽度。
            config: 本模块的显式配置对象。
            depth: MTP 的额外预测深度，非负整数。

        Returns:
            None；参数与子层注册在 self 中。
        """
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
        """执行本模块的前向计算；输入与输出 shape 约定如下。

        Args:
            hidden: float [B,S,D] 主模型 hidden states。
            tokens: long [B,S] 未右移的真实 token 序列。
            embedding: 共享 nn.Embedding，权重 [V,D]。
            head: 共享词表投影 nn.Linear，权重 [V,D]。
            valid: 可选 bool [B,S] 有效 token mask，True=参与监督。

        Returns:
            tuple: predictions_j[B,S-j-1,V] 的 tuple、scalar auxiliary loss、Routing；j 从 1 开始。
        """
        if hidden.shape[:2] != tokens.shape or tokens.ndim != 2:
            raise ValueError("MTP needs aligned hidden states and tokens")
        valid = torch.ones_like(tokens, dtype=torch.bool) if valid is None else valid
        h, outputs, auxiliary, routing = hidden, [], hidden.new_zeros(()), ()
        active = valid
        for i, block in enumerate(self.blocks):
            if h.shape[1] < 3:
                break
            h = h[:, :-1]  # 深度j=i+1：[B,S-j+1,D] -> [B,S-j,D]
            # 第 i 层可见 x(t+i+1)，但因果 Block 不能读取它的目标 x(t+i+2)。
            future = embedding(tokens[:, i + 1 :])  # tokens[B,S-j] -> [B,S-j,D]
            active = active[:, :-1] & valid[:, i + 1 :]
            # concat[B,S-j,2D] -> projection[B,S-j,D]，不拼接sequence轴。
            h = self.projections[i](
                torch.cat((self.hidden_norms[i](h), self.token_norms[i](future)), -1)
            )
            h, _, aux, stats = block(h, True, active)
            outputs.append(head(self.output_norms[i](h))[:, :-1])  # [B,S-j-1,V]，对齐tokens[:,j+1:]
            auxiliary, routing = auxiliary + aux, routing + stats
        return tuple(outputs), auxiliary, routing
