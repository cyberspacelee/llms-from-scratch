"""逐 token FFN：普通激活与乘法门控共用上投影/下投影。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """[B,T,D] -> [B,T,D_ff] -> [B,T,D]，不同 token 之间不交换信息。

    普通：activation(x@W_upᵀ)@W_downᵀ，参数 2DD_ff、matmul FLOPs 4BTDD_ff。
    门控：activation(x@W_gateᵀ)*(x@W_upᵀ)，再下投影；参数 3DD_ff，
    matmul FLOPs 6BTDD_ff。GLU/GeGLU/SwiGLU 对 gate 分别用 sigmoid/GELU/SiLU。
    这里无 bias、无 KV 状态；三种执行阶段只是 B/T 不同。
    """

    def __init__(
        self, dim: int, hidden_dim: int, activation: str = "swiglu", bias: bool = False
    ) -> None:
        """根据尺寸和配置创建参数/子层。

        Args:
            dim: 输入/输出 hidden width D；不要求等于 Attention 投影宽度。
            hidden_dim: FFN intermediate size D_ff。
            activation: FFN 激活名：relu/gelu/glu/geglu/swiglu。

        Returns:
            None；参数与子层注册在 self 中。
        """
        super().__init__()
        if activation not in {"relu", "gelu", "glu", "geglu", "swiglu"}:
            raise ValueError("unknown activation")
        self.activation = activation
        self.up = nn.Linear(dim, hidden_dim, bias=bias)
        self.gate = nn.Linear(dim, hidden_dim, bias=bias) if activation.endswith("glu") else None
        self.down = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行本模块的前向计算；输入与输出 shape 约定如下。

        Args:
            x: float [B,T,D] 或 [...,D]。

        Returns:
            float [...,D]，中间 up/gate[...,D_ff]。
        """
        function = {
            "relu": F.relu,
            "gelu": F.gelu,
            "glu": torch.sigmoid,
            "geglu": F.gelu,
            "swiglu": F.silu,
        }[self.activation]
        # up/gate[...,D] -> [...,D_ff]；门控是相同shape的逐元素乘法。
        hidden = function(self.up(x)) if self.gate is None else function(self.gate(x)) * self.up(x)
        return self.down(hidden)  # [...,D_ff] -> [...,D]
