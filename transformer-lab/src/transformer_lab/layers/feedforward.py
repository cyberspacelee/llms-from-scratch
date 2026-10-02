"""逐 token FFN：普通激活与乘法门控共用上投影/下投影。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    """[B,T,D] -> [B,T,F] -> [B,T,D]，不同 token 之间不交换信息。

    普通：activation(x@W_upᵀ)@W_downᵀ，参数 2DF、matmul FLOPs 4BTDF。
    门控：activation(x@W_gateᵀ)*(x@W_upᵀ)，再下投影；参数 3DF，
    matmul FLOPs 6BTDF。GLU/GeGLU/SwiGLU 对 gate 分别用 sigmoid/GELU/SiLU。
    这里无 bias、无 KV 状态；三种执行阶段只是 B/T 不同。
    """

    def __init__(self, dim: int, hidden_dim: int, activation: str = "swiglu") -> None:
        super().__init__()
        if activation not in {"relu", "gelu", "glu", "geglu", "swiglu"}:
            raise ValueError("unknown activation")
        self.activation = activation
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.gate = nn.Linear(dim, hidden_dim, bias=False) if activation.endswith("glu") else None
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        function = {
            "relu": F.relu,
            "gelu": F.gelu,
            "glu": torch.sigmoid,
            "geglu": F.gelu,
            "swiglu": F.silu,
        }[self.activation]
        hidden = function(self.up(x)) if self.gate is None else function(self.gate(x)) * self.up(x)
        return self.down(hidden)
