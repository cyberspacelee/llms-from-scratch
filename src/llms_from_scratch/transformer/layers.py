"""教学版的归一化、前馈网络与两种残差块（Post-Norm / Pre-Norm）。

``model.RMSNorm`` 与 ``model.SwiGLU`` 是全书共用的实现；这里的版本展开每一步，
测试保证 ``RMSNorm``、``LayerNorm`` 与 PyTorch 参考实现及 ``model`` 中的版本一致。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .attention import MultiHeadAttention


# region norms
class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * γ + β，统计量沿最后一维（每个 token 独立）计算。"""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)  # 有偏方差，与 nn.LayerNorm 一致
        return (x - mean) / torch.sqrt(var + self.eps) * self.weight + self.bias


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x²) + eps) * γ：去掉减均值与偏置，只做尺度归一化。"""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight
# endregion


# region ffn
class GELUFeedForward(nn.Module):
    """原始 Transformer / GPT-2 的 FFN：d -> 4d -> d，中间用 GELU。参数量 8d²（不计偏置）。"""

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


class SwiGLUFeedForward(nn.Module):
    """FFN(x) = W2 (SiLU(W1 x) ⊙ W3 x)。三个矩阵，取 d_ff ≈ 8d/3 使参数量与 4d 的 GELU FFN 持平。"""

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        d_ff = d_ff or swiglu_hidden_dim(d_model)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)  # 门控分支
        self.w3 = nn.Linear(d_model, d_ff, bias=False)  # 数值分支
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        return self.w2(gate * torch.sigmoid(gate) * self.w3(x))  # SiLU(g) = g·σ(g)


def swiglu_hidden_dim(d_model: int, multiple_of: int = 64) -> int:
    """8d/3 向上取整到 multiple_of 的倍数（与 ``GPTConfig`` 的默认 d_ff 相同）。"""
    return multiple_of * math.ceil(8 * d_model / 3 / multiple_of)
# endregion


# region blocks
class TransformerBlock(nn.Module):
    """一个解码器块，三种排布：

    - ``"pre"``  （GPT-2 之后的主流）：x + F(N(x))，残差流本身从不被归一化；
    - ``"post"`` （原始 Transformer）：N(x + F(x))，每个子层之后归一化残差流；
    - ``"plain"``（对照组，无残差）：N(F(x))。
    """

    def __init__(self, d_model: int, n_heads: int, arrangement: str = "pre", norm: str = "rms") -> None:
        super().__init__()
        if arrangement not in ("pre", "post", "plain"):
            raise ValueError(arrangement)
        make_norm = RMSNorm if norm == "rms" else LayerNorm
        self.arrangement = arrangement
        self.norm1, self.norm2 = make_norm(d_model), make_norm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = SwiGLUFeedForward(d_model)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor | None = None) -> torch.Tensor:
        if self.arrangement == "pre":
            x = x + self.attn(self.norm1(x), freqs)
            return x + self.ffn(self.norm2(x))
        if self.arrangement == "post":
            x = self.norm1(x + self.attn(x, freqs))
            return self.norm2(x + self.ffn(x))
        x = self.norm1(self.attn(x, freqs))
        return self.norm2(self.ffn(x))
# endregion


# region grad_norms
def residual_gradient_norms(
    n_layers: int, arrangement: str, std: float | None = None, scale_residual: bool = False,
    d_model: int = 64, n_heads: int = 4, seq_len: int = 32, batch: int = 8, seed: int = 0,
) -> list[float]:
    """初始化时堆叠 n_layers 个块，返回损失对每层输入 h_0 … h_L 的梯度范数 ‖∂ℓ/∂h_l‖。

    std: 权重矩阵的初始化标准差（默认 1/sqrt(d_model)）；scale_residual: 是否像 GPT-2 那样
    把写回残差流的投影（o_proj、w2）再乘以 1/sqrt(2L)。Pre-Norm 栈末尾加一个 final norm。
    """
    torch.manual_seed(seed)
    std = std or d_model**-0.5
    blocks = [TransformerBlock(d_model, n_heads, arrangement) for _ in range(n_layers)]
    for block in blocks:
        for p in block.parameters():
            if p.dim() == 2:
                nn.init.normal_(p, std=std)
        if scale_residual:
            for w in (block.attn.o_proj.weight, block.ffn.w2.weight):
                nn.init.normal_(w, std=std / math.sqrt(2 * n_layers))
    h = torch.randn(batch, seq_len, d_model, requires_grad=True)
    hidden = [h]
    for block in blocks:
        h = block(h)
        hidden.append(h)
    out = RMSNorm(d_model)(h) if arrangement == "pre" else h
    loss = F.mse_loss(out, torch.randn_like(out))
    return [float(g.norm()) for g in torch.autograd.grad(loss, hidden)]
# endregion
