"""教学版注意力：手写 softmax、缩放点积注意力、单头与多头因果自注意力。

这些实现刻意展开每一步，便于打印形状与核对数值；``model.CausalSelfAttention``
是同一计算的紧凑版本（外加 GQA 与 KV Cache），测试保证两者数值一致。
张量布局：输入 ``x`` 为 ``[B, T, d_model]``，多头内部为 ``[B, h, T, d_h]``。
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .model import apply_rope


# region softmax
def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """数值稳定的 softmax：先减去最大值，exp 的参数都 <= 0，不会上溢。

    整行都是 -inf（被完全掩码）时返回全 0，而不是 NaN。
    """
    x_max = x.amax(dim=dim, keepdim=True)
    x_max = torch.where(torch.isfinite(x_max), x_max, torch.zeros_like(x_max))
    e = torch.exp(x - x_max)
    return e / e.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(e.dtype).tiny)
# endregion


# region sdpa
def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d_k) + mask) V。

    q: [..., T, d_k]，k: [..., S, d_k]，v: [..., S, d_v]；
    mask: 可广播到 [..., T, S] 的布尔张量，True 表示“允许关注”。
    返回 (输出 [..., T, d_v], 注意力权重 [..., T, S])。
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])  # [..., T, S]
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = softmax(scores, dim=-1)
    return weights @ v, weights
# endregion


def causal_mask(t: int, s: int | None = None, device: torch.device | str | None = None) -> torch.Tensor:
    """下三角布尔掩码 [t, s]。s > t 时（已有缓存的前缀）把对角线右移 s - t。"""
    s = t if s is None else s
    return torch.ones(t, s, dtype=torch.bool, device=device).tril(diagonal=s - t)


def padding_mask(lengths: torch.Tensor, s: int) -> torch.Tensor:
    """右侧填充的键掩码 [B, 1, 1, s]：位置 j < lengths[b] 为 True。"""
    keep = torch.arange(s, device=lengths.device)[None, :] < lengths[:, None]
    return keep[:, None, None, :]


# region single_head
class SingleHeadAttention(nn.Module):
    """单头因果自注意力：每个位置对自己及之前的位置做加权平均。"""

    def __init__(self, d_model: int, d_head: int) -> None:
        super().__init__()
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.W_q(x), self.W_k(x), self.W_v(x)  # 各 [B, T, d_head]
        mask = causal_mask(x.shape[1], device=x.device)  # [T, T]，广播到每个样本
        return scaled_dot_product_attention(q, k, v, mask)
# endregion


# region heads
def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """[B, T, h*d_h] -> [B, h, T, d_h]：view 拆出头维，transpose 把头维挪到批维旁边。"""
    B, T, D = x.shape
    return x.view(B, T, n_heads, D // n_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """[B, h, T, d_h] -> [B, T, h*d_h]。transpose 后内存不连续，需要 reshape（或 contiguous().view）。"""
    B, h, T, d_h = x.shape
    return x.transpose(1, 2).reshape(B, T, h * d_h)
# endregion


# region mha
class MultiHeadAttention(nn.Module):
    """多头因果自注意力。参数名与 ``model.CausalSelfAttention`` 相同，可直接互载权重。"""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.n_heads = n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.last_weights: torch.Tensor | None = None  # [B, h, T, T]，供可视化

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor | None = None,
        key_padding: torch.Tensor | None = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """x: [B, T, d_model]；freqs: RoPE 旋转因子 [T, d_h/2]（None 表示不加位置）；
        key_padding: 每个样本的有效长度 [B]（右侧填充）。"""
        B, T, _ = x.shape
        q = split_heads(self.q_proj(x), self.n_heads)  # [B, h, T, d_h]
        k = split_heads(self.k_proj(x), self.n_heads)
        v = split_heads(self.v_proj(x), self.n_heads)
        if freqs is not None:
            q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        mask = causal_mask(T, device=x.device) if causal else None
        if key_padding is not None:
            pad = padding_mask(key_padding, T)  # [B, 1, 1, T]
            mask = pad if mask is None else mask & pad
        out, self.last_weights = scaled_dot_product_attention(q, k, v, mask)
        return self.o_proj(merge_heads(out))
# endregion
