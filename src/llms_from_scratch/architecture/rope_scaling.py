"""RoPE 的上下文扩展：位置插值（PI）、NTK-aware 缩放与 YaRN。

三者都只改 RoPE 的逆频率 θ_k = base^(-2k/d)（k = 0 … d/2-1），模型其余部分不变。
返回的逆频率可交给 ``rotary_factors`` 生成与 ``transformer.model.rope_frequencies``
同格式的复数旋转因子。
"""

from __future__ import annotations

import math

import torch


def inv_freq(head_dim: int, base: float = 10000.0) -> torch.Tensor:
    """θ_k = base^(-2k/d)，形状 [head_dim // 2]，从高频（k=0）到低频。"""
    return base ** (-torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)


def wavelength(theta: torch.Tensor) -> torch.Tensor:
    """第 k 对维度转一整圈需要的 token 数 λ_k = 2π / θ_k。"""
    return 2 * math.pi / theta


def rotary_factors(theta: torch.Tensor, length: int) -> torch.Tensor:
    """e^{i·m·θ_k}，形状 [length, head_dim // 2]，可直接交给 ``apply_rope``。"""
    angles = torch.outer(torch.arange(length, dtype=torch.float64), theta)
    return torch.polar(torch.ones_like(angles), angles).to(torch.complex64)


# region pi_ntk
def pi_inv_freq(head_dim: int, scale: float, base: float = 10000.0) -> torch.Tensor:
    """位置插值：把位置 m 换成 m / s，等价于所有频率除以 s。"""
    return inv_freq(head_dim, base) / scale


def ntk_inv_freq(head_dim: int, scale: float, base: float = 10000.0) -> torch.Tensor:
    """NTK-aware：改底数 base' = base · s^{d/(d-2)}，最高频不变、最低频恰好除以 s。"""
    return inv_freq(head_dim, base * scale ** (head_dim / (head_dim - 2)))
# endregion


# region yarn
def yarn_inv_freq(head_dim: int, scale: float, original_len: int, base: float = 10000.0,
                  alpha: float = 1.0, beta: float = 32.0) -> torch.Tensor:
    """YaRN 的 “NTK-by-parts” 频率：按每对维度在原上下文里转的圈数 r 分段处理。

    r < α（波长超过原上下文，低频）：完全插值 θ/s；
    r > β（在原上下文里转了很多圈，高频）：保持 θ 不变；
    中间用线性斜坡 γ(r) = (r - α) / (β - α) 混合。
    """
    theta = inv_freq(head_dim, base)
    r = original_len / wavelength(theta)
    gamma = ((r - alpha) / (beta - alpha)).clamp(0.0, 1.0)
    return (1 - gamma) * theta / scale + gamma * theta


def yarn_mscale(scale: float) -> float:
    """注意力温度修正 sqrt(1/t) = 0.1·ln(s) + 1；q 与 k 各乘一次，logits 乘它的平方。"""
    return 1.0 if scale <= 1 else 0.1 * math.log(scale) + 1.0
# endregion
