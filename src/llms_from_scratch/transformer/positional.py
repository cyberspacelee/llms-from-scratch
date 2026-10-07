"""位置编码：正弦绝对位置编码，以及用实数旋转写出的 RoPE。

``model.rope_frequencies`` / ``model.apply_rope`` 用复数乘法实现 RoPE；
这里的 ``rope_rotate`` 用 cos/sin 显式旋转每一对坐标，测试保证两者一致。
维度配对采用“相邻两维为一对”（交错布局，与原论文及 LLaMA 原始实现相同）；
``rope_rotate_half`` 给出 GPT-NeoX / Hugging Face 常用的“前后两半为一对”布局。
"""

from __future__ import annotations

import torch


# region sinusoidal
def sinusoidal_encoding(length: int, d_model: int, base: float = 10000.0) -> torch.Tensor:
    """PE[p, 2i] = sin(p·ω_i)，PE[p, 2i+1] = cos(p·ω_i)，ω_i = base^(-2i/d)。形状 [length, d_model]。"""
    if d_model % 2:
        raise ValueError("d_model 必须是偶数")
    omega = base ** (-torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)  # [d/2]
    angles = torch.arange(length, dtype=torch.float32)[:, None] * omega[None, :]  # [L, d/2]
    pe = torch.empty(length, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe
# endregion


# region rope
def rope_angles(head_dim: int, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """第 m 个位置、第 i 对坐标的旋转角 m·θ_i，θ_i = theta^(-2i/d_h)。形状 [..., head_dim/2]。"""
    inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    return positions.float()[..., None] * inv_freq


def rope_rotate(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """把 x[..., T, d_h] 的每一对相邻坐标 (x_{2i}, x_{2i+1}) 按角度 m·θ_i 逆时针旋转。

    positions: [T]，第 t 行的绝对位置。
    """
    angle = rope_angles(x.shape[-1], positions, theta)  # [T, d_h/2]
    cos, sin = torch.cos(angle), torch.sin(angle)
    x1, x2 = x[..., 0::2].float(), x[..., 1::2].float()  # 每对的第一、第二个分量
    out = torch.empty(x.shape, dtype=torch.float32, device=x.device)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.to(x.dtype)
# endregion


def rope_rotate_half(x: torch.Tensor, positions: torch.Tensor, theta: float = 10000.0) -> torch.Tensor:
    """“前后两半”布局：第 i 对是 (x_i, x_{i + d_h/2})。与交错布局只差一个固定的维度置换。"""
    half = x.shape[-1] // 2
    angle = rope_angles(x.shape[-1], positions, theta)
    cos, sin = torch.cos(angle), torch.sin(angle)
    x1, x2 = x[..., :half].float(), x[..., half:].float()
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)
