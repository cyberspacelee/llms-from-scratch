"""08 · 2021：Rotary Position Embedding，只旋转 Q/K，去掉加性位置编码。"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..attention.position import apply_rope
from ..config import ModelConfig, PositionConfig
from ..experiments.runners import consistency
from .ch06_gated_ffn import config as ffn_config


def config() -> ModelConfig:
    """无输入；返回 RoPE MHA 配方，不将 MQA 支线强制叠加到主线。

    Args:
        无显式输入。

    Returns:
        ModelConfig 本章的独立配方。
    """
    c = ffn_config()
    return replace(
        c,
        block=replace(
            c.block, attention=replace(c.block.attention, position=PositionConfig(kind="rope"))
        ),
    )


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 RoPE shape、范数/共同平移不变量及 self/cross 对照。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    q, k = (torch.randn(2, 4, 5, 8, device=device, dtype=torch.float64) for _ in range(2))
    positions = torch.arange(5, device=device)  # [T]
    qr, kr = apply_rope(q, positions), apply_rope(k, positions)  # [B,H_q,T,D_h] 不变
    torch.testing.assert_close(q.square().sum(-1), qr.square().sum(-1))
    torch.testing.assert_close(
        qr @ kr.transpose(-1, -2),
        apply_rope(q, positions + 10) @ apply_rope(k, positions + 10).transpose(-1, -2),
    )
    c = config()
    return {
        "input_shape": list(q.shape),
        "positions_shape": list(positions.shape),
        "paired_coordinates_shape": [2, 4, 5, 4, 2],
        "output_shape": list(qr.shape),
        "decoder": consistency(c, device),
        "cross_attention": consistency(replace(c, architecture="encoder_decoder"), device),
    }
