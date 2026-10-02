"""05 · 2019–2020：Post/Pre-LayerNorm 与 RMSNorm，先只改变归一化。"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.nn import functional as F

from ..config import ModelConfig
from ..experiments.runners import consistency
from ..layers import LayerNorm, RMSNorm
from .ch02_decoder_only import config as decoder_config


def config() -> ModelConfig:
    """无输入；返回 Pre-RMSNorm Decoder；不提前启用 gated FFN/RoPE/GQA。

    Args:
        无显式输入。

    Returns:
        ModelConfig 本章的独立配方。
    """
    c = decoder_config()
    return replace(c, block=replace(c.block, norm="rms", norm_order="pre"))


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 [...,D] Norm 数值对照与四种 Block 配方检查。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    x = torch.randn(2, 5, 32, device=device, dtype=torch.float64)  # [B,T,D]
    ln, rms = LayerNorm(32).to(device).double(), RMSNorm(32).to(device).double()
    torch.testing.assert_close(ln(x), F.layer_norm(x, (32,), ln.weight, ln.bias, ln.eps))
    torch.testing.assert_close(rms(x), F.rms_norm(x, (32,), rms.weight, rms.eps))
    c = decoder_config()
    return {
        "input_shape": list(x.shape),
        "output_shape": list(rms(x).shape),
        "statistics_shape": [2, 5, 1],
        "variants": {
            f"{norm}_{order}": consistency(
                replace(c, block=replace(c.block, norm=norm, norm_order=order)), device
            )
            for norm in ("layer", "rms")
            for order in ("post", "pre")
        },
    }
