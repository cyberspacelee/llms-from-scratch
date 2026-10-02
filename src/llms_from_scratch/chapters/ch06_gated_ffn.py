"""06 · 2020：ReLU/GELU → GLU/GeGLU/SwiGLU，逐 token 门控 FFN。"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..analysis import ffn_cost, parameter_count
from ..config import ModelConfig
from ..experiments.runners import consistency
from ..layers import FeedForward
from .ch05_normalization import config as normalization_config


def config() -> ModelConfig:
    """无输入；返回 Pre-RMSNorm + SwiGLU 配方，位置机制仍为 Sinusoidal。

    Args:
        无显式输入。

    Returns:
        ModelConfig 本章的独立配方。
    """
    c = normalization_config()
    return replace(c, block=replace(c.block, activation="swiglu"))


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回每种 FFN 的 shape、参数数目和梯度核对。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    c = config()
    report = {}
    for activation in ("relu", "gelu", "glu", "geglu", "swiglu"):
        ff = FeedForward(c.dim, c.block.ff_dim, activation).to(device)
        x = torch.randn(2, 5, c.dim, device=device, requires_grad=True)  # [B,T,D]
        y = ff(x)  # up/gate [B,T,D_ff] -> product [B,T,D_ff] -> down [B,T,D]
        y.square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        cost = ffn_cost(c.dim, replace(c.block, activation=activation))
        assert parameter_count(ff) == cost["total_parameters"]
        report[activation] = {
            "input_shape": list(x.shape),
            "intermediate_shape": [2, 5, c.block.ff_dim],
            "output_shape": list(y.shape),
            "cost": cost,
        }
    return {"ffns": report, "model_check": consistency(c, device)}
