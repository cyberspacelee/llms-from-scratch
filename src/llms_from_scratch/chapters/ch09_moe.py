"""09 · 2021–2024：moe，机制与数值核对独立成章。"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..analysis import ffn_cost, parameter_count
from ..config import ModelConfig
from ..experiments.runners import consistency
from ..layers import MixtureOfExperts
from .ch08_rope import config as rope_config


def config() -> ModelConfig:
    """无输入；返回本章机制的显式配置，其他支线不自动启用。

    Args:
        无显式输入。

    Returns:
        ModelConfig 本章的独立配方。
    """
    c = rope_config()
    return replace(
        c,
        block=replace(
            c.block, experts=4, top_k=2, shared_experts=1, router_score="sigmoid", balance="bias"
        ),
    )


def run(device: torch.device) -> dict[str, object]:
    """路由只激活 K 个 routed expert；shared expert 对每个有效 token 激活。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    c = config()
    moe = MixtureOfExperts(c.dim, c.block).to(device)
    x = torch.randn(2, 3, c.dim, device=device, requires_grad=True)
    y, auxiliary, counts = moe(x)
    assert counts.sum().item() == x.shape[0] * x.shape[1] * c.block.top_k
    (y.square().mean() + auxiliary).backward()
    assert moe.router.weight.grad is not None and torch.isfinite(moe.router.weight.grad).all()
    # 训练循环应在 optimizer.step 之后汇总 counts，再更新 selection bias。
    moe.update_balance(counts)
    return {
        "dispatch_counts": counts.tolist(),
        "cost": ffn_cost(c.dim, c.block),
        "total_parameters": parameter_count(moe),
        "model_check": consistency(c, device),
    }
