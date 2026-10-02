"""14 · 2024：mtp，机制与数值核对独立成章。"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..config import ModelConfig
from ..models import Transformer
from ..training import language_model_loss
from .ch13_mla import config as mla_config


def config() -> ModelConfig:
    """无输入；返回本章机制的显式配置，其他支线不自动启用。

    Args:
        无显式输入。

    Returns:
        ModelConfig 本章的独立配方。
    """
    return replace(mla_config(), mtp_depth=2)


def run(device: torch.device) -> dict[str, object]:
    """NTP 标签为 x(t+1)；第一个额外 MTP 深度标签为 x(t+2)。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    c = config()
    model = Transformer(c).to(device)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
    output = model(ids)
    total, terms, routing = language_model_loss(model, output, ids)
    total.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.grad is not None for p in model.mtp.parameters())
    predictions, _, _ = model.mtp(output.hidden, ids, model.embedding, model.head)
    assert [p.shape[1] for p in predictions] == [4, 3]
    return {
        "terms": {name: term.item() for name, term in terms.items()},
        "total": total.item(),
        "ntp_labels": ids[:, 1:].tolist(),
        "mtp_labels": [ids[:, 2:].tolist(), ids[:, 3:].tolist()],
        "mtp_shapes": [list(p.shape) for p in predictions],
        "routing_records": len(routing),
    }
