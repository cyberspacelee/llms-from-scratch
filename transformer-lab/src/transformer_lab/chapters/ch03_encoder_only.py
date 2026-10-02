"""03 · 2018：BERT 路线的双向 Encoder 与 Masked Language Modeling。

只演示双向可见性和 masked-token 监督，不复刻 BERT tokenizer/embedding/NSP。
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..config import ModelConfig
from ..models import Transformer
from ..training import token_loss
from .ch01_naive_transformer import classic_config


def config() -> ModelConfig:
    """无输入；返回双向 Encoder-only 配方，沿用经典位置编码和 FFN。

    Args:
        无显式输入。

    Returns:
        ModelConfig 本章的独立配方。
    """
    return replace(classic_config(), architecture="encoder")


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 mask位置、logits shape、MLM loss 和双向性检查。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    model = Transformer(config()).to(device).double()
    ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)  # [B=1,S=5]
    selected = torch.tensor([[False, True, False, True, False]], device=device)  # bool [B,S]
    masked = ids.masked_fill(selected, 31)  # [B,S]，31 作为本例的 MASK ID
    output = model(masked)  # hidden [B,S,D], logits [B,S,V]
    loss = token_loss(output.logits, ids, selected)  # [N_masked,V] vs [N_masked] -> scalar
    loss.backward()
    assert (
        model.embedding.weight.grad is not None
        and torch.isfinite(model.embedding.weight.grad).all()
    )
    changed = masked.clone()
    changed[:, -1] = 6
    assert not torch.allclose(output.logits[:, :1], model(changed).logits[:, :1])
    return {
        "masked_ids": masked.tolist(),
        "selected": selected.tolist(),
        "hidden_shape": list(output.hidden.shape),
        "logits_shape": list(output.logits.shape),
        "mlm_loss": loss.item(),
        "checks": ["future source changes earlier logits", "finite MLM gradients"],
    }
