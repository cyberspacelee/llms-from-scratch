"""13 · 2024：mla，机制与数值核对独立成章。"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..analysis import attention_cost
from ..config import AttentionConfig, ModelConfig
from ..experiments.runners import consistency
from ..models import make_attention
from .ch11_gqa import config as gqa_config


def config() -> ModelConfig:
    """无输入；返回本章机制的显式配置，其他支线不自动启用。

    Args:
        无显式输入。

    Returns:
        ModelConfig 本章的独立配方。
    """
    c = gqa_config()
    a = AttentionConfig(
        kind="mla", heads=4, head_dim=8, value_dim=8, kv_rank=8, q_rank=8, rope_dim=4
    )
    return replace(c, block=replace(c.block, attention=a))


def run(device: torch.device) -> dict[str, object]:
    """两个路径必须共享权重，decode 必须共享同一段 latent 前缀。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    c = config()
    a = c.block.attention
    naive = make_attention(c.dim, replace(a, mla_impl="naive")).to(device).double().eval()
    absorbed = make_attention(c.dim, replace(a, mla_impl="absorbed")).to(device).double().eval()
    absorbed.load_state_dict(naive.state_dict())
    x = torch.randn(1, 6, c.dim, device=device, dtype=torch.float64)
    with torch.no_grad():
        _, prefix = naive(x[:, :5], causal=True, use_cache=True)
        left, _ = naive(x[:, 5:], cache=prefix, causal=True, query_offset=5)
        right, state = absorbed(x[:, 5:], cache=prefix, causal=True, query_offset=5, use_cache=True)
        torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-7)
    return {
        "latent_shape": list(state.key.shape),
        "rotary_key_shape": list(state.value.shape),
        "naive_absorbed_decode_error": (left - right).abs().max().item(),
        "cost_fp32_S1024": attention_cost(a, c.dim, key_tokens=1024, bytes_per_element=4).to_dict(),
        "model_check": consistency(c, device),
    }
