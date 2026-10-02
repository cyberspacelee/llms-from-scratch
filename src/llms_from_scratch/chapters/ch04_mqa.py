"""04 · 2019：Multi-Query Attention，独立查看 KV 投影与 query 分组。

MQA H_kv=1；GQA 1<H_kv<H_q；两者的 Q 头数保持不变。
"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..analysis import attention_cost
from ..attention import scaled_dot_product_attention
from ..attention.position import apply_rope
from ..config import ModelConfig
from ..experiments.runners import consistency
from ..models import make_attention
from .ch02_decoder_only import config as previous_config


def config() -> ModelConfig:
    """无输入；返回 MQA 配方，仅调整 KV 投影头数。

    Args:
        无显式输入。

    Returns:
        ModelConfig 本章的独立配方。
    """
    c = previous_config()
    return replace(
        c, block=replace(c.block, attention=replace(c.block.attention, kind="mqa", kv_heads=1))
    )


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 Q/K/V shape、逐头分组对照误差和缓存成本。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    c = config()
    a = c.block.attention
    module = make_attention(c.dim, a).to(device).double().eval()
    x = torch.randn(2, 5, c.dim, device=device, dtype=torch.float64)  # [B,T,D]
    y, cache = module(x, causal=True, use_cache=True)
    # Q: [B,T,D] -> [B,T,H_q,D_h] -> [B,H_q,T,D_h]。
    q = module.q(x).reshape(2, 5, a.heads, a.head_dim).transpose(1, 2)
    if a.position.kind == "rope":
        q = apply_rope(q, torch.arange(5, device=device), a.position)
    # cache K/V: [B,H_kv,T,D_h]，不保存临时扩展到 H_q 的副本。
    pieces = []
    visible = torch.ones(5, 5, device=device, dtype=torch.bool).tril()
    for head in range(a.heads):
        group = head // (a.heads // a.effective_kv_heads)
        pieces.append(
            scaled_dot_product_attention(
                q[:, head : head + 1],
                cache.key[:, group : group + 1],
                cache.value[:, group : group + 1],
                visible,
            )
        )  # [B,1,T,D_h]
    reference = module.output(torch.cat(pieces, 1).transpose(1, 2).reshape(2, 5, c.dim))  # [B,T,D]
    torch.testing.assert_close(y, reference)
    cost = attention_cost(a, c.dim, batch=2, query_tokens=5, key_tokens=5, bytes_per_element=8)
    assert cache.nbytes == cost.cache_bytes
    return {
        "query_shape": list(q.shape),
        "key_shape": list(cache.key.shape),
        "value_shape": list(cache.value.shape),
        "output_shape": list(y.shape),
        "query_to_kv_head": [h // (a.heads // a.effective_kv_heads) for h in range(a.heads)],
        "reference_error": (y - reference).abs().max().item(),
        "cost": cost.to_dict(),
        "model_check": consistency(c, device),
    }
