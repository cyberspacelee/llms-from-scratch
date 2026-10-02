"""16 · 推理基础支线：Prefill、Decode、Self/Cross KV 与流式 Encoder。"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..config import ModelConfig
from ..experiments.runners import consistency
from ..models import Transformer
from .ch11_gqa import config as gqa_config


def config() -> ModelConfig:
    """无输入；返回 GQA Decoder 配方，缓存是执行策略，不改变模型权重。

    Args:
        无显式输入。

    Returns:
        ModelConfig 本章的独立配方。
    """
    return gqa_config()


def streaming_encoder(device: torch.device) -> dict[str, object]:
    """因果 Encoder 可追加；输入层数和输出层数不同；编码 memory 可以复用。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 因果 Encoder追加、memory复用误差和模型检查。
    """
    c = replace(
        config(),
        architecture="encoder_decoder",
        encoder_causal=True,
        cross_causal=True,
        encoder_layers=1,
    )
    model = Transformer(c).to(device).double().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
    with torch.no_grad():
        full = model.encode(ids, use_cache=True)
        first = model.encode(ids[:, :3], use_cache=True)
        second = model.encode(ids[:, 3:], past=first, use_cache=True)
        torch.testing.assert_close(full.hidden, second.hidden, atol=1e-9, rtol=1e-7)
        direct = model(ids, source_ids=ids).logits
        reused = model(ids, memory=second).logits
        torch.testing.assert_close(direct, reused, atol=1e-9, rtol=1e-7)
    return {
        "encoder_layers": len(model.encoder_blocks),
        "decoder_layers": len(model.blocks),
        "encoder_append_error": (full.hidden - second.hidden).abs().max().item(),
        "memory_reuse_error": (direct - reused).abs().max().item(),
        "check": consistency(c, device),
    }


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 chunk/decode shape、完整前向误差和静态 Cross 对照。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    c = config()
    model = Transformer(c).to(device).double().eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)  # [B=1,S=6]
    cache, parts, shapes, start = None, [], [], 0
    with torch.no_grad():
        full = model(ids).logits  # [B,S,V]
        for size in (3, 1, 2):
            current = ids[:, start : start + size]  # [B,S_q]，decode 可为一个 token 或 chunk
            output = model(current, cache=cache, use_cache=True)
            cache = output.cache
            parts.append(output.logits)  # [B,S_q,V]，不包含已缓存位置的 logits
            shapes.append(
                {
                    "ids": list(current.shape),
                    "logits": list(output.logits.shape),
                    "kv": list(cache.layers[0].self_attention.key.shape),
                }
            )
            start += size  # S_kv=P+S_q
        actual = torch.cat(parts, 1)  # [B,S,V]
        torch.testing.assert_close(actual, full, atol=1e-9, rtol=1e-7)
    return {
        "chunks": shapes,
        "full_cached_error": (actual - full).abs().max().item(),
        "cross_cache": consistency(replace(c, architecture="encoder_decoder"), device),
        "streaming_encoder": streaming_encoder(device),
    }
