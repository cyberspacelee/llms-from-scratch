"""24 · 2024–2026：YOCO / DeepSeek V4.1 CED 的跨层 KV 共享。

全局 KV 由 lower causal layers 构建，upper consumers 复用；不是经典 seq2seq static Cross。
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from ..attention import scaled_dot_product_attention
from ..models import Transformer
from .ch08_rope import config


def read_shared_kv(
    x: torch.Tensor,
    shared_kv: torch.Tensor,
    query: nn.Linear,
    output: nn.Linear,
    num_heads: int = 4,
    query_offset: int = 0,
) -> torch.Tensor:
    """输入 x[B,S_q,D]、shared_kv[B,1,S_kv,D_h]、Q/O 投影和绝对 query 起点。

    返回 hidden[B,S_q,D]；共享 latent 同时充当 K/V，不重复保存到每个 consumer。

    Args:
        x: float [B,S_q,D] 当前 consumer 的 hidden。
        shared_kv: float [B,1,S_kv,D_h]，多个 consumer 只读共享。
        query: nn.Linear，权重 [H_q*D_h,D] 的 query 投影。
        output: nn.Linear，weight[D,H_q*D_h]，将合头结果投影回 D。
        num_heads: query head 数 H_q，必须为正整数。
        query_offset: query 绝对位置起点 P；Self cache 时等于前缀长度。

    Returns:
        float [B,S_q,D] 共享全局 KV分支输出。
    """
    if (
        x.ndim != 3
        or shared_kv.ndim != 4
        or shared_kv.shape[:2] != (x.shape[0], 1)
        or type(num_heads) is not int
        or num_heads < 1
        or query.out_features != num_heads * shared_kv.shape[-1]
        or output.in_features != query.out_features
        or query_offset < 0
        or query_offset + x.shape[1] > shared_kv.shape[-2]
    ):
        raise ValueError("incompatible shared KV or query position")
    batch_size, query_len, _ = x.shape
    q = (
        query(x).reshape(batch_size, query_len, num_heads, shared_kv.shape[-1]).transpose(1, 2)
    )  # [B,H_q,S_q,D_h]
    keys = shared_kv.expand(-1, num_heads, -1, -1)  # [B,H_q,S_kv,D_h]，view 共享底层存储
    qp = torch.arange(query_offset, query_offset + query_len, device=x.device)
    kp = torch.arange(shared_kv.shape[-2], device=x.device)
    visible = kp[None, :] <= qp[:, None]  # [S_q,S_kv]
    context = scaled_dot_product_attention(q, keys, keys, visible)
    return output(context.transpose(1, 2).reshape(batch_size, query_len, -1))


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 shared/global cache shape、多个 consumer和 chunk一致性检查。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    lower = (
        Transformer(replace(config(), architecture="encoder", encoder_causal=True))
        .to(device)
        .double()
    )
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
    hidden = lower(ids).hidden  # [B,S,D]，lower stack 因果输出
    projection = nn.Linear(32, 8, bias=False, device=device, dtype=hidden.dtype)
    shared = projection(hidden).unsqueeze(1)  # [B,1,S_kv,D_h]，只构建一次
    queries = nn.ModuleList(
        [nn.Linear(32, 32, bias=False, device=device, dtype=hidden.dtype) for _ in range(2)]
    )
    outputs = nn.ModuleList(
        [nn.Linear(32, 32, bias=False, device=device, dtype=hidden.dtype) for _ in range(2)]
    )
    x = hidden
    for query, output in zip(queries, outputs, strict=True):
        full = read_shared_kv(x, shared, query, output)
        pieces = [
            read_shared_kv(x[:, i : i + 1], shared[:, :, : i + 1], query, output, query_offset=i)
            for i in range(6)
        ]
        torch.testing.assert_close(full, torch.cat(pieces, 1))
        x = x + full
    x.square().mean().backward()
    assert projection.weight.grad is not None and torch.isfinite(projection.weight.grad).all()
    assert lower.embedding.weight.grad is not None
    return {
        "lower_hidden_shape": list(hidden.shape),
        "shared_global_kv_shape": list(shared.shape),
        "consumer_count": len(queries),
        "output_shape": list(x.shape),
        "shared_elements": shared.numel(),
        "separate_copies_elements": len(queries) * shared.numel(),
        "checks": ["shared K/V gradient", "chunk/full with absolute positions"],
        "note": "CED/CSA2 mechanism reference; per-layer local windows, hierarchical indices and prefill bypass scheduling are explained in the chapter",
    }
