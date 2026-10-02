"""07 · 2020：Sparse/Local Attention 与 Linear Attention 两条效率路线。"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.nn import functional as F

from ..attention import gathered_attention, scaled_dot_product_attention
from ..attention.patterns import attention_mask
from ..config import AttentionConfig, PositionConfig
from ..models import make_attention


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 sparse/dense、linear/parallel 两组同权重数学对照。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    positions = torch.arange(6, device=device)  # [S]
    a = AttentionConfig(position=PositionConfig(kind="none"))
    patterns = {
        name: attention_mask(
            positions, positions, replace(a, pattern=name, window=2, block_size=2), True
        )[0, 0]
        .int()
        .tolist()
        for name in ("global", "sliding", "local", "block_sparse", "token_sparse")
    }
    q, k, v = (torch.randn(1, 2, 6, 8, device=device, dtype=torch.float64) for _ in range(3))
    indices = torch.stack(((positions - 1).clamp_min(0), positions), -1)  # [S_q,M=2]
    valid = torch.ones_like(indices, dtype=torch.bool)
    valid[0, 0] = False
    mask = torch.zeros(6, 6, device=device, dtype=torch.bool)  # [S_q,S_kv]
    mask[positions[:, None].expand_as(indices)[valid], indices[valid]] = True
    sparse = gathered_attention(q, k, v, indices, valid)  # [B,H_q,S_q,D_v]
    dense = scaled_dot_product_attention(q, k, v, mask)
    torch.testing.assert_close(sparse, dense)
    linear = make_attention(32, replace(a, kind="linear")).to(device).double()
    x = torch.randn(1, 6, 32, device=device, dtype=torch.float64)  # [B,S,D]
    q, k = (
        F.elu(layer(x).reshape(1, 6, 4, 8).transpose(1, 2)) + 1 for layer in (linear.q, linear.k)
    )
    v = linear.v(x).reshape(1, 6, 4, 8).transpose(1, 2)
    weights = (q @ k.transpose(-1, -2)) * torch.ones(6, 6, device=device).tril()  # [B,H_q,S,S]
    context = weights @ v / weights.sum(-1, keepdim=True)  # [B,H_q,S,D_v]
    parallel = linear.output(context.transpose(1, 2).reshape(1, 6, 32))
    recurrent, state = linear(x, causal=True, use_cache=True)
    torch.testing.assert_close(parallel, recurrent)
    return {
        "visible_matrices": patterns,
        "indices_shape": list(indices.shape),
        "sparse_output_shape": list(sparse.shape),
        "sparse_dense_error": (sparse - dense).abs().max().item(),
        "linear_parallel_error": (parallel - recurrent).abs().max().item(),
        "linear_state_shape": list(state.state.shape),
    }
