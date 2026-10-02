"""19 · 长上下文研究支线：旧块摘要 + 精确近期 token，压缩 sequence 轴。

这是 mean-pooling 原理，不是 DeepSeek CSA/HCA 的完整 learned compressor。
"""

from __future__ import annotations

import torch

from ..inference import compressed_causal_attention
from ..inference.compression import compress_sequence


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 pooling shape、chunk/full一致性和无未来泄漏检查。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    q, k, v = (torch.randn(1, 2, 7, 8, device=device, dtype=torch.float64) for _ in range(3))
    pooled_k, pooled_v, ends = compress_sequence(k, v, block_size=2)  # [B,H_q,3,D_h/D_v], ends[3]
    full = compressed_causal_attention(q, k, v, 2, 2)  # [B,H_q,T_q,D_v]
    parts = [
        compressed_causal_attention(q[:, :, i : i + 1], k[:, :, : i + 1], v[:, :, : i + 1], 2, 2, i)
        for i in range(7)
    ]
    torch.testing.assert_close(full, torch.cat(parts, -2))
    changed = v.clone()
    changed[:, :, 4:] += 100
    torch.testing.assert_close(
        full[:, :, :4], compressed_causal_attention(q, k, changed, 2, 2)[:, :, :4]
    )
    return {
        "input_shape": list(k.shape),
        "pooled_key_shape": list(pooled_k.shape),
        "pooled_value_shape": list(pooled_v.shape),
        "block_end_positions": ends.tolist(),
        "output_shape": list(full.shape),
        "checks": ["full vs tokenwise", "no future summary leakage"],
    }
