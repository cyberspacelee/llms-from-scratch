"""10 · 2022：分块 Online Softmax 原理与 PyTorch SDPA 数值对照。

下面是可微 PyTorch 参考，不是 FlashAttention 的 GPU kernel。
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

from ..attention import scaled_dot_product_attention


def online_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, block_size: int = 2
) -> torch.Tensor:
    """输入 Q[B,H_q,T_q,D_h]、K[B,H_q,T_kv,D_h]、V[B,H_q,T_kv,D_v]。

    返回 O[B,H_q,T_q,D_v]；按 KV 轴分块，不分配完整 T_q×T_kv scores。
    这里只处理无 mask 的非空浮点张量；本例不测 kernel IO/速度。

    Args:
        q: float Q[B,H_q,T_q,D_h]。
        k: float K[B,H_q,T_kv,D_h]。
        v: float V[B,H_q,T_kv,D_v]。
        block_size: 块宽 R，正整数。

    Returns:
        float [B,H_q,T_q,D_v]，fp32/fp64统计dtype。
    """
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or q.shape[:2] != k.shape[:2]
        or k.shape[:3] != v.shape[:3]
        or q.shape[-1] != k.shape[-1]
        or min(q.shape + k.shape + v.shape) < 1
    ):
        raise ValueError("expected nonempty aligned attention tensors")
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be positive")
    if any(
        t.dtype != q.dtype or t.device != q.device or not t.is_floating_point() for t in (q, k, v)
    ):
        raise ValueError("attention tensors must share floating dtype/device")
    dtype = torch.float64 if q.dtype == torch.float64 else torch.float32
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    maximum = q.new_full((*q.shape[:-1], 1), -torch.inf)  # [B,H_q,T_q,1]
    denominator = torch.zeros_like(maximum)  # [B,H_q,T_q,1]
    numerator = q.new_zeros((*q.shape[:-1], v.shape[-1]))  # [B,H_q,T_q,D_v]
    # ponytail: only KV tiling; tile Q too for bounded per-tile memory on long queries.
    for start in range(0, k.shape[-2], block_size):
        scores = (
            q @ k[:, :, start : start + block_size].transpose(-1, -2) * q.shape[-1] ** -0.5
        )  # [B,H_q,T_q,C]
        updated_max = torch.maximum(maximum, scores.amax(-1, keepdim=True))
        correction = (maximum - updated_max).exp()  # [B,H_q,T_q,1]
        weights = (scores - updated_max).exp()  # [B,H_q,T_q,C]
        denominator = correction * denominator + weights.sum(-1, keepdim=True)
        numerator = correction * numerator + weights @ v[:, :, start : start + block_size]
        maximum = updated_max
    return numerator / denominator  # [B,H_q,T_q,D_v]，保持统计 dtype


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 online/manual/SDPA 输出与输入梯度对照报告。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    tensors = [
        torch.randn(2, 3, length, width, device=device, dtype=torch.float64)
        for length, width in ((5, 8), (7, 8), (7, 6))
    ]
    reference_inputs = [t.clone().requires_grad_() for t in tensors]
    online_inputs = [t.clone().requires_grad_() for t in tensors]
    manual = scaled_dot_product_attention(*reference_inputs)
    online = online_attention(*online_inputs, block_size=3)
    sdpa = F.scaled_dot_product_attention(*tensors)
    torch.testing.assert_close(online, manual)
    torch.testing.assert_close(sdpa, manual)
    manual.square().mean().backward()
    online.square().mean().backward()
    for left, right in zip(reference_inputs, online_inputs, strict=True):
        torch.testing.assert_close(left.grad, right.grad)
    return {
        "query_shape": list(tensors[0].shape),
        "key_shape": list(tensors[1].shape),
        "value_shape": list(tensors[2].shape),
        "output_shape": list(online.shape),
        "score_tile_shape": [2, 3, 5, 3],
        "online_manual_error": (online - manual).abs().max().item(),
        "sdpa_manual_error": (sdpa - manual).abs().max().item(),
        "note": "CPU reference; SDPA does not guarantee a FlashAttention backend",
    }
