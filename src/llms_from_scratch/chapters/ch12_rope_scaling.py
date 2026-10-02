"""12 · 2023：Linear/NTK/YaRN 位置频率扩展；频率在会话中固定。"""

from __future__ import annotations

from dataclasses import replace

import torch

from ..attention.position import apply_rope, rotary_frequencies
from ..experiments.runners import consistency
from .ch11_gqa import config


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回各缩放的频率、shape、范数和缓存一致性报告。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    c = config()
    a = c.block.attention
    x = torch.randn(2, 4, 6, 8, device=device, dtype=torch.float64)  # [B,H_q,T,D_h]
    positions = torch.arange(6, device=device)
    report = {}
    for scaling in ("linear", "ntk", "yarn"):
        position = replace(a.position, scaling=scaling, factor=4)
        rotated = apply_rope(x, positions, position)
        torch.testing.assert_close(x.square().sum(-1), rotated.square().sum(-1))
        report[scaling] = {
            "frequencies": rotary_frequencies(8, position, device).tolist(),
            "input_shape": list(x.shape),
            "output_shape": list(rotated.shape),
            "check": consistency(
                replace(c, block=replace(c.block, attention=replace(a, position=position))), device
            ),
        }
    return report
