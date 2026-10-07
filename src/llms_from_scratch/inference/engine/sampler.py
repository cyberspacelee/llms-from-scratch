"""采样器：每个请求有自己的温度、top-k 与随机数生成器。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class SamplingParams:
    max_tokens: int = 16
    temperature: float = 0.0  # 0 表示贪心
    top_k: int | None = None
    seed: int | None = None


# region sample
def sample(logits: torch.Tensor, params: list[SamplingParams],
           generators: list[torch.Generator | None]) -> list[int]:
    """logits: [num_reqs, V]。与 GPT.generate 使用相同的温度、top-k 与 multinomial 步骤。"""
    out = []
    for row, p, g in zip(logits, params, generators, strict=True):
        if p.temperature == 0:
            out.append(int(row.argmax()))
            continue
        row = row / p.temperature
        if p.top_k is not None:
            kth = torch.topk(row, min(p.top_k, row.shape[-1])).values[-1]
            row = row.masked_fill(row < kth, float("-inf"))
        out.append(int(torch.multinomial(F.softmax(row, -1)[None], 1, generator=g)))
    return out
# endregion sample
