"""滑动窗口注意力、环形 KV 缓存与注意力汇聚点（attention sink）。

窗口注意力让位置 t 只看 (t - W, t]，于是 decode 时只需保留最近 W 个位置的 K/V：
用一个长度为 W 的环形缓冲区，第 t 个 token 写到槽位 t mod W（Mistral 7B 的 rolling
buffer cache）。StreamingLLM 进一步把最早的 n_sink 个 token 永久留在缓存里。
"""

from __future__ import annotations

import math

import torch


def window_mask(T: int, window: int, n_sink: int = 0) -> torch.Tensor:
    """[T, T] 布尔掩码：因果，且只保留最近 window 个位置以及开头 n_sink 个位置。"""
    i = torch.arange(T).unsqueeze(1)
    j = torch.arange(T).unsqueeze(0)
    recent = (j <= i) & (j > i - window)
    sink = (j < n_sink) & (j <= i)
    return recent | sink


def masked_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    return scores.masked_fill(~mask, float("-inf")).softmax(-1) @ v


# region ring
class SinkRingCache:
    """n_sink 个固定槽位 + window 个环形槽位，显存与已生成长度无关。"""

    def __init__(self, window: int, dim: int, n_sink: int = 0) -> None:
        self.window, self.n_sink = window, n_sink
        self.k = torch.zeros(n_sink + window, dim)
        self.v = torch.zeros(n_sink + window, dim)
        self.filled = torch.zeros(n_sink + window, dtype=torch.bool)
        self.pos = torch.full((n_sink + window,), -1)  # 每个槽位存的是哪个位置

    def append(self, t: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if t < self.n_sink:
            slot = t  # 汇聚点：永不覆盖
        else:
            slot = self.n_sink + (t - self.n_sink) % self.window  # 环形：覆盖最旧的
        self.k[slot], self.v[slot], self.filled[slot], self.pos[slot] = k, v, True, t

    def attend(self, q: torch.Tensor) -> torch.Tensor:
        """用当前查询读取缓存。键的顺序无关紧要：softmax 对键的排列不变。"""
        return masked_attention(q.unsqueeze(0), self.k, self.v, self.filled.unsqueeze(0))[0]
# endregion
