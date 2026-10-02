"""02 · 2018：GPT 路线的因果 Decoder-only 与 Next-Token Prediction。

读取 NaiveDecoder.forward，删除 Encoder/Cross 两条数据路径。
"""

from __future__ import annotations

import math

import torch
from torch import nn

from ..attention.position import sinusoidal
from ..config import ModelConfig
from ..experiments.runners import consistency
from ..training import next_token_loss
from .ch01_naive_transformer import NaiveAttention, classic_config


class NaiveDecoder(nn.Module):
    """固定两分支 Decoder；ids[B,T] -> embedding/hidden[B,T,D] -> logits[B,T,V]。"""

    def __init__(self, vocab_size: int = 32, dim: int = 32, num_heads: int = 4) -> None:
        """输入词表 V、hidden D、头数 H_q；构造模型，返回 None。

        Args:
            vocab_size: 词表大小 V，正整数。
            dim: model dimension D，必须能被 num_heads 整除。
            num_heads: query head 数 H_q，必须为正整数。

        Returns:
            None；参数与子层注册在 self 中。
        """
        super().__init__()
        if type(vocab_size) is not int or vocab_size < 1:
            raise ValueError("vocabulary size must be positive")
        self.dim = dim
        self.embedding = nn.Embedding(vocab_size, dim)
        self.attention = NaiveAttention(dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(dim, 2 * dim, bias=False), nn.ReLU(), nn.Linear(2 * dim, dim, bias=False)
        )
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """输入非空 long ids[B,T]，无 padding；返回 next-token logits[B,T,V]。

        Args:
            ids: 非空 long [B,T]，与模型同设备且 ID 在词表内。

        Returns:
            float logits[B,T,V]。
        """
        if ids.ndim != 2 or ids.dtype != torch.long or ids.numel() == 0:
            raise ValueError("expected nonempty long ids[B,T]")
        if (
            ids.device != self.embedding.weight.device
            or ids.min() < 0
            or ids.max() >= self.embedding.num_embeddings
        ):
            raise ValueError("invalid token device or vocabulary index")
        x = self.embedding(ids) * math.sqrt(self.dim)  # [B,T,D]
        positions = torch.arange(ids.shape[1], device=ids.device)  # [T]
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        x = x + sinusoidal(positions, self.dim, dtype).to(x.dtype)  # [T,D] broadcast
        x = self.norms[0](x + self.attention(x, x, causal=True))  # [B,T,D]
        x = self.norms[1](x + self.ff(x))  # FFN [B,T,D] -> [B,T,2D] -> [B,T,D]
        return self.head(x)  # [B,T,V]


def config() -> ModelConfig:
    """无输入；返回只含因果 Decoder 的经典配方，作为之后章节的起点。

    Args:
        无显式输入。

    Returns:
        ModelConfig 本章的独立配方。
    """
    from dataclasses import replace

    return replace(classic_config(), architecture="decoder")


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 NTP 标签/shape、训练梯度和因果缓存对照报告。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    model = NaiveDecoder().to(device)
    ids = torch.tensor([[1, 2, 3, 4, 5]], device=device)  # [1,5]
    logits = model(ids)  # [1,5,32]
    loss = next_token_loss(logits, ids)  # logits[:,:-1] [1,4,32] vs ids[:,1:] [1,4]
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    changed = ids.clone()
    changed[:, 3:] = 9
    torch.testing.assert_close(logits[:, :3], model(changed)[:, :3])
    return {
        "ids_shape": list(ids.shape),
        "logits_shape": list(logits.shape),
        "ntp_labels": ids[:, 1:].tolist(),
        "loss": loss.item(),
        "core_check": consistency(config(), device),
    }
