"""23 · 2026：DeepSeek Engram / Qwen PLE 的因果 n-gram 条件记忆。

小表 bigram + context gate + dilated depthwise conv；每层可持有自己的 lookup 表。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class NGramMemory(nn.Module):
    """ids[B,T] 与 hidden[B,T,D] -> lexical/context feature[B,T,D]，无未来 token。"""

    def __init__(self, dim: int = 32, table_size: int = 257) -> None:
        """输入 hidden D、hash 表大小；创建词法表、context gate 和因果卷积，返回 None。

        Args:
            dim: 输入/输出 hidden width D；不要求等于 Attention 投影宽度。
            table_size: hash lookup 表行数，大于 1。

        Returns:
            None；参数与子层注册在 self 中。
        """
        super().__init__()
        if type(dim) is not int or dim < 1 or type(table_size) is not int or table_size < 2:
            raise ValueError("dimensions and table size must be positive")
        self.table_size = table_size
        self.embedding = nn.Embedding(table_size, dim)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.conv = nn.Conv1d(dim, dim, 3, groups=dim, dilation=2, bias=False)

    def forward(self, ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """输入非负 long ids[B,T]、hidden[B,T,D]；返回 gated memory[B,T,D]。

        哈希会碰撞；此处只演示 bigram，无 tokenizer 压缩或增量 hash/conv cache。

        Args:
            ids: 非空 long [B,T]，与模型同设备且 ID 在词表内。
            hidden: float [B,T,D] 主模型 hidden states。

        Returns:
            float lexical/context features[B,T,D]。
        """
        if (
            ids.ndim != 2
            or ids.dtype != torch.long
            or ids.numel() == 0
            or (ids < 0).any()
            or hidden.ndim != 3
            or hidden.shape[:2] != ids.shape
            or hidden.shape[-1] != self.embedding.embedding_dim
            or ids.device != hidden.device
        ):
            raise ValueError("expected aligned token IDs and hidden states")
        previous = F.pad(ids, (1, 0))[:, :-1]  # [B,T]，首位置左侧零
        # ponytail: small polynomial hash, collisions intentional; multi-order/head prime tables reduce them.
        hashes = ((previous % self.table_size) * 131 + ids % self.table_size) % self.table_size
        lexical = self.embedding(hashes)  # [B,T,D]
        scores = (F.normalize(hidden, dim=-1) * F.normalize(self.key(lexical), dim=-1)).sum(
            -1, keepdim=True
        )  # [B,T,1]
        gated = scores.sigmoid() * self.value(lexical)  # [B,T,D]
        causal = self.conv(F.pad(gated.transpose(1, 2), (4, 0))).transpose(
            1, 2
        )  # [B,D,T+4] -> [B,T,D]
        return gated + F.silu(causal)


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 n-gram feature shape、未来隔离和 lookup 表梯度检查。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    module = NGramMemory().to(device).double()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], device=device)
    hidden = torch.randn(1, 6, 32, device=device, dtype=torch.float64)
    output = module(ids, hidden)
    changed = ids.clone()
    changed[:, 4:] = 9
    altered_hidden = hidden.clone()
    altered_hidden[:, 4:] += 100
    torch.testing.assert_close(output[:, :4], module(changed, altered_hidden)[:, :4])
    output.square().mean().backward()
    assert (
        module.embedding.weight.grad is not None
        and torch.isfinite(module.embedding.weight.grad).all()
    )
    return {
        "ids_shape": list(ids.shape),
        "hidden_shape": list(hidden.shape),
        "hash_shape": list(ids.shape),
        "memory_shape": list(output.shape),
        "checks": ["causal n-gram and dilated conv", "finite lookup gradients"],
        "note": "Engram and PLE have different official hash/gating layouts; this is their shared principle",
    }
