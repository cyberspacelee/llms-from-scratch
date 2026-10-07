"""多 token 预测（MTP）：并行预测头（Gloeckle 等 2024）与 DeepSeek-V3 的顺序 MTP 模块。

记号：序列 t_0 … t_{T-1}。主模型在位置 i 预测 t_{i+1}；第 k 个 MTP 深度在位置 i 预测 t_{i+1+k}。
主干复用 ``transformer.model.GPT``，MTP 模块复用它的 ``Block``，并与主模型共享嵌入和输出头。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from llms_from_scratch.transformer.model import GPT, Block, RMSNorm


# region targets
def mtp_targets(tokens: torch.Tensor, k: int) -> torch.Tensor:
    """深度 k（k=0 为主模型）的标签：位置 i 的目标是 t_{i+1+k}。

    tokens[B, T] → targets[B, T-1-k]，只有前 T-1-k 个位置有标签。
    """
    return tokens[:, 1 + k:]
# endregion


def hidden_states(model: GPT, idx: torch.Tensor) -> torch.Tensor:
    """主干最后一层的输出（最终 RMSNorm 与输出头之前），形状 [B, T, d]。"""
    x = model.embed(idx)
    freqs = model.freqs[: idx.shape[1]]
    for block in model.blocks:
        x = block(x, freqs)
    return x


class ParallelHeads(nn.Module):
    """Gloeckle 等：共享主干之上并列 n 个独立的 Transformer 层，第 j 个预测 t_{i+1+j}。"""

    def __init__(self, model: GPT, n_future: int) -> None:
        super().__init__()
        self.model = model
        self.heads = nn.ModuleList(Block(model.config, model.config.n_layers + j) for j in range(n_future))

    def forward(self, idx: torch.Tensor) -> list[torch.Tensor]:
        h = hidden_states(self.model, idx)
        freqs = self.model.freqs[: idx.shape[1]]
        # 每个头都只看主干输出 h，彼此之间没有依赖，可以同时计算
        return [self.model.lm_head(self.model.norm(head(h, freqs))) for head in self.heads]


# region sequential
class MTPModule(nn.Module):
    """DeepSeek-V3 的一个 MTP 深度：h'_i = M [RMSNorm(h_i^{k-1}); RMSNorm(Emb(t_{i+k}))]，再过一个块。"""

    def __init__(self, model: GPT, depth: int) -> None:
        super().__init__()
        d = model.config.d_model
        self.norm_h = RMSNorm(d)
        self.norm_e = RMSNorm(d)
        self.proj = nn.Linear(2 * d, d, bias=False)  # M_k
        self.block = Block(model.config, model.config.n_layers + depth)


class SequentialMTP(nn.Module):
    def __init__(self, model: GPT, depth: int = 1) -> None:
        super().__init__()
        self.model = model
        self.modules_ = nn.ModuleList(MTPModule(model, k) for k in range(depth))

    def forward(self, idx: torch.Tensor) -> list[torch.Tensor]:
        """返回 [主模型 logits, 深度 1 logits, …]；深度 k 的 logits 长度为 T - k。"""
        m = self.model
        h = hidden_states(m, idx)
        outputs = [m.lm_head(m.norm(h))]
        for k, mod in enumerate(self.modules_, start=1):
            T_k = idx.shape[1] - k
            # 位置 i 拼接上一深度的表示 h_i 与“下一个真实 token” t_{i+k} 的嵌入：保持完整的因果链
            emb = m.embed(idx[:, k:])  # Emb(t_{i+k})，i = 0 … T-1-k
            h = mod.proj(torch.cat([mod.norm_h(h[:, :T_k]), mod.norm_e(emb)], dim=-1))
            h = mod.block(h, m.freqs[:T_k])
            outputs.append(m.lm_head(m.norm(h)))  # 共享输出头
        return outputs
# endregion


# region loss
def mtp_loss(outputs: list[torch.Tensor], idx: torch.Tensor, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (主损失, MTP 损失)。MTP 损失 = λ/D · Σ_k CE(深度 k 的预测, t_{i+1+k})。"""
    def ce(logits: torch.Tensor, k: int) -> torch.Tensor:
        targets = mtp_targets(idx, k)
        logits = logits[:, : targets.shape[1]]
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    main = ce(outputs[0], 0)
    D = len(outputs) - 1
    extra = sum(ce(o, k) for k, o in enumerate(outputs[1:], start=1)) if D else torch.zeros(())
    return main, lam / max(D, 1) * extra
# endregion
