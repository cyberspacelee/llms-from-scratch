"""监督微调（SFT）：带掩码的交叉熵、样本打包与文档隔离的注意力。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from llms_from_scratch.post_training.chat_template import IGNORE_INDEX
from llms_from_scratch.post_training.common import gpt_hidden
from llms_from_scratch.transformer.model import GPT


# region sft_loss
def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """logits[B, T, V]、labels[B, T]（与 input_ids 对齐，-100 表示不监督）。

    位置 t 的 logits 预测位置 t+1 的 token，因此先错开一位；
    再对所有被监督的 token 取平均（而不是先对每条样本平均）。
    """
    shift_logits = logits[:, :-1].reshape(-1, logits.shape[-1])
    shift_labels = labels[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)
# endregion sft_loss


@dataclass
class PackedBatch:
    input_ids: torch.Tensor  # [B, L]
    labels: torch.Tensor  # [B, L]
    doc_ids: torch.Tensor  # [B, L]，同一行内每个 token 所属文档的编号，填充为 -1
    position_ids: torch.Tensor  # [B, L]，每个文档内部从 0 开始


# region pack
def pack_examples(examples: list[tuple[list[int], list[int]]], max_len: int,
                  pad_id: int) -> PackedBatch:
    """把若干 (ids, labels) 首尾相接装进长度为 max_len 的行（贪心，按顺序装箱）。

    每个文档的第一个标签强制为 -100：它会被上一个文档的最后一个 token“预测”，
    这种跨文档的预测没有意义。
    """
    rows: list[list[tuple[list[int], list[int]]]] = [[]]
    used = 0
    for ids, labels in examples:
        if len(ids) > max_len:
            raise ValueError("单条样本超过 max_len，应先截断")
        if used + len(ids) > max_len:
            rows.append([])
            used = 0
        rows[-1].append((ids, labels))
        used += len(ids)
    B = len(rows)
    input_ids = torch.full((B, max_len), pad_id)
    out_labels = torch.full((B, max_len), IGNORE_INDEX)
    doc_ids = torch.full((B, max_len), -1)
    position_ids = torch.zeros(B, max_len, dtype=torch.long)
    for r, row in enumerate(rows):
        start = 0
        for d, (ids, labels) in enumerate(row):
            end = start + len(ids)
            input_ids[r, start:end] = torch.tensor(ids)
            out_labels[r, start:end] = torch.tensor([IGNORE_INDEX] + labels[1:])
            doc_ids[r, start:end] = d
            position_ids[r, start:end] = torch.arange(len(ids))
            start = end
    return PackedBatch(input_ids, out_labels, doc_ids, position_ids)


def document_causal_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """[B, L] → [B, L, L] 布尔掩码：同一文档内且 key 不在 query 之后才允许注意。"""
    L = doc_ids.shape[1]
    same_doc = doc_ids.unsqueeze(2) == doc_ids.unsqueeze(1)
    causal = torch.ones(L, L, dtype=torch.bool, device=doc_ids.device).tril()
    return same_doc & causal
# endregion pack


def packed_logits(model: GPT, batch: PackedBatch) -> torch.Tensor:
    """在打包后的批次上做前向：块对角因果掩码 + 每个文档位置从 0 重新计数。"""
    hidden = gpt_hidden(model, batch.input_ids, document_causal_mask(batch.doc_ids),
                        batch.position_ids)
    return model.lm_head(hidden)


def pad_examples(examples: list[tuple[list[int], list[int]]], pad_id: int
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """不打包时的做法：右侧填充到同一长度，填充位置标签为 -100。"""
    L = max(len(ids) for ids, _ in examples)
    ids = torch.full((len(examples), L), pad_id)
    labels = torch.full((len(examples), L), IGNORE_INDEX)
    for i, (x, y) in enumerate(examples):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
    return ids, labels


def train_sft(model: GPT, examples: list[tuple[list[int], list[int]]], pad_id: int,
              steps: int = 100, lr: float = 3e-3, batch_size: int = 16, seed: int = 0) -> list[float]:
    """最朴素的 SFT 循环：随机取一批、填充、带掩码交叉熵、AdamW。返回每步损失。"""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    model.train()
    for _ in range(steps):
        pick = torch.randint(len(examples), (batch_size,), generator=g).tolist()
        ids, labels = pad_examples([examples[i] for i in pick], pad_id)
        loss = sft_loss(model(ids), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses
