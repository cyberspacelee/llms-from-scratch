"""评估：困惑度与 bits-per-byte、对数似然多项选择、pass@k、n-gram 污染检测。"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# region bpb
def perplexity(nll_sum: float, num_tokens: int) -> float:
    """PPL = exp(平均每 token 负对数似然)，NLL 以 nat 为单位。"""
    return math.exp(nll_sum / num_tokens)


def bits_per_byte(nll_sum: float, num_bytes: int) -> float:
    """BPB = 总 NLL（nat）/ (ln 2 · UTF-8 字节数)。与分词器无关，可以跨模型比较。"""
    return nll_sum / (math.log(2) * num_bytes)


@torch.no_grad()
def sequence_nll(model: nn.Module, ids: torch.Tensor) -> tuple[float, int]:
    """整段序列的总 NLL：第一个 token 作为条件，预测其余 T-1 个。返回（NLL 总和，预测数）。"""
    logits = model(ids[None, :-1])[0].float()
    nll = F.cross_entropy(logits, ids[1:], reduction="sum")
    return nll.item(), ids.numel() - 1
# endregion bpb


# region multiple_choice
@torch.no_grad()
def continuation_logprob(model: nn.Module, context: list[int], continuation: list[int]) -> float:
    """log P(continuation | context) = Σ_t log P(c_t | context, c_<t)。"""
    ids = torch.tensor(context + continuation)
    logits = model(ids[None, :-1])[0].float()
    logp = logits.log_softmax(-1)
    start = len(context) - 1                       # 第一个续写 token 由位置 len(context)-1 预测
    targets = ids[len(context):]
    return logp[start:start + len(continuation)].gather(1, targets[:, None]).sum().item()


def score_choices(model: nn.Module, encode: Callable[[str], list[int]], context: str,
                  choices: Sequence[str], normalize: str = "none") -> list[float]:
    """每个选项的得分。normalize：none 原始对数似然；token 除以 token 数；byte 除以字节数。"""
    ctx = encode(context)
    scores = []
    for choice in choices:
        cont = encode(choice)
        lp = continuation_logprob(model, ctx, cont)
        if normalize == "token":
            lp /= len(cont)
        elif normalize == "byte":
            lp /= len(choice.encode())
        scores.append(lp)
    return scores


def multiple_choice_accuracy(model: nn.Module, encode: Callable[[str], list[int]],
                             items: Sequence[dict], normalize: str = "none") -> float:
    """items 中每项含 context、choices、answer（正确选项下标）；得分最高者为预测。"""
    correct = 0
    for item in items:
        scores = score_choices(model, encode, item["context"], item["choices"], normalize)
        correct += int(int(np.argmax(scores)) == item["answer"])
    return correct / len(items)
# endregion multiple_choice


# region pass_at_k
def pass_at_k(n: int, c: int, k: int) -> float:
    """无偏估计 1 - C(n-c, k) / C(n, k)：n 个样本中有 c 个正确时，随机取 k 个至少一个正确的概率。

    写成连乘 1 - Π_{i=n-c+1}^{n} (1 - k/i)，避免大组合数溢出（Chen et al. 2021）。
    """
    if n - c < k:
        return 1.0
    return 1.0 - float(np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def naive_pass_at_k(successes: Sequence[bool], k: int) -> float:
    """有偏的朴素做法：只看前 k 个样本是否有正确的。用于与无偏估计对照。"""
    return float(any(successes[:k]))
# endregion pass_at_k


# region answers
def extract_final_number(text: str) -> str | None:
    """GSM8K 式答案抽取：优先取 '####' 之后的数，否则取文本中最后一个数。"""
    if "####" in text:
        text = text.split("####")[-1]
    numbers = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None


def accuracy_stderr(accuracy: float, n: int) -> float:
    """二项分布的标准误 √(p(1-p)/n)：n 道题的基准上 ±2 个标准误大致是 95% 置信区间。"""
    return math.sqrt(accuracy * (1 - accuracy) / n)
# endregion answers


# region contamination
def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class NgramIndex:
    """训练语料的 n-gram 集合，用于检测评估样本是否在训练中出现过（GPT-3 用 13-gram）。"""

    def __init__(self, n: int = 13) -> None:
        self.n = n
        self.grams: set[tuple[str, ...]] = set()

    def add(self, text: str) -> None:
        words = _tokens(text)
        for i in range(len(words) - self.n + 1):
            self.grams.add(tuple(words[i:i + self.n]))

    def overlap(self, text: str) -> float:
        """评估样本中出现在训练语料里的 n-gram 比例；样本短于 n 个词时返回 0。"""
        words = _tokens(text)
        grams = [tuple(words[i:i + self.n]) for i in range(len(words) - self.n + 1)]
        if not grams:
            return 0.0
        return sum(g in self.grams for g in grams) / len(grams)

    def is_contaminated(self, text: str, min_fraction: float = 0.0) -> bool:
        """overlap > min_fraction 即视为污染；GPT-3 的规则是“有任一 13-gram 命中”。"""
        return self.overlap(text) > min_fraction
# endregion contamination
