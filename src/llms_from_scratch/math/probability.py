"""概率、最大似然与交叉熵：语言模型损失背后的数学。

对应《概率、最大似然与交叉熵》一章。所有对数默认是自然对数（单位 nat），
除以 ln 2 换算成 bit。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# region information
def entropy(p: Tensor) -> Tensor:
    """H(p) = −Σ p log p，约定 0 log 0 = 0。"""
    return -(torch.where(p > 0, p * torch.log(p), torch.zeros_like(p))).sum(-1)


def cross_entropy(p: Tensor, q: Tensor) -> Tensor:
    """H(p, q) = −Σ p log q：用为 q 设计的编码去编码来自 p 的样本，平均码长。"""
    return -(torch.where(p > 0, p * torch.log(q), torch.zeros_like(p))).sum(-1)


def kl_divergence(p: Tensor, q: Tensor) -> Tensor:
    """KL(p ‖ q) = Σ p log(p/q) = H(p, q) − H(p) ≥ 0。"""
    return cross_entropy(p, q) - entropy(p)


# endregion


# region bigram
def bigram_counts(tokens: list[int], vocab_size: int) -> Tensor:
    """统计相邻 token 对 (a, b) 出现的次数，返回 (V, V) 计数矩阵。"""
    counts = torch.zeros(vocab_size, vocab_size, dtype=torch.float64)
    for a, b in zip(tokens[:-1], tokens[1:], strict=True):
        counts[a, b] += 1
    return counts


def bigram_mle(tokens: list[int], vocab_size: int, smoothing: float = 0.0) -> Tensor:
    """二元模型的最大似然估计：P(b | a) = count(a, b) / count(a, ·)。

    ``smoothing`` > 0 时为加性（Laplace）平滑，避免未见过的组合概率为 0。
    """
    counts = bigram_counts(tokens, vocab_size) + smoothing
    return counts / counts.sum(dim=-1, keepdim=True)


def sequence_nll(tokens: list[int], probs: Tensor) -> float:
    """自回归分解下整个序列的负对数似然：−Σ_t log P(x_t | x_{t−1})（第一个 token 视为给定）。"""
    return -sum(math.log(probs[a, b].item()) for a, b in zip(tokens[:-1], tokens[1:], strict=True))


def fit_bigram_logits(tokens: list[int], vocab_size: int, steps: int = 500, lr: float = 1.0) -> Tensor:
    """用梯度下降最小化交叉熵来训练一个 (V, V) 的 logits 表，返回 softmax 后的条件概率。

    这是“神经网络版”的二元模型：它收敛到的正是 ``bigram_mle`` 的计数估计。
    """
    prev = torch.tensor(tokens[:-1])
    nxt = torch.tensor(tokens[1:])
    logits = torch.zeros(vocab_size, vocab_size, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.SGD([logits], lr=lr)
    for _ in range(steps):
        loss = torch.nn.functional.cross_entropy(logits[prev], nxt)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.softmax(logits.detach(), dim=-1)


# endregion


# region metrics
def perplexity(mean_nll: float) -> float:
    """困惑度 = exp(平均每 token 负对数似然)：等价于“在多少个等可能选项中猜”。"""
    return math.exp(mean_nll)


def bits_per_byte(total_nll_nats: float, num_bytes: int) -> float:
    """把总负对数似然（nat）摊到原始文本的每个字节上，并换算成 bit。与分词器无关。"""
    return total_nll_nats / (math.log(2) * num_bytes)


def softmax_with_temperature(logits: Tensor, temperature: float) -> Tensor:
    """p_i ∝ exp(z_i / τ)。τ → 0 趋于 argmax，τ → ∞ 趋于均匀分布。"""
    return torch.softmax(logits / temperature, dim=-1)


# endregion
