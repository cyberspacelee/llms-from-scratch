"""第一个神经语言模型：字符级 bigram 计数模型、可训练 bigram 与 Bengio (2003) 式 MLP。

语料 ``data/corpus.txt`` 是若干公有领域英文文本的节选（葛底斯堡演说、《独立宣言》开篇、
《傲慢与偏见》《爱丽丝漫游奇境》《双城记》《白鲸》开头、莎士比亚十四行诗、KJV 圣经片段），
约 13 KB，随仓库分发，不需要联网。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

CORPUS = Path(__file__).with_name("data") / "corpus.txt"


# region tokenizer
def load_corpus() -> str:
    return CORPUS.read_text(encoding="utf-8")


class CharTokenizer:
    """字符级分词：词表就是语料中出现过的全部字符，按码位排序。"""

    def __init__(self, text: str) -> None:
        self.itos = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def train_val_split(ids: torch.Tensor, val_fraction: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """按位置切分：前 90% 训练，后 10% 验证。随机切窗口会让相邻窗口同时落在两边而泄漏。"""
    cut = int(len(ids) * (1 - val_fraction))
    return ids[:cut], ids[cut:]
# endregion tokenizer


# region bigram_counts
def bigram_counts(ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """C[a, b] = 字符 a 后面紧跟字符 b 的次数。"""
    counts = torch.zeros(vocab_size, vocab_size, dtype=torch.float64)
    counts.index_put_((ids[:-1], ids[1:]), torch.ones(len(ids) - 1, dtype=torch.float64), accumulate=True)
    return counts


def bigram_probs(counts: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """P(b | a) = (C[a,b] + α) / Σ_b (C[a,b] + α)。α>0 是加法平滑，避免没见过的组合概率为 0。"""
    smoothed = counts + alpha
    return smoothed / smoothed.sum(dim=1, keepdim=True)


def bigram_nll(probs: torch.Tensor, ids: torch.Tensor) -> float:
    """平均负对数似然 −(1/N) Σ log P(x_{t+1} | x_t)，单位 nat/字符。"""
    return -torch.log(probs[ids[:-1], ids[1:]]).mean().item()
# endregion bigram_counts


# region bigram_model
class BigramLM(nn.Module):
    """可训练的 bigram：一张 V×V 的表，第 a 行就是“上一个字符是 a”时的 logits。

    nn.Embedding 查第 a 行，等价于 one_hot(a) @ W——没有隐藏层的单层网络。
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.logits = nn.Embedding(vocab_size, vocab_size)
        nn.init.zeros_(self.logits.weight)  # 从均匀分布出发：初始损失恰为 log V

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.logits(context[:, -1])  # 只看最后一个字符
# endregion bigram_model


# region mlp_model
class MLPLM(nn.Module):
    """Bengio et al. (2003) 的神经概率语言模型（不含直连项）。

    取前 n 个字符的嵌入 C(x_{t-n}), …, C(x_{t-1})，拼成 n·m 维向量，
    经 tanh 隐藏层得到 h，再线性映射到 V 个 logits：
        h = tanh(W_h [e_1; …; e_n] + b_h),   logits = W_o h + b_o
    """

    def __init__(self, vocab_size: int, context: int, d_embed: int = 16, d_hidden: int = 128) -> None:
        super().__init__()
        self.context = context
        self.embed = nn.Embedding(vocab_size, d_embed)
        self.hidden = nn.Linear(context * d_embed, d_hidden)
        self.out = nn.Linear(d_hidden, vocab_size)
        with torch.no_grad():  # 让初始 logits 接近 0，初始损失≈log V（见“初始损失”一节）
            self.out.weight.mul_(0.1)
            self.out.bias.zero_()

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        e = self.embed(context)                  # [B, n, m]
        h = torch.tanh(self.hidden(e.flatten(1)))  # 拼接：[B, n·m] -> [B, d_hidden]
        return self.out(h)                       # [B, V]
# endregion mlp_model


# region windows
def context_windows(ids: torch.Tensor, context: int, pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """为每个位置 t 构造 (前 context 个字符, 第 t 个字符)。开头不足的部分用 pad_id 填充。"""
    padded = torch.cat([torch.full((context,), pad_id, dtype=ids.dtype), ids])
    X = padded.unfold(0, context, 1)[: len(ids)]  # unfold 返回滑动窗口视图，不复制
    return X, ids
# endregion windows


# region train
@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, Y: torch.Tensor) -> float:
    model.eval()
    loss = F.cross_entropy(model(X), Y).item()
    model.train()
    return loss


def train_lm(model: nn.Module, train: tuple[torch.Tensor, torch.Tensor],
             val: tuple[torch.Tensor, torch.Tensor], steps: int = 2000, batch_size: int = 128,
             lr: float = 3e-3, weight_decay: float = 0.0, seed: int = 0,
             eval_every: int = 250) -> list[tuple[int, float, float]]:
    """小批量 AdamW 训练，返回 [(step, train_loss, val_loss), …]（损失均在完整数据集上评估）。"""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    X, Y = train
    history = []
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            history.append((step, evaluate(model, X, Y), evaluate(model, *val)))
        if step == steps:
            break
        idx = torch.randint(len(X), (batch_size,), generator=g)
        loss = F.cross_entropy(model(X[idx]), Y[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return history
# endregion train


# region sample
@torch.no_grad()
def sample(model: nn.Module, tokenizer: CharTokenizer, prompt: str, length: int, context: int,
           temperature: float = 1.0, generator: torch.Generator | None = None) -> str:
    """自回归采样：每一步只把最近 context 个字符喂给模型，从 softmax(logits/T) 中抽下一个字符。"""
    model.eval()
    pad = tokenizer.stoi[" "]
    ids = [pad] * context + tokenizer.encode(prompt).tolist()
    for _ in range(length):
        x = torch.tensor([ids[-context:]])
        probs = F.softmax(model(x)[0] / temperature, dim=-1)
        ids.append(int(torch.multinomial(probs, 1, generator=generator)))
    return tokenizer.decode(ids[context:])
# endregion sample
