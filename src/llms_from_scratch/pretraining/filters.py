"""文本过滤：Gopher/C4 式启发式规则、PII 掩码，以及 fastText 式的线性文本分类器。

阈值取自原始论文（Gopher: Rae et al. 2021 附录 A；C4: Raffel et al. 2020 第 2.2 节）。
分类器既可做语言识别，也可做 DCLM / FineWeb-Edu 式的质量打分。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

STOP_WORDS = {"the", "be", "to", "of", "and", "that", "have", "with"}
_BULLETS = ("•", "-", "*", "·", "●", "◦", "‣")


# region gopher
def gopher_quality_filter(text: str) -> tuple[bool, str | None]:
    """Gopher 的质量规则。返回（是否保留，未通过的规则名）。"""
    words = text.split()
    n = len(words)
    if not 50 <= n <= 100_000:
        return False, "word_count"
    mean_len = sum(len(w) for w in words) / n
    if not 3 <= mean_len <= 10:
        return False, "mean_word_length"
    if (text.count("#") + text.count("...") + text.count("…")) / n > 0.1:
        return False, "symbol_ratio"
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and sum(line.startswith(_BULLETS) for line in lines) / len(lines) > 0.9:
        return False, "bullet_lines"
    if lines and sum(line.endswith(("...", "…")) for line in lines) / len(lines) > 0.3:
        return False, "ellipsis_lines"
    if sum(bool(re.search(r"[^\W\d_]", w)) for w in words) / n < 0.8:
        return False, "alphabetic_words"
    if len(STOP_WORDS & {w.lower().strip(".,;:!?\"'") for w in words}) < 2:
        return False, "stop_words"
    return True, None


def gopher_repetition_filter(text: str) -> tuple[bool, str | None]:
    """Gopher 的重复度规则：重复行比例、最高频 n-gram 字符占比、重复 n-gram 字符占比。"""
    lines = [line for line in text.splitlines() if line.strip()]
    if lines:
        counts = Counter(lines)
        if sum(c for c in counts.values() if c > 1) / len(lines) > 0.3:
            return False, "duplicate_lines"
    words = text.split()
    total = sum(len(w) for w in words) or 1
    for n, limit in ((2, 0.20), (3, 0.18), (4, 0.16)):
        grams = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
        if grams:
            gram, count = grams.most_common(1)[0]
            if count > 1 and count * sum(len(w) for w in gram) / total > limit:
                return False, f"top_{n}gram"
    for n, limit in ((5, 0.15), (6, 0.14), (7, 0.13), (8, 0.12), (9, 0.11), (10, 0.10)):
        seen: dict[tuple[str, ...], int] = {}
        covered = [False] * len(words)
        for i in range(len(words) - n + 1):
            gram = tuple(words[i:i + n])
            if gram in seen:  # 第二次及以后出现的 n-gram 所覆盖的词都算“重复”
                for j in range(i, i + n):
                    covered[j] = True
                for j in range(seen[gram], seen[gram] + n):
                    covered[j] = True
            else:
                seen[gram] = i
        dup_chars = sum(len(w) for w, c in zip(words, covered, strict=True) if c)
        if dup_chars / total > limit:
            return False, f"duplicate_{n}gram"
    return True, None
# endregion gopher


# region c4
def c4_filter(text: str) -> str | None:
    """C4 的规则：逐行清洗，再做文档级判断。返回清洗后的文本，被丢弃时返回 None。"""
    lowered = text.lower()
    if "lorem ipsum" in lowered or "{" in text:
        return None
    kept = []
    for line in text.splitlines():
        line = line.strip()
        if not line.endswith((".", "!", "?", '"')):
            continue  # 只保留以终止标点结尾的行
        if len(line.split()) < 5 or "javascript" in line.lower():
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    if len(re.findall(r"[.!?](?:\s|$)", cleaned)) < 3:
        return None  # 少于 3 个句子
    return cleaned
# endregion c4


# region pii
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)")
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    """把邮箱、北美电话号码、IPv4 地址替换成占位符，返回（新文本，各类计数）。"""
    counts = {}
    for name, pattern, token in (
        ("email", _EMAIL, "|||EMAIL_ADDRESS|||"),
        ("phone", _PHONE, "|||PHONE_NUMBER|||"),
        ("ip", _IPV4, "|||IP_ADDRESS|||"),
    ):
        text, counts[name] = pattern.subn(token, text)
    return text, counts
# endregion pii


# region fasttext
def hashed_ngrams(text: str, n_min: int = 1, n_max: int = 3, buckets: int = 1 << 16) -> list[int]:
    """fastText 式特征：字符 n-gram（两端加边界符）哈希到固定数量的桶。"""
    padded = f"<{text.lower()}>"
    ids = []
    for n in range(n_min, n_max + 1):
        for i in range(len(padded) - n + 1):
            # 用确定性的多项式哈希，避免 Python hash() 随进程随机化
            h = 0
            for ch in padded[i:i + n]:
                h = (h * 1_000_003 + ord(ch)) & 0xFFFFFFFF
            ids.append(h % buckets)
    return ids


class FastTextClassifier(nn.Module):
    """平均 n-gram 嵌入 → 线性层 → softmax。这就是 fastText 的全部结构。"""

    def __init__(self, num_classes: int, dim: int = 16, buckets: int = 1 << 16) -> None:
        super().__init__()
        self.buckets = buckets
        self.embed = nn.EmbeddingBag(buckets, dim, mode="mean")
        self.out = nn.Linear(dim, num_classes)
        nn.init.uniform_(self.embed.weight, -1 / dim, 1 / dim)   # 与 fastText 相同的小初始化

    def featurize(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        ids = [hashed_ngrams(t, buckets=self.buckets) for t in texts]
        offsets = torch.tensor([0] + [len(x) for x in ids[:-1]]).cumsum(0)
        return torch.tensor([i for x in ids for i in x]), offsets

    def forward(self, texts: list[str]) -> torch.Tensor:
        return self.out(self.embed(*self.featurize(texts)))

    @torch.no_grad()
    def predict_proba(self, texts: list[str]) -> torch.Tensor:
        return self(texts).softmax(-1)


@dataclass
class TrainResult:
    model: FastTextClassifier
    losses: list[float]


def train_fasttext(texts: list[str], labels: list[int], num_classes: int, epochs: int = 30,
                   lr: float = 0.05, seed: int = 0) -> TrainResult:
    """全批量 Adam 训练（语料很小时足够）；真实 fastText 用逐样本 SGD + 线性衰减学习率。"""
    torch.manual_seed(seed)
    model = FastTextClassifier(num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    y = torch.tensor(labels)
    losses = []
    for _ in range(epochs):
        loss = F.cross_entropy(model(texts), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return TrainResult(model, losses)
# endregion fasttext
