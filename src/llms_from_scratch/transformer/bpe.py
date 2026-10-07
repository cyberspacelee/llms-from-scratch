"""字节级 BPE 分词器：训练、编码与解码。

与 CS336 作业一的约定一致：
- 基础词表是 256 个字节；特殊 token 紧随其后（ID 256, 257, ...），之后是按顺序学到的合并；
- 训练前先按特殊 token 切开文档，再用 GPT-2 风格的正则预分词，合并不跨越预分词边界；
- 频次并列时选字节序更大的那一对（Python 中 ``max`` 对 ``(bytes, bytes)`` 元组的顺序）。

只依赖标准库。GPT-2 的正则用到 ``\\p{L}``、``\\p{N}``，标准库 ``re`` 不支持，
这里用 ``[^\\W\\d_]``（字母）与 ``\\d``（十进制数字）近似。
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from itertools import pairwise

# region pattern
# GPT-2 / tiktoken 的预分词正则（标准库 re 的近似写法）：
#   's 't 'll 've 're 等英文缩写 | 可带一个前导空格的字母串 | 数字串 | 标点串
#   | 不跟非空白字符的空白（行尾空白） | 其余空白
GPT2_SPLIT_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
)
# endregion


def pretokenize(text: str) -> list[str]:
    """把一段（不含特殊 token 的）文本切成预分词块。块拼接起来恰好是原文。"""
    return GPT2_SPLIT_PATTERN.findall(text)


def split_on_special(text: str, special_tokens: Iterable[str]) -> list[str]:
    """按特殊 token 切分文本，并保留特殊 token 本身作为独立片段。

    较长的特殊 token 优先匹配，这样 ``<|eot|><|eot|>`` 这类重叠的定义也能正确识别。
    """
    specials = sorted(special_tokens, key=len, reverse=True)
    if not specials:
        return [text] if text else []
    pattern = "(" + "|".join(re.escape(s) for s in specials) + ")"
    return [part for part in re.split(pattern, text) if part]


# region count
def count_pretokens(text: str, special_tokens: Iterable[str] = ()) -> Counter[tuple[bytes, ...]]:
    """统计每个预分词块（表示为单字节 bytes 的元组）出现的次数。

    特殊 token 是硬边界：它们两侧的文本分别预分词，特殊 token 自身不参与计数。
    """
    specials = set(special_tokens)
    counts: Counter[tuple[bytes, ...]] = Counter()
    for segment in split_on_special(text, specials):
        if segment in specials:
            continue
        for piece in pretokenize(segment):
            counts[tuple(bytes([b]) for b in piece.encode("utf-8"))] += 1
    return counts
# endregion


# region train
def train_bpe_from_counts(
    word_counts: dict[tuple[bytes, ...], int],
    vocab_size: int,
    special_tokens: Iterable[str] = (),
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """在“预分词块 → 次数”的统计上学习合并，返回 (vocab, merges)。

    增量更新：维护每个相邻对的总频次 ``pair_counts`` 与包含它的块 ``where``；
    一次合并只重算受影响的块，而不是重新扫描全部语料。
    """
    specials = list(dict.fromkeys(special_tokens))
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for s in specials:
        vocab[len(vocab)] = s.encode("utf-8")
    num_merges = vocab_size - len(vocab)
    if num_merges < 0:
        raise ValueError("vocab_size 小于 256 + 特殊 token 数")

    words = [list(w) for w in word_counts]  # 每个块当前的切分（bytes 列表）
    freqs = list(word_counts.values())
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    where: defaultdict[tuple[bytes, bytes], set[int]] = defaultdict(set)
    for i, w in enumerate(words):
        for pair in pairwise(w):
            pair_counts[pair] += freqs[i]
            where[pair].add(i)

    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        if not pair_counts:
            break  # 每个块都已合并成一个 token，无对可合
        # 频次最高者；并列时比较 (bytes, bytes)，取字节序更大的一对
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merged = best[0] + best[1]
        merges.append(best)
        vocab[len(vocab)] = merged
        for i in list(where[best]):
            old, f = words[i], freqs[i]
            new = merge_pair(old, best, merged)
            # 先撤销旧切分贡献的所有相邻对，再加上新切分的相邻对
            for pair in pairwise(old):
                pair_counts[pair] -= f
                if pair_counts[pair] == 0:
                    del pair_counts[pair]
                where[pair].discard(i)
            for pair in pairwise(new):
                pair_counts[pair] += f
                where[pair].add(i)
            words[i] = new
        where.pop(best, None)
    return vocab, merges
# endregion


def merge_pair(word: list[bytes], pair: tuple[bytes, bytes], merged: bytes) -> list[bytes]:
    """从左到右把 ``word`` 中不重叠的 ``pair`` 替换为 ``merged``。"""
    out: list[bytes] = []
    i = 0
    while i < len(word):
        if i + 1 < len(word) and word[i] == pair[0] and word[i + 1] == pair[1]:
            out.append(merged)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


def train_bpe(
    text: str, vocab_size: int, special_tokens: Iterable[str] = ()
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """在原始文本上训练字节级 BPE：预分词计数 + 增量合并。"""
    specials = list(special_tokens)
    return train_bpe_from_counts(count_pretokens(text, specials), vocab_size, specials)


class Tokenizer:
    """用 (vocab, merges, special_tokens) 做编码与解码。``decode(encode(s)) == s``。"""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: Iterable[str] = (),
    ) -> None:
        self.vocab = dict(vocab)
        self.special_tokens = list(dict.fromkeys(special_tokens))
        self.byte_to_id = {b: i for i, b in self.vocab.items()}
        for s in self.special_tokens:  # 允许在编码时追加训练时没有的特殊 token
            if s.encode("utf-8") not in self.byte_to_id:
                self.vocab[len(self.vocab)] = s.encode("utf-8")
                self.byte_to_id[s.encode("utf-8")] = len(self.vocab) - 1
        self.ranks = {pair: r for r, pair in enumerate(merges)}  # 合并的优先级 = 学到的先后
        self.merges = list(merges)
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def train(cls, text: str, vocab_size: int, special_tokens: Iterable[str] = ()) -> Tokenizer:
        specials = list(special_tokens)
        vocab, merges = train_bpe(text, vocab_size, specials)
        return cls(vocab, merges, specials)

    def __len__(self) -> int:
        return len(self.vocab)

    # region encode
    def _encode_piece(self, piece: str) -> list[int]:
        """对一个预分词块反复应用“当前最早学到的”合并，直到没有可合并的相邻对。"""
        if piece in self._cache:
            return self._cache[piece]
        parts = [bytes([b]) for b in piece.encode("utf-8")]
        while len(parts) > 1:
            pairs = set(pairwise(parts))
            best = min(pairs, key=lambda p: self.ranks.get(p, float("inf")))
            if best not in self.ranks:
                break
            parts = merge_pair(parts, best, best[0] + best[1])
        ids = [self.byte_to_id[p] for p in parts]
        self._cache[piece] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        specials = set(self.special_tokens)
        for segment in split_on_special(text, specials):
            if segment in specials:
                ids.append(self.byte_to_id[segment.encode("utf-8")])
            else:
                for piece in pretokenize(segment):
                    ids.extend(self._encode_piece(piece))
        return ids
    # endregion

    def encode_iterable(self, lines: Iterable[str]) -> Iterator[int]:
        """逐行编码大文件而不一次读入内存（假设特殊 token 不跨行）。"""
        for line in lines:
            yield from self.encode(line)

    def decode(self, ids: Iterable[int]) -> str:
        """先拼接字节再整体按 UTF-8 解码；非法字节序列替换为 U+FFFD。"""
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
