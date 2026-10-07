"""玩具可验证任务与字符级分词器。

后训练的每个机制（SFT 掩码、偏好损失、GRPO、蒸馏）都需要一个能在 CPU 上几秒内跑完、
答案可以被程序自动判定的任务。本模块提供：

- ``CharTokenizer``：字符级分词器，外加若干整体编码为单个 id 的特殊 token；
- ``addition_problems``：个位数加法 ``"3+4="`` → ``"7"``（和小于 10 时答案只有一个字符）；
- ``reverse_problems``：反转字符串 ``"abc"`` → ``"cba"``；
- ``parse_answer`` / ``addition_reward``：从模型回复中抽取答案并给出 0/1 奖励。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

PAD, BOS, EOS = "<|pad|>", "<|bos|>", "<|endoftext|>"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"


class CharTokenizer:
    """每个普通字符一个 id；特殊 token 整体匹配，永远不会被拆成字符。"""

    def __init__(self, chars: str, specials: tuple[str, ...] = (PAD, BOS, EOS)) -> None:
        self.specials = list(specials)
        self.itos = self.specials + sorted(set(chars))
        self.stoi = {s: i for i, s in enumerate(self.itos)}
        # 特殊 token 按长度降序匹配，避免前缀冲突
        pattern = "|".join(re.escape(s) for s in sorted(self.specials, key=len, reverse=True))
        self._split = re.compile(f"({pattern})")

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for piece in self._split.split(text):
            if piece in self.stoi and piece in self.specials:
                ids.append(self.stoi[piece])
            else:
                ids.extend(self.stoi[ch] for ch in piece)
        return ids

    def decode(self, ids: list[int], skip_special: bool = False) -> str:
        out = [self.itos[i] for i in ids]
        if skip_special:
            out = [s for s in out if s not in self.specials]
        return "".join(out)


def arithmetic_tokenizer() -> CharTokenizer:
    """GRPO 等强化学习实验用的小词表：数字、运算符与三个特殊 token，共 15 个 id。"""
    return CharTokenizer("0123456789+=")


def chat_tokenizer() -> CharTokenizer:
    """SFT 与对话模板用的词表：小写字母、数字、标点，以及 ChatML 式角色标记。"""
    chars = "abcdefghijklmnopqrstuvwxyz0123456789+-=?.,:!' \n"
    return CharTokenizer(chars, specials=(PAD, BOS, EOS, IM_START, IM_END))


@dataclass(frozen=True)
class Problem:
    prompt: str
    answer: str


def addition_problems(max_digit: int = 4) -> list[Problem]:
    """所有 a+b（0 ≤ a, b ≤ max_digit）。max_digit ≤ 4 时和是一位数。"""
    return [
        Problem(f"{a}+{b}=", str(a + b))
        for a in range(max_digit + 1)
        for b in range(max_digit + 1)
    ]


def reverse_problems(n: int, length: int = 4, alphabet: str = "abcd", seed: int = 0) -> list[Problem]:
    """随机字符串反转任务。"""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = "".join(rng.choice(alphabet) for _ in range(length))
        out.append(Problem(s, s[::-1]))
    return out


def parse_answer(response: str) -> str | None:
    """取回复开头、遇到第一个非数字字符（包括 EOS）之前的数字串；没有数字则返回 None。"""
    match = re.match(r"\d+", response)
    return match.group(0) if match else None


def addition_reward(response: str, answer: str) -> float:
    """可验证奖励：抽取的答案与标准答案完全相同得 1，否则得 0。"""
    return float(parse_answer(response) == answer)
