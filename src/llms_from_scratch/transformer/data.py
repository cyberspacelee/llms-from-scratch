"""训练数据：离线可复现的小故事语料、语料编码与随机窗口采样。

``synthetic_stories`` 用模板与随机词表生成 TinyStories 风格的英文短故事。它完全离线、
秒级生成，又带有真实的长程依赖（主角名字、代词与物品要在后文保持一致），
适合在 CPU 上几分钟内训练出能写出通顺句子的小模型。真实实验可换成 TinyStories 文本文件。
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch

EOT = "<|endoftext|>"

_GIRLS = ["Lily", "Mia", "Anna", "Sue", "Emma", "Lucy", "Zoe", "Rose", "Amy", "Kim"]
_BOYS = ["Tom", "Ben", "Max", "Sam", "Leo", "Jack", "Tim", "Joe", "Dan", "Bob"]
_ANIMALS = ["cat", "dog", "bird", "bunny", "frog", "duck", "fox", "bear", "owl", "mouse"]
_PLACES = ["park", "garden", "forest", "beach", "farm", "hill", "lake", "field"]
_OBJECTS = ["ball", "kite", "box", "hat", "book", "cup", "toy car", "balloon", "drum", "shell"]
_COLORS = ["red", "blue", "green", "yellow", "pink", "big", "small", "shiny", "soft", "little"]
_HIDES = ["under a tree", "behind a big rock", "in the tall grass", "next to the pond",
          "under a bush", "on top of a log"]
_FOODS = ["apples", "cookies", "carrots", "berries", "bread", "cake"]
_FEARS = ["the dark", "the loud thunder", "the big waves", "the tall slide", "the deep water"]


def _story(rng: random.Random) -> str:
    """生成一个故事。主角、配角、物品在全文中保持一致——模型必须“回看”前文才能写对。"""
    girl = rng.random() < 0.5
    name = rng.choice(_GIRLS if girl else _BOYS)
    he, his = ("she", "her") if girl else ("he", "his")
    He = he.capitalize()
    friend = rng.choice([n for n in _GIRLS + _BOYS if n != name])
    animal = rng.choice(_ANIMALS)
    place = rng.choice(_PLACES)
    obj, color = rng.choice(_OBJECTS), rng.choice(_COLORS)
    kind = "girl" if girl else "boy"
    opening = rng.choice([
        f"Once upon a time, there was a little {kind} named {name}.",
        f"One day, a little {kind} named {name} went to the {place}.",
        f"There was a {kind} named {name} who loved to play outside.",
    ])
    plot = rng.randrange(4)
    if plot == 0:  # 丢失与寻找
        body = [
            f"{name} had a {color} {obj}. {He} loved {his} {obj} very much.",
            f"{name} took the {obj} to the {place} to play.",
            f"When the sun went down, {name} could not find {his} {obj}. {He} was very sad.",
            f"Then a {animal} came by. \"Why are you sad?\" asked the {animal}.",
            f"\"I lost my {color} {obj},\" said {name}.",
            f"\"I will help you,\" said the {animal}. They looked everywhere.",
            f"At last, the {animal} found the {obj} {rng.choice(_HIDES)}.",
            f"{name} was so happy. \"Thank you!\" {he} said.",
            f"From that day on, {name} and the {animal} were best friends.",
        ]
    elif plot == 1:  # 分享
        food = rng.choice(_FOODS)
        body = [
            f"{name} had a basket of {food}. {He} wanted to eat them all.",
            f"{He} met {friend} at the {place}. {friend} looked hungry.",
            f"{name} thought for a moment. Then {he} gave some {food} to {friend}.",
            f"{friend} smiled and said, \"Thank you, {name}!\"",
            f"They sat together and ate the {food}. The {food} were very good.",
            f"{name} learned that sharing makes everyone happy.",
        ]
    elif plot == 2:  # 克服恐惧
        fear = rng.choice(_FEARS)
        body = [
            f"{name} was scared of {fear}.",
            f"One day, {name} and {friend} went to the {place}.",
            f"{friend} said, \"Do not be scared, {name}. I am here with you.\"",
            f"{name} held {friend}'s hand and took a deep breath.",
            f"{He} was brave, and soon {he} was not scared of {fear} anymore.",
            f"{name} and {friend} laughed and played until it was time to go home.",
        ]
    else:  # 礼物
        body = [
            f"It was {friend}'s birthday. {name} wanted to give {friend} a gift.",
            f"{name} went to the {place} and found a {color} {obj}.",
            f"{He} put the {obj} in a box and gave it to {friend}.",
            f"{friend} opened the box and saw the {color} {obj}.",
            f"\"I love it!\" said {friend}. {friend} gave {name} a big hug.",
            f"{name} felt warm and happy inside.",
        ]
    return " ".join([opening, *body])


def synthetic_stories(n: int, seed: int = 0) -> list[str]:
    """生成 n 个确定性的短故事（同一 seed 结果相同）。"""
    rng = random.Random(seed)
    return [_story(rng) for _ in range(n)]


def load_documents(path: str | None = None, n_stories: int = 4000, seed: int = 0) -> list[str]:
    """读取以 <|endoftext|> 分隔的文本文件（如 TinyStories）；不给路径则生成合成故事。"""
    if path is None:
        return synthetic_stories(n_stories, seed)
    with open(path, encoding="utf-8") as f:
        return [d.strip() for d in f.read().split(EOT) if d.strip()]


def encode_documents(tokenizer, docs: Sequence[str]) -> np.ndarray:
    """把文档编码后首尾相接成一条长 token 流，文档之间插入 <|endoftext|>。"""
    eot = tokenizer.encode(EOT)
    ids: list[int] = []
    for doc in docs:
        ids.extend(tokenizer.encode(doc))
        ids.extend(eot)
    dtype = np.uint16 if len(tokenizer) <= 65536 else np.int32  # 词表 < 65536 时省一半内存
    return np.asarray(ids, dtype=dtype)


# region get_batch
def get_batch(
    tokens: np.ndarray, batch_size: int, context_length: int,
    device: str | torch.device = "cpu", generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从 token 流中随机取 batch_size 个长度为 context_length 的窗口。

    返回输入 x 与右移一位的目标 y，形状都是 [B, T]：y[b, t] = tokens[s_b + t + 1]。
    ``tokens`` 可以是 np.memmap，只有被采到的窗口才会从磁盘读入。
    """
    starts = torch.randint(0, len(tokens) - context_length, (batch_size,), generator=generator)
    x = np.stack([tokens[s : s + context_length] for s in starts.tolist()])
    y = np.stack([tokens[s + 1 : s + 1 + context_length] for s in starts.tolist()])
    to = lambda a: torch.from_numpy(a.astype(np.int64)).to(device)  # noqa: E731
    return to(x), to(y)
# endregion
