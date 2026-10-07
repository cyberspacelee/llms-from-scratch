import math
from itertools import combinations

import pytest
import torch
from torch import nn

from llms_from_scratch.pretraining.evaluation import (
    NgramIndex,
    accuracy_stderr,
    bits_per_byte,
    continuation_logprob,
    extract_final_number,
    multiple_choice_accuracy,
    pass_at_k,
    perplexity,
    score_choices,
    sequence_nll,
)
from llms_from_scratch.transformer.model import GPT, GPTConfig


class Bigram(nn.Module):
    """logits[t] 只依赖当前 token 的查表模型，便于手算对数似然。"""

    def __init__(self, table: torch.Tensor) -> None:
        super().__init__()
        self.table = table

    def forward(self, idx):
        return self.table[idx]


def encode(text: str) -> list[int]:
    return list(text.encode())


def test_bpb_and_perplexity_relation():
    nll, tokens, nbytes = 120.0, 40, 160
    assert perplexity(nll, tokens) == pytest.approx(math.exp(3.0))
    # bpb = (token 数 / 字节数) · log2(PPL)
    assert bits_per_byte(nll, nbytes) == pytest.approx(tokens / nbytes * math.log2(math.exp(3.0)))


def test_sequence_nll_and_continuation_logprob_on_gpt():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=256, context_length=32, d_model=32, n_layers=1, n_heads=2))
    ctx, cont = encode("abc"), encode("de")
    lp = continuation_logprob(model, ctx, cont)
    with torch.no_grad():
        logp = model(torch.tensor([ctx + cont[:-1]]))[0].log_softmax(-1)
    manual = logp[2, cont[0]] + logp[3, cont[1]]
    assert lp == pytest.approx(manual.item(), abs=1e-5)
    nll, n = sequence_nll(model, torch.tensor(ctx + cont))
    assert n == 4 and nll > 0


def test_length_normalization_changes_the_decision():
    table = torch.full((256, 256), -20.0)
    table[:, ord("a")] = 1.0                 # log P(a) ≈ -0.31
    table[:, ord("b")] = 0.0                 # log P(b) ≈ -1.31
    model = Bigram(table)
    choices = ["b", "aaaaaaaa"]              # 短选项 1 个 token；长选项 8 个高概率 token
    raw = score_choices(model, encode, "q:", choices)
    norm = score_choices(model, encode, "q:", choices, normalize="byte")
    assert raw[0] > raw[1]                   # 原始对数似然偏向短选项
    assert norm[1] > norm[0]                 # 按字节归一化后偏向长选项
    item = {"context": "q:", "choices": choices, "answer": 1}
    assert multiple_choice_accuracy(model, encode, [item], normalize="byte") == 1.0
    assert multiple_choice_accuracy(model, encode, [item]) == 0.0


def test_pass_at_k_is_unbiased():
    n, c, k = 10, 3, 4
    # 枚举所有大小为 k 的子集：至少含一个正确样本的比例
    samples = [True] * c + [False] * (n - c)
    exact = sum(any(s) for s in combinations(samples, k)) / math.comb(n, k)
    assert pass_at_k(n, c, k) == pytest.approx(exact)
    assert pass_at_k(n, c, 1) == pytest.approx(c / n)
    assert pass_at_k(5, 0, 2) == 0.0 and pass_at_k(5, 4, 2) == 1.0


def test_extract_final_number_and_stderr():
    assert extract_final_number("So she pays 3 * 4 = 12 dollars.\n#### 1,200") == "1200"
    assert extract_final_number("The answer is -7.5") == "-7.5"
    assert extract_final_number("no digits") is None
    assert accuracy_stderr(0.5, 100) == pytest.approx(0.05)


def test_ngram_contamination():
    index = NgramIndex(n=5)
    index.add("Janet's ducks lay 16 eggs per day and she eats three for breakfast every morning")
    leaked = "Question: Janet's ducks lay 16 eggs per day. How many are left?"
    clean = "A train leaves the station at noon traveling sixty miles per hour."
    assert index.is_contaminated(leaked) and not index.is_contaminated(clean)
    assert 0 < index.overlap(leaked) < 1
