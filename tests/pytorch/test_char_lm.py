import math

import torch
import torch.nn.functional as F

from llms_from_scratch.pytorch.char_lm import (
    MLPLM,
    BigramLM,
    CharTokenizer,
    bigram_counts,
    bigram_nll,
    bigram_probs,
    context_windows,
    load_corpus,
    sample,
    train_lm,
    train_val_split,
)


def _data():
    text = load_corpus()
    tok = CharTokenizer(text)
    return text, tok, tok.encode(text)


def test_tokenizer_roundtrip():
    text, tok, ids = _data()
    assert len(text) > 10_000
    assert tok.decode(ids) == text
    assert tok.vocab_size == len(set(text))


def test_bigram_counts_and_mle():
    ids = torch.tensor([0, 1, 0, 1, 1])
    C = bigram_counts(ids, 2)
    assert C.tolist() == [[0, 2], [1, 1]]
    P = bigram_probs(C, alpha=0.0)
    assert P.tolist() == [[0, 1], [0.5, 0.5]]
    assert math.isclose(bigram_nll(P, ids), -(0 + 0 + math.log(0.5) * 2) / 4)


def test_trained_bigram_approaches_count_model():
    _, tok, ids = _data()
    tr, va = train_val_split(ids)
    pad = tok.stoi[" "]
    X, Y = context_windows(tr, 1, pad)
    counts = bigram_counts(torch.cat([torch.tensor([pad]), tr]), tok.vocab_size)
    best = bigram_nll(bigram_probs(counts, alpha=0.0), torch.cat([torch.tensor([pad]), tr]))
    torch.manual_seed(0)
    model = BigramLM(tok.vocab_size)
    assert math.isclose(F.cross_entropy(model(X), Y).item(), math.log(tok.vocab_size), rel_tol=1e-6)
    opt = torch.optim.Adam(model.parameters(), lr=0.5)
    for _ in range(300):  # 全批量训练：最小化的正是计数模型最大化的似然
        opt.zero_grad()
        loss = F.cross_entropy(model(X), Y)
        loss.backward()
        opt.step()
    assert loss.item() >= best - 1e-6
    assert loss.item() < best + 0.02


def test_context_windows():
    X, Y = context_windows(torch.tensor([5, 6, 7]), 2, pad_id=0)
    assert X.tolist() == [[0, 0], [0, 5], [5, 6]] and Y.tolist() == [5, 6, 7]


def test_mlp_beats_bigram_and_samples():
    _, tok, ids = _data()
    tr, va = train_val_split(ids)
    pad = tok.stoi[" "]
    torch.manual_seed(0)
    model = MLPLM(tok.vocab_size, context=3, d_embed=8, d_hidden=32)
    hist = train_lm(model, context_windows(tr, 3, pad), context_windows(va, 3, pad),
                    steps=600, batch_size=128, lr=1e-2, eval_every=600)
    bigram_val = bigram_nll(bigram_probs(bigram_counts(tr, tok.vocab_size), 0.1), va)
    assert hist[0][2] > 3.9  # 初始损失约 log V
    assert hist[-1][2] < bigram_val  # 3 个字符的上下文胜过 1 个
    text = sample(model, tok, "The ", 40, context=3, generator=torch.Generator().manual_seed(0))
    assert text.startswith("The ") and len(text) == 44
