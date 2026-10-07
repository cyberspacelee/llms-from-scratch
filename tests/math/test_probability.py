import math

import torch
import torch.nn.functional as F

from llms_from_scratch.math.probability import (
    bigram_mle,
    bits_per_byte,
    cross_entropy,
    entropy,
    fit_bigram_logits,
    kl_divergence,
    perplexity,
    sequence_nll,
    softmax_with_temperature,
)

D = torch.float64


def test_entropy_values():
    assert math.isclose(entropy(torch.tensor([0.5, 0.5], dtype=D)).item(), math.log(2))
    assert entropy(torch.tensor([1.0, 0.0], dtype=D)).item() == 0.0
    uniform = torch.full((8,), 1 / 8, dtype=D)
    assert math.isclose(entropy(uniform).item() / math.log(2), 3.0)


def test_gibbs_inequality_and_kl():
    torch.manual_seed(0)
    for _ in range(20):
        p = torch.softmax(torch.randn(6, dtype=D), -1)
        q = torch.softmax(torch.randn(6, dtype=D), -1)
        assert cross_entropy(p, q) >= entropy(p)
        assert kl_divergence(p, q) >= 0
        torch.testing.assert_close(
            kl_divergence(p, q), F.kl_div(q.log(), p, reduction="sum")
        )
    assert abs(kl_divergence(p, p).item()) < 1e-12


def test_bigram_mle_and_chain_rule():
    tokens = [0, 1, 2, 0, 1, 0, 2, 2, 1, 0]
    probs = bigram_mle(tokens, 3)
    torch.testing.assert_close(probs.sum(-1), torch.ones(3, dtype=D))
    # 0 后面出现过 1, 1, 2 → P(1|0) = 2/3
    assert math.isclose(probs[0, 1].item(), 2 / 3)
    nll = sequence_nll(tokens, probs)
    log_prob = sum(math.log(probs[a, b]) for a, b in zip(tokens, tokens[1:]))
    assert math.isclose(nll, -log_prob)


def test_mle_is_minimum_cross_entropy():
    """梯度下降最小化交叉熵，得到的正是计数给出的最大似然估计。"""
    tokens = [0, 1, 2, 0, 1, 0, 2, 2, 1, 0, 0, 1, 1, 2, 0]
    target = bigram_mle(tokens, 3)
    learned = fit_bigram_logits(tokens, 3, steps=3000, lr=5.0)
    torch.testing.assert_close(learned, target, atol=2e-3, rtol=0)
    # 任何其他条件分布的交叉熵都更大
    other = bigram_mle(tokens, 3, smoothing=1.0)
    assert sequence_nll(tokens, target) < sequence_nll(tokens, other)


def test_perplexity_and_bpb():
    assert math.isclose(perplexity(math.log(50)), 50)
    # 每个字节 1 bit：总 nll = n ln 2
    assert math.isclose(bits_per_byte(1000 * math.log(2), 1000), 1.0)


def test_temperature_limits():
    z = torch.tensor([2.0, 1.0, 0.0], dtype=D)
    torch.testing.assert_close(softmax_with_temperature(z, 1.0), torch.softmax(z, -1))
    cold = softmax_with_temperature(z, 0.01)
    hot = softmax_with_temperature(z, 1e4)
    assert cold[0] > 0.999
    torch.testing.assert_close(hot, torch.full((3,), 1 / 3, dtype=D), atol=1e-3, rtol=0)
    assert entropy(softmax_with_temperature(z, 0.5)) < entropy(softmax_with_temperature(z, 2.0))
