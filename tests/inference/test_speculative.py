import pytest
import torch
import torch.nn.functional as F

from llms_from_scratch.inference.speculative import (
    accept_or_resample,
    acceptance_rate,
    expected_speedup,
    expected_tokens_per_step,
    speculative_generate,
    tree_attention_mask,
    tree_depths,
    verify,
)
from llms_from_scratch.transformer.model import GPT, GPTConfig


def test_accept_reject_preserves_target_distribution():
    """大量独立试验：草稿 x ~ q，经接受-拒绝后得到的 token 的经验分布应等于 p。"""
    g = torch.Generator().manual_seed(0)
    p = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05])
    q = torch.tensor([0.1, 0.1, 0.3, 0.3, 0.2])  # 与 p 差别很大的草稿分布
    n = 200_000
    x = torch.multinomial(q, n, replacement=True, generator=g)
    accepted, tokens = accept_or_resample(p.expand(n, -1), q.expand(n, -1), x, g)
    empirical = torch.bincount(tokens, minlength=5).float() / n
    assert (empirical - p).abs().max() < 0.005
    # 接受率等于 Σ min(p, q)
    assert accepted.float().mean().item() == pytest.approx(acceptance_rate(p, q).item(),
                                                           abs=0.005)


def test_verify_multi_token_first_position_distribution():
    g = torch.Generator().manual_seed(1)
    p = F.softmax(torch.randn(3, 6, generator=g), -1)
    q = F.softmax(torch.randn(2, 6, generator=g), -1)
    counts = torch.zeros(6)
    for _ in range(4_000):
        drafts = [int(torch.multinomial(q[i], 1, generator=g)) for i in range(2)]
        out = verify(p, q, drafts, g)
        assert 1 <= len(out) <= 3
        counts[out[0]] += 1
    assert (counts / counts.sum() - p[0]).abs().max() < 0.03


def test_expected_tokens_formula_matches_simulation():
    g = torch.Generator().manual_seed(2)
    alpha, k, rounds = 0.7, 4, 100_000
    accept = torch.rand(rounds, k, generator=g) < alpha
    # 每轮产出 = 第一次拒绝前接受的个数 + 1（修正 token 或奖励 token）
    first_reject = torch.where(accept.all(1), torch.tensor(k), (~accept).float().argmax(1))
    assert (first_reject + 1).float().mean().item() == pytest.approx(
        expected_tokens_per_step(alpha, k), rel=0.01)
    assert expected_tokens_per_step(1.0, k) == k + 1
    assert expected_speedup(0.8, 4, 0.05) > 2.5 > expected_speedup(0.5, 4, 0.05)


def test_tree_mask_and_depths():
    # 根之后两个候选 a(0)、b(1)；a 下面 c(2)、d(3)；c 下面 e(4)
    parents = [-1, -1, 0, 0, 2]
    mask = tree_attention_mask(parents)
    assert mask[4].nonzero().flatten().tolist() == [0, 2, 4]
    assert mask[3].nonzero().flatten().tolist() == [0, 3]
    assert mask[1].nonzero().flatten().tolist() == [1]
    assert tree_depths(parents) == [0, 0, 1, 1, 2]
    # 一条链的树掩码就是普通的因果掩码
    assert torch.equal(tree_attention_mask([-1, 0, 1, 2]), torch.ones(4, 4, dtype=torch.bool).tril())


def test_tree_attention_equals_per_path_attention():
    torch.manual_seed(0)
    parents = [-1, -1, 0, 0, 2]
    prefix, n, d = 3, len(parents), 8
    q = torch.randn(n, d)
    k, v = torch.randn(prefix + n, d), torch.randn(prefix + n, d)
    full_mask = torch.cat([torch.ones(n, prefix, dtype=torch.bool), tree_attention_mask(parents)], 1)
    tree_out = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=full_mask)[0]
    for i in range(n):
        path, j = [], i
        while j != -1:
            path.insert(0, prefix + j)
            j = parents[j]
        keys = list(range(prefix)) + path
        ref = F.scaled_dot_product_attention(q[i:i + 1][None], k[keys][None], v[keys][None])[0]
        torch.testing.assert_close(tree_out[i:i + 1], ref, atol=1e-6, rtol=1e-5)


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    cfg = GPTConfig(vocab_size=64, context_length=48, d_model=64, n_layers=2, n_heads=4,
                    n_kv_heads=2)
    target = GPT(cfg).eval()
    torch.manual_seed(1)
    draft = GPT(GPTConfig(vocab_size=64, context_length=48, d_model=32, n_layers=1, n_heads=2)).eval()
    return target, draft


@pytest.mark.parametrize("k", [1, 3, 5])
def test_greedy_speculative_equals_plain_greedy(models, k):
    target, draft = models
    for seed in range(3):
        prompt = torch.randint(0, 64, (1, 5 + seed), generator=torch.Generator().manual_seed(seed))
        expected = target.generate(prompt, 20, temperature=0)
        out, stats = speculative_generate(target, draft, prompt, 20, k=k)
        assert torch.equal(out, expected)


def test_self_draft_accepts_everything_and_respects_context(models):
    target, _ = models
    prompt = torch.arange(10)[None]
    out, stats = speculative_generate(target, target, prompt, 20, k=4)
    assert torch.equal(out, target.generate(prompt, 20, temperature=0))
    assert stats.accepted == stats.drafted and stats.tokens_per_round > 4
    long = torch.arange(40)[None] % 64  # 只剩 8 个位置
    out, _ = speculative_generate(target, target, long, 20, k=4)
    assert torch.equal(out, target.generate(long, 20, temperature=0))


def test_sampling_mode_runs_and_is_reproducible(models):
    target, draft = models
    prompt = torch.arange(6)[None]
    a, _ = speculative_generate(target, draft, prompt, 15, k=3, temperature=1.0,
                                generator=torch.Generator().manual_seed(5))
    b, _ = speculative_generate(target, draft, prompt, 15, k=3, temperature=1.0,
                                generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b) and a.shape == (1, 21)
