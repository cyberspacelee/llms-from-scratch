import math

import torch

from llms_from_scratch.post_training.rl_losses import (
    aggregate,
    clipped_surrogate,
    dynamic_sampling_keep,
    effective_token_weights,
    entropy_from_logits,
    gspo_loss,
    overlong_penalty,
    tis_weights,
)

# 两条回复：短的 2 个 token，长的 6 个 token
MASK = torch.tensor([[1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.bool)


def test_three_aggregations_on_hand_example():
    loss = torch.tensor([[1.0, 1.0, 0, 0, 0, 0], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
    assert aggregate(loss, MASK, "seq-mean-token-mean").item() == (1 + 2) / 2
    assert aggregate(loss, MASK, "token-mean").item() == (2 + 12) / 8
    assert aggregate(loss, MASK, "seq-mean-token-sum-norm", max_len=8).item() == (2 + 12) / 16


def test_effective_token_weights():
    w_grpo = effective_token_weights(MASK, "seq-mean-token-mean")
    w_dapo = effective_token_weights(MASK, "token-mean")
    w_dr = effective_token_weights(MASK, "seq-mean-token-sum-norm", max_len=8)
    torch.testing.assert_close(w_grpo[0, 0], torch.tensor(1 / 4))  # 1/(N·|o_1|) = 1/(2·2)
    torch.testing.assert_close(w_grpo[1, 0], torch.tensor(1 / 12))  # 短回复的 token 权重是长回复的 3 倍
    assert torch.all(w_dapo[MASK] == 1 / 8)
    assert torch.all(w_dr[MASK] == 1 / 16)
    for w in (w_grpo, w_dapo, w_dr):
        assert torch.all(w[~MASK] == 0)


def test_length_bias_of_grpo_on_negative_samples():
    """负优势时，GRPO 对长回复每个 token 的惩罚更轻——这就是“错的回复越写越长”的来源。"""
    w = effective_token_weights(MASK, "seq-mean-token-mean")
    per_token_penalty_short = w[0, 0]
    per_token_penalty_long = w[1, 0]
    assert per_token_penalty_long < per_token_penalty_short


def test_clip_higher_allows_rare_tokens_to_grow():
    old = torch.log(torch.tensor([[0.01]]))
    logp = torch.log(torch.tensor([[0.0125]])).requires_grad_()  # ρ = 1.25
    adv = torch.tensor([1.0])
    clipped_surrogate(logp, old, adv, eps_low=0.2, eps_high=0.2).sum().backward()
    assert logp.grad.item() == 0  # 对称裁剪：1.25 > 1.2，梯度消失
    logp.grad = None
    clipped_surrogate(logp, old, adv, eps_low=0.2, eps_high=0.28).sum().backward()
    assert logp.grad.item() < 0  # clip-higher：仍在区间内，继续推高


def test_dynamic_sampling_and_overlong_penalty():
    r = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert dynamic_sampling_keep(r).tolist() == [False, False, True]
    lengths = torch.tensor([10, 16, 18, 20, 24])
    torch.testing.assert_close(overlong_penalty(lengths, max_len=20, cache_len=4),
                               torch.tensor([0.0, 0.0, -0.5, -1.0, -1.0]))


def test_gspo_ratio_is_geometric_mean():
    old = torch.zeros(1, 3)
    logp = torch.tensor([[0.3, -0.1, 0.1]], requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.bool)
    loss = gspo_loss(logp, old, torch.tensor([1.0]), mask, eps_low=1.0, eps_high=1.0)
    torch.testing.assert_close(loss, -torch.tensor(math.exp(0.1)))
    loss.backward()
    # 序列级比值：每个 token 得到相同的梯度 s/|y|
    torch.testing.assert_close(logp.grad, -torch.full((1, 3), math.exp(0.1) / 3))
    logp.grad = None
    gspo_loss(logp, old, torch.tensor([1.0]), mask).backward()  # 默认 ε 很小，s = e^0.1 被裁
    assert torch.all(logp.grad == 0)


def test_tis_weights_are_truncated_and_detached():
    train = torch.tensor([0.0, -1.0, -3.0], requires_grad=True)
    rollout = torch.tensor([0.0, -2.0, -1.0])
    w = tis_weights(train, rollout, cap=2.0)
    torch.testing.assert_close(w, torch.tensor([1.0, 2.0, math.exp(-2.0)]))
    assert not w.requires_grad


def test_entropy():
    h = entropy_from_logits(torch.zeros(2, 4))
    torch.testing.assert_close(h, torch.full((2,), math.log(4)))
    assert entropy_from_logits(torch.tensor([100.0, 0.0, 0.0])).item() < 1e-6
