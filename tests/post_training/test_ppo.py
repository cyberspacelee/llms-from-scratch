import torch

from llms_from_scratch.post_training.ppo import (
    compute_gae,
    kl_penalized_rewards,
    ppo_policy_loss,
    ppo_value_loss,
    reinforce_gradient_stats,
    whiten,
)


def test_gae_matches_hand_computation():
    # 三步回复，γ = 1，λ = 0.5；奖励只在最后一步
    r = torch.tensor([[0.0, 0.0, 1.0]])
    v = torch.tensor([[0.5, 0.6, 0.8]])
    m = torch.ones(1, 3, dtype=torch.bool)
    adv, ret = compute_gae(r, v, m, gamma=1.0, lam=0.5)
    d0, d1, d2 = 0 + 0.6 - 0.5, 0 + 0.8 - 0.6, 1.0 + 0 - 0.8  # TD 残差
    a2 = d2
    a1 = d1 + 0.5 * a2
    a0 = d0 + 0.5 * a1
    torch.testing.assert_close(adv, torch.tensor([[a0, a1, a2]]))
    torch.testing.assert_close(ret, adv + v)


def test_gae_limits():
    torch.manual_seed(0)
    r, v = torch.randn(2, 5), torch.randn(2, 5)
    m = torch.ones(2, 5, dtype=torch.bool)
    # λ = 0：单步 TD 残差
    adv0, _ = compute_gae(r, v, m, gamma=0.9, lam=0.0)
    next_v = torch.cat([v[:, 1:], torch.zeros(2, 1)], 1)
    torch.testing.assert_close(adv0, r + 0.9 * next_v - v)
    # λ = 1：蒙特卡洛回报减基线
    adv1, _ = compute_gae(r, v, m, gamma=0.9, lam=1.0)
    mc = torch.zeros_like(r)
    acc = torch.zeros(2)
    for t in reversed(range(5)):
        acc = r[:, t] + 0.9 * acc
        mc[:, t] = acc
    torch.testing.assert_close(adv1, mc - v)


def test_gae_respects_prompt_and_padding_mask():
    r = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
    v = torch.tensor([[9.0, 0.2, 0.4, 0.6, 9.0]])
    m = torch.tensor([[False, True, True, True, False]])
    adv, _ = compute_gae(r, v, m, gamma=1.0, lam=1.0)
    assert adv[0, 0] == 0 and adv[0, 4] == 0
    torch.testing.assert_close(adv[0, 1:4], torch.tensor([0.8, 0.6, 0.4]))


def test_kl_rewards_placement():
    score = torch.tensor([2.0])
    logp = torch.tensor([[0.0, -1.0, -2.0, 0.0]])
    ref = torch.tensor([[0.0, -1.5, -1.0, 0.0]])
    mask = torch.tensor([[False, True, True, False]])
    r = kl_penalized_rewards(score, logp, ref, mask, beta=0.1)
    torch.testing.assert_close(r, torch.tensor([[0.0, -0.05, 0.1 + 2.0, 0.0]]))


def test_clip_removes_gradient_outside_trust_region():
    old = torch.zeros(1, 4)
    logp = torch.log(torch.tensor([[1.5, 0.5, 1.1, 0.7]])).requires_grad_()
    adv = torch.tensor([[1.0, -1.0, 1.0, 1.0]])
    m = torch.ones(1, 4, dtype=torch.bool)
    ppo_policy_loss(logp, old, adv, m, clip_eps=0.2).backward()
    g = logp.grad[0]
    assert g[0] == 0  # ρ = 1.5 > 1.2 且 Â > 0：已经涨够了，不再推
    assert g[1] == 0  # ρ = 0.5 < 0.8 且 Â < 0：已经降够了
    assert g[2] < 0  # 区间内正常推高（损失梯度为负 → 对数概率上升）
    assert g[3] < 0  # ρ = 0.7 < 0.8 但 Â > 0：min 取未裁剪项，允许修正


def test_value_loss_is_pessimistic():
    v_old = torch.tensor([[0.0]])
    v_new = torch.tensor([[1.0]])
    ret = torch.tensor([[1.0]])
    loss = ppo_value_loss(v_new, v_old, ret, torch.ones(1, 1, dtype=torch.bool), clip_eps=0.2)
    torch.testing.assert_close(loss, torch.tensor(0.5 * 0.8**2))


def test_baseline_keeps_mean_and_reduces_variance():
    logits = torch.tensor([1.0, 0.0, -1.0, 0.5], dtype=torch.float64)
    rewards = torch.tensor([5.0, 4.0, 6.0, 5.5], dtype=torch.float64)
    mean0, var0 = reinforce_gradient_stats(logits, rewards, baseline=0.0)
    probs = logits.softmax(-1)
    expected_r = (probs * rewards).sum()
    mean_b, var_b = reinforce_gradient_stats(logits, rewards, baseline=expected_r.item())
    # 真实梯度 ∇E[r] = π ⊙ (r − E[r])
    torch.testing.assert_close(mean0, probs * (rewards - expected_r))
    torch.testing.assert_close(mean_b, mean0)
    assert var_b < 0.1 * var0


def test_whiten():
    x = torch.tensor([[1.0, 2.0, 3.0, 100.0]])
    m = torch.tensor([[True, True, True, False]])
    w = whiten(x, m)
    assert w[0, 3] == 0
    torch.testing.assert_close(w[0, :3].mean(), torch.tensor(0.0), atol=1e-6, rtol=0)
