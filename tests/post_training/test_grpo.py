import torch

from llms_from_scratch.post_training.common import response_logprobs, tiny_gpt
from llms_from_scratch.post_training.grpo import (
    GRPOConfig,
    answer_token_prob,
    group_advantages,
    grpo_loss,
    k3_kl,
    train_grpo,
)
from llms_from_scratch.post_training.tasks import (
    addition_problems,
    addition_reward,
    arithmetic_tokenizer,
)


def test_group_advantages():
    r = torch.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
    adv = group_advantages(r)
    std = torch.tensor([1.0, 0.0, 0.0, 1.0]).std()  # torch.std 默认无偏（除以 G-1）
    torch.testing.assert_close(adv[0], torch.tensor([0.5, -0.5, -0.5, 0.5]) / (std + 1e-6))
    assert torch.all(adv[1] == 0)  # 全对：没有学习信号
    torch.testing.assert_close(adv.sum(1), torch.zeros(3), atol=1e-6, rtol=0)
    torch.testing.assert_close(group_advantages(r, normalize_std=False)[2],
                               torch.tensor([0.75, -0.25, -0.25, -0.25]))


def test_k3_is_unbiased_nonnegative_and_its_gradient_is_forward_kl():
    torch.manual_seed(0)
    logits = torch.randn(6, dtype=torch.float64, requires_grad=True)
    ref_logits = torch.randn(6, dtype=torch.float64)
    logp, ref_logp = logits.log_softmax(0), ref_logits.log_softmax(0)
    p = logp.exp()
    k3 = k3_kl(logp, ref_logp)
    assert torch.all(k3 >= 0)
    exact = (p * (logp - ref_logp)).sum()  # KL(π_θ ‖ π_ref)
    expectation = (p.detach() * k3).sum()  # E_{y~π_θ}[k3(y)]，采样分布不求导
    torch.testing.assert_close(expectation, exact)
    # 把 k3 当作损失、样本固定时，梯度的期望是 ∇ KL(π_ref ‖ π_θ)，不是 ∇ KL(π_θ ‖ π_ref)
    (g_k3,) = torch.autograd.grad(expectation, logits, retain_graph=True)
    forward_kl = (ref_logp.exp() * (ref_logp - logp)).sum()
    (g_fwd,) = torch.autograd.grad(forward_kl, logits, retain_graph=True)
    (g_rev,) = torch.autograd.grad(exact, logits)
    torch.testing.assert_close(g_k3, g_fwd)
    assert (g_k3 - g_rev).abs().max() > 1e-3


def test_grpo_loss_gradient_direction():
    """一步更新后，正优势回复的对数概率上升、负优势回复的下降。"""
    torch.set_num_threads(1)
    model = tiny_gpt(15, seed=0, context_length=16)
    seqs = torch.tensor([[3, 4, 5, 6], [3, 4, 7, 8]])
    mask = torch.tensor([[False, True, True], [False, True, True]])
    adv = torch.tensor([1.0, -1.0])
    with torch.no_grad():
        old = response_logprobs(model, seqs)
    opt = torch.optim.SGD(model.parameters(), lr=0.05)
    loss = grpo_loss(response_logprobs(model, seqs), old, old, adv, mask)
    assert abs(loss.item()) < 1e-6  # ρ = 1、KL = 0、优势均值为 0
    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
        new = response_logprobs(model, seqs)
    delta = ((new - old) * mask).sum(1)
    assert delta[0] > 0 > delta[1]


def test_clipping_stops_gradient_after_large_ratio():
    old = torch.zeros(1, 2)
    logp = torch.tensor([[0.5, 0.0]], requires_grad=True)  # ρ = e^0.5 ≈ 1.65 与 1
    grpo_loss(logp, old, logp.detach(), torch.tensor([1.0]), torch.ones(1, 2), beta=0.0).backward()
    assert logp.grad[0, 0] == 0 and logp.grad[0, 1] < 0


def test_train_grpo_improves_policy():
    torch.set_num_threads(1)
    tok = arithmetic_tokenizer()
    problems = addition_problems(4)
    model = tiny_gpt(tok.vocab_size, seed=0, context_length=16)
    before = answer_token_prob(model, tok, problems)
    logs = train_grpo(model, tok, problems, addition_reward,
                      GRPOConfig(steps=8, lr=1e-2, beta=0.01))
    after = answer_token_prob(model, tok, problems)
    assert len(logs) == 8 and all(torch.isfinite(torch.tensor(row["loss"])) for row in logs)
    assert after > 1.5 * before
