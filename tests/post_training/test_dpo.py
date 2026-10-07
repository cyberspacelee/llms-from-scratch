import copy
import math

import torch

from llms_from_scratch.post_training.common import tiny_gpt
from llms_from_scratch.post_training.dpo import (
    dpo_loss,
    ipo_loss,
    kto_loss,
    model_sequence_logprob,
    sequence_logprob,
    simpo_loss,
)
from llms_from_scratch.post_training.tasks import addition_problems, arithmetic_tokenizer


def test_sequence_logprob_sums_only_response_tokens():
    # 词表 3；序列 [0, 1, 2, 1]，prompt = 前两个 token，回复 = 后两个
    probs = torch.tensor([[[0.2, 0.5, 0.3], [0.1, 0.1, 0.8], [0.3, 0.6, 0.1], [1 / 3] * 3]])
    logits = probs.log()
    ids = torch.tensor([[0, 1, 2, 1]])
    mask = torch.tensor([[False, False, True, True]])
    lp = sequence_logprob(logits, ids, mask)
    torch.testing.assert_close(lp, torch.tensor([math.log(0.8) + math.log(0.6)]))
    avg = sequence_logprob(logits, ids, mask, average=True)
    torch.testing.assert_close(avg, lp / 2)


def test_dpo_matches_hand_calculation():
    pc, pr = torch.tensor([math.log(0.44)]), torch.tensor([math.log(0.18)])
    rc, rr = torch.tensor([math.log(0.30)]), torch.tensor([math.log(0.20)])
    pc.requires_grad_()
    pr.requires_grad_()
    loss, rew_c, rew_r = dpo_loss(pc, pr, rc, rr, beta=0.2)
    margin = 0.2 * math.log((0.44 / 0.18) / (0.30 / 0.20))
    assert abs(loss.item() - math.log1p(math.exp(-margin))) < 1e-6
    torch.testing.assert_close(rew_c - rew_r, torch.tensor([margin]))
    loss.backward()
    # ∂L/∂log π_w = −β σ(−margin)，∂L/∂log π_l = +β σ(−margin)
    w = 0.2 * (1 / (1 + math.exp(margin)))
    torch.testing.assert_close(pc.grad, torch.tensor([-w]))
    torch.testing.assert_close(pr.grad, torch.tensor([w]))


def test_dpo_at_reference_is_log2():
    x = torch.randn(5)
    loss, _, _ = dpo_loss(x, x - 1, x, x - 1, beta=0.5)
    assert abs(loss.item() - math.log(2)) < 1e-6


def test_ipo_minimum_at_target_margin():
    tau = 0.25
    h = torch.tensor([1 / (2 * tau)])
    zero = torch.zeros(1)
    assert ipo_loss(h, zero, zero, zero, tau).item() < 1e-12
    assert ipo_loss(h + 3, zero, zero, zero, tau).item() > 0  # 超过目标也会被拉回


def test_simpo_and_kto_basic():
    a = torch.tensor([-1.0])
    b = torch.tensor([-2.0])
    assert abs(simpo_loss(a, b, beta=2.0, gamma=1.0).item() - math.log1p(math.exp(-1.0))) < 1e-6
    lp = torch.tensor([0.5, 0.5], requires_grad=True)
    loss = kto_loss(lp, torch.zeros(2), torch.tensor([True, False]), torch.tensor(0.0), beta=1.0)
    loss.backward()
    assert lp.grad[0] < 0 < lp.grad[1]  # 好样本往上推，坏样本往下压


def test_dpo_training_increases_margin_on_tiny_gpt():
    torch.set_num_threads(1)
    tok = arithmetic_tokenizer()
    problems = addition_problems(4)
    chosen = torch.tensor([tok.encode(p.prompt + p.answer) for p in problems])
    rejected = torch.tensor(
        [tok.encode(p.prompt + str((int(p.answer) + 3) % 10)) for p in problems]
    )
    mask = torch.zeros_like(chosen, dtype=torch.bool)
    mask[:, -1] = True  # 回复只有最后一个 token
    policy = tiny_gpt(tok.vocab_size, seed=0, context_length=16)
    ref = copy.deepcopy(policy).requires_grad_(False)
    with torch.no_grad():
        ref_c = model_sequence_logprob(ref, chosen, mask)
        ref_r = model_sequence_logprob(ref, rejected, mask)
    opt = torch.optim.AdamW(policy.parameters(), lr=3e-3)
    for _ in range(30):
        loss, rc, rr = dpo_loss(model_sequence_logprob(policy, chosen, mask),
                                model_sequence_logprob(policy, rejected, mask),
                                ref_c, ref_r, beta=0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5 * math.log(2)
    assert (rc - rr).mean() > 0
