import math

import torch

from llms_from_scratch.post_training.common import tiny_gpt
from llms_from_scratch.post_training.reward_model import (
    RewardModel,
    bradley_terry_loss,
    bradley_terry_prob,
    preference_accuracy,
)
from llms_from_scratch.post_training.tasks import addition_problems, arithmetic_tokenizer


def test_bt_loss_values_and_gradient():
    rc = torch.tensor([0.0, 2.0], requires_grad=True)
    rr = torch.tensor([0.0, 0.0])
    loss = bradley_terry_loss(rc, rr)
    expected = (math.log(2) + math.log1p(math.exp(-2))) / 2
    assert abs(loss.item() - expected) < 1e-6
    loss.backward()
    # ∂ℓ/∂r_w = −σ(r_l − r_w)/B：总把选中回复的奖励往上推，已经排对的样本推得更轻
    torch.testing.assert_close(rc.grad, -torch.sigmoid(-torch.tensor([0.0, 2.0])) / 2)
    assert bradley_terry_prob(torch.tensor(1.0), torch.tensor(1.0)).item() == 0.5


def test_bt_loss_is_shift_invariant():
    rc, rr = torch.randn(8), torch.randn(8)
    torch.testing.assert_close(bradley_terry_loss(rc, rr), bradley_terry_loss(rc + 5, rr + 5))


def test_reward_model_learns_to_rank_correct_answers():
    """偏好对：同一道加法题，正确答案优于错误答案。训练后奖励模型能把对排对。"""
    torch.set_num_threads(1)
    tok = arithmetic_tokenizer()
    gen = torch.Generator().manual_seed(0)
    chosen, rejected = [], []
    for p in addition_problems(4):
        wrong = (int(p.answer) + 1 + torch.randint(0, 8, (1,), generator=gen).item()) % 10
        chosen.append(tok.encode(p.prompt + p.answer))
        rejected.append(tok.encode(p.prompt + str(wrong)))
    xc, xr = torch.tensor(chosen), torch.tensor(rejected)
    lengths = torch.full((len(chosen),), xc.shape[1])
    rm = RewardModel(tiny_gpt(tok.vocab_size, seed=0, context_length=16))
    opt = torch.optim.AdamW(rm.parameters(), lr=3e-3)
    for _ in range(80):
        loss = bradley_terry_loss(rm(xc, lengths), rm(xr, lengths))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = preference_accuracy(rm(xc, lengths), rm(xr, lengths))
    assert loss.item() < 0.3 and acc > 0.9


def test_reward_read_at_last_valid_token():
    rm = RewardModel(tiny_gpt(15, seed=0, context_length=16))
    torch.nn.init.normal_(rm.head.weight)
    x = torch.tensor([[3, 4, 5, 0, 0], [3, 4, 5, 9, 9]])
    scores = rm(x, torch.tensor([3, 3]))
    torch.testing.assert_close(scores[0], scores[1])  # 填充内容不影响分数
