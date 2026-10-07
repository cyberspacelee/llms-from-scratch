import torch

from llms_from_scratch.post_training.common import tiny_gpt
from llms_from_scratch.post_training.distill import (
    exact_reverse_kl,
    fit_gaussian,
    kd_loss,
    kl_divergence,
    on_policy_distill_step,
)
from llms_from_scratch.post_training.tasks import addition_problems, arithmetic_tokenizer


def test_kd_loss_matches_kl_and_vanishes_at_teacher():
    torch.manual_seed(0)
    s, t = torch.randn(3, 5), torch.randn(3, 5)
    expected = 4.0 * kl_divergence(t / 2, s / 2).mean()
    torch.testing.assert_close(kd_loss(s, t, temperature=2.0), expected)
    assert kd_loss(t, t, temperature=3.0).item() < 1e-7


def test_kd_gradient_and_temperature_squared_factor():
    """∂/∂z_s [T²·KL] = T·(q_s − q_t)；T 很大且 logits 零均值时 ≈ (z_s − z_t)/N。"""
    torch.manual_seed(0)
    N = 6
    zs = torch.randn(N, dtype=torch.float64)
    zt = torch.randn(N, dtype=torch.float64)
    zs, zt = zs - zs.mean(), zt - zt.mean()
    for T in (1.0, 4.0):
        z = zs.clone().requires_grad_()
        kd_loss(z[None], zt[None], temperature=T).backward()
        expected = T * ((zs / T).softmax(-1) - (zt / T).softmax(-1))
        torch.testing.assert_close(z.grad, expected)
    z = zs.clone().requires_grad_()
    kd_loss(z[None], zt[None], temperature=1000.0).backward()
    torch.testing.assert_close(z.grad, (zs - zt) / N, atol=1e-3, rtol=0)


def test_forward_kl_covers_modes_reverse_kl_seeks_one():
    m_f, s_f = fit_gaussian("forward")
    m_r, s_r = fit_gaussian("reverse")
    # 前向 KL 的最优解是矩匹配：均值 0，方差 0.6² + 2²
    assert abs(m_f) < 1e-3 and abs(s_f - (0.36 + 4) ** 0.5) < 1e-2
    # 反向 KL 锁定一个峰
    assert abs(m_r - 2.0) < 0.05 and abs(s_r - 0.6) < 0.05


def test_on_policy_distillation_reduces_reverse_kl():
    torch.set_num_threads(1)
    tok = arithmetic_tokenizer()
    prompts = torch.tensor([tok.encode(p.prompt) for p in addition_problems(4)]).repeat(4, 1)
    teacher = tiny_gpt(tok.vocab_size, seed=1, context_length=16).requires_grad_(False)
    with torch.no_grad():  # 让教师的分布更尖锐，与学生差异更明显
        teacher.lm_head.weight.mul_(20)
    student = tiny_gpt(tok.vocab_size, seed=2, context_length=16)
    opt = torch.optim.AdamW(student.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(0)
    first = on_policy_distill_step(student, teacher, prompts, opt, 3, tok.eos_id, gen)
    roll = first["rollout"]
    before = exact_reverse_kl(student, teacher, roll.sequences, roll.response_mask)
    for _ in range(15):
        on_policy_distill_step(student, teacher, prompts, opt, 3, tok.eos_id, gen)
    after = exact_reverse_kl(student, teacher, roll.sequences, roll.response_mask)
    assert after < 0.7 * before


def test_on_policy_signal_is_zero_when_student_equals_teacher():
    tok = arithmetic_tokenizer()
    student = tiny_gpt(tok.vocab_size, seed=3, context_length=16)
    teacher = tiny_gpt(tok.vocab_size, seed=3, context_length=16)
    opt = torch.optim.SGD(student.parameters(), lr=0.1)
    prompts = torch.tensor([tok.encode("1+2=")] * 4)
    out = on_policy_distill_step(student, teacher, prompts, opt, 2, tok.eos_id)
    assert out["loss"] == 0 and out["reverse_kl_estimate"] == 0
