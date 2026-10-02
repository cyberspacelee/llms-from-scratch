"""Teacher distributions, temperature and four assistant targets, independently checked."""

import math

import torch
from torch.nn import functional as F

from examples.post_training.instruction_tuning import OWNERS, STREAM, TOKENS, reply_mask


def distillation_loss(student_logits, teacher_logits, valid, temperature=1.0):
    """tau² KL over valid positions; finite student and defined teacher distributions.

    Teacher -inf entries represent zero probability (including one-hot targets).
    Ignored positions do not participate in either validation or differentiation.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive")
    if (
        student_logits.shape != teacher_logits.shape
        or valid.shape != student_logits.shape[:-1]
        or valid.dtype != torch.bool
    ):
        raise ValueError("aligned logits and boolean target mask required")
    if not valid.any():
        return student_logits[valid].sum()
    # Select before softmax: ignored non-finite rows must not enter the backward graph.
    teacher_values = teacher_logits.detach()[valid] / temperature
    student_values = student_logits[valid] / temperature
    if not torch.isfinite(student_values).all():
        raise ValueError("valid student logits must remain finite after temperature scaling")
    if (
        torch.isnan(teacher_values).any()
        or torch.isposinf(teacher_values).any()
        or not torch.isfinite(teacher_values).any(-1).all()
    ):
        raise ValueError("valid teacher rows need finite support and no NaN or +inf")
    teacher = teacher_values.softmax(-1)
    student_log = student_values.log_softmax(-1)
    # Native KL handles p=0 without a dtype-dependent epsilon (one-hot teachers are valid).
    kl = F.kl_div(student_log, teacher, reduction="none").sum(-1)
    return temperature**2 * kl.mean()


def verify_distillation():
    torch.manual_seed(13)
    valid = reply_mask(OWNERS)
    student = torch.randn(len(valid), len(TOKENS), dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(student, requires_grad=True)
    distillation_loss(student, teacher, valid).backward()
    assert teacher.grad is None
    expected = student.detach().softmax(-1) - teacher.softmax(-1)
    expected[~valid] = 0
    expected /= valid.sum()
    torch.testing.assert_close(student.grad, expected, atol=1e-12, rtol=1e-12)
    one_hot = torch.full_like(student, -torch.inf).scatter(1, STREAM[1:, None], 0.0)
    torch.testing.assert_close(
        distillation_loss(student, one_hot, valid),
        F.cross_entropy(student[valid], STREAM[1:][valid]),
        atol=1e-12,
        rtol=1e-12,
    )
    student.grad = None
    distillation_loss(student, teacher, valid, temperature=4.0).backward()
    soft = (student.detach() / 4).softmax(-1) - (teacher / 4).softmax(-1)
    soft[~valid] = 0
    torch.testing.assert_close(student.grad, 4 * soft / valid.sum(), atol=1e-12, rtol=1e-12)
    assert distillation_loss(teacher, teacher, valid).abs() < 1e-12
    oracle_student = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True)
    oracle_teacher = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float64).log().requires_grad_()
    oracle_loss = distillation_loss(oracle_student, oracle_teacher, torch.tensor([True]))
    expected_loss = sum(p * math.log(p * 3) for p in (0.7, 0.2, 0.1))
    torch.testing.assert_close(oracle_loss, torch.tensor(expected_loss, dtype=torch.float64))
    oracle_loss.backward()
    torch.testing.assert_close(
        oracle_student.grad,
        torch.tensor([[1 / 3 - 0.7, 1 / 3 - 0.2, 1 / 3 - 0.1]], dtype=torch.float64),
    )
    assert oracle_teacher.grad is None
    empty = distillation_loss(oracle_student, oracle_teacher, torch.tensor([False]))
    assert empty.item() == 0 and empty.requires_grad
    print(f"distillation: teacher [.7,.2,.1], student uniform, KL={oracle_loss.item():.9f}")
    print(
        "distillation: four assistant targets, zero prompt logits gradient, one-hot equals SFT; teacher frozen; empty contribution zero"
    )


def verify_model_distillation():
    """Reply mask and detached teacher through the same canonical LM used by SFT."""
    from llms_from_scratch import Transformer, basic_decoder_config

    torch.set_num_threads(1)
    torch.manual_seed(59)
    config = basic_decoder_config(
        len(TOKENS), dim=8, ff_dim=16, layers=1, heads=2, max_length=len(STREAM)
    )
    teacher = Transformer(config).double().eval().requires_grad_(False)
    student = Transformer(config).double()
    inputs, valid = STREAM[None, :-1], reply_mask(OWNERS)[None]
    with torch.no_grad():
        teacher_logits = teacher(inputs).logits
    optimizer = torch.optim.AdamW(student.parameters(), lr=0.002, weight_decay=0.0)
    initial = None
    for _ in range(15):
        optimizer.zero_grad(set_to_none=True)
        objective = distillation_loss(
            student(inputs).logits, teacher_logits, valid, temperature=2.0
        )
        initial = objective.item() if initial is None else initial
        objective.backward()
        optimizer.step()
    assert objective.item() < initial
    assert all(p.grad is None for p in teacher.parameters())
    print(
        f"Distillation canonical LM: frozen teacher and reply-only tau²KL; {initial:.9f} -> {objective.item():.9f}"
    )


if __name__ == "__main__":
    verify_distillation()
    verify_model_distillation()
