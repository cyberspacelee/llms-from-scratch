"""One two-turn conversation through SFT masks, gradients, and distillation."""

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "principles"))
from decoder import Decoder
from language_model import sequence_loss


TOKENS = {"SYS": 0, "RULE": 1, "USER": 2, "Q1": 3, "ASSISTANT": 4,
          "OK": 5, "Q2": 6, "END": 7}
STREAM = torch.tensor([0, 1, 2, 3, 4, 5, 7, 2, 6, 4, 5, 7])
OWNERS = ["system", "system", "user", "user", "header", "assistant",
          "assistant", "user", "user", "header", "assistant", "assistant"]


def reply_mask(owners):
    return torch.tensor([owner == "assistant" for owner in owners[1:]], dtype=torch.bool)


def verify():
    torch.manual_seed(17)
    ids = STREAM
    valid = reply_mask(OWNERS)
    assert valid.nonzero().flatten().tolist() == [4, 5, 9, 10]
    assert ids[2] == ids[7] == TOKENS["USER"]
    assert ids[4] == ids[9] == TOKENS["ASSISTANT"]
    embedding = torch.nn.Embedding(len(TOKENS), 4).double()
    head = torch.nn.Linear(4, len(TOKENS), bias=False).double()
    hidden = embedding(ids[:-1])
    hidden.retain_grad()
    length = len(hidden)
    allowed = torch.ones(length, length, dtype=torch.float64).tril()
    # Uniform causal attention makes the prompt-to-reply gradient path explicit.
    attention = allowed / allowed.sum(dim=-1, keepdim=True)
    logits = head(attention @ hidden)
    logits.retain_grad()
    losses = F.cross_entropy(logits, ids[1:], reduction="none")
    loss = losses[valid].mean()
    loss.backward()
    assert torch.count_nonzero(logits.grad[~valid]) == 0
    assert torch.count_nonzero(logits.grad[valid]) > 0
    assert hidden.grad[0].abs().sum() > 0
    assert embedding.weight.grad[ids[0]].abs().sum() > 0
    labels = ids[1:].clone()
    labels[~valid] = -100
    torch.testing.assert_close(loss, F.cross_entropy(logits, labels, ignore_index=-100))
    print("SFT: assistant prediction rows", valid.nonzero().flatten().tolist())
    print("SFT: masked logits have zero direct gradient; prompt embedding receives gradient")

    torch.set_num_threads(1)
    torch.manual_seed(37)
    model = Decoder(len(TOKENS), width=16, heads=2, ff_width=32, layers=2,
                    max_length=len(ids)).double()
    conversation = ids[None]
    targets = conversation[:, 1:]
    assistant = valid[None]

    @torch.no_grad()
    def measure():
        model.eval()
        nll = sequence_loss(model(conversation[:, :-1]), targets, assistant).item()
        generated = conversation[:, :5].clone()
        replies = []
        for turn in range(2):
            for _ in range(2):
                following = model(generated)[:, -1].argmax(-1, keepdim=True)
                generated = torch.cat((generated, following), dim=1)
            replies.append(generated[0, -2:].tolist())
            if turn == 0:
                generated = torch.cat((generated, conversation[:, 7:10]), dim=1)
        return nll, replies

    before, before_reply = measure()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0, foreach=False)
    for _ in range(40):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        objective = sequence_loss(model(conversation[:, :-1]), targets, assistant)
        objective.backward()
        optimizer.step()
    after, after_reply = measure()
    assert after < before
    assert after_reply == [[TOKENS["OK"], TOKENS["END"]]] * 2
    print(f"SFT: same conversation reply NLL {before:.6f} -> {after:.6f}")
    print("SFT: generated replies", before_reply, "->", after_reply,
          "; target [[5, 7], [5, 7]]")


def distillation_loss(student_logits, teacher_logits, valid, temperature=1.0):
    """tau^2 * KL(teacher_tau || student_tau), averaged over positions."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher = (teacher_logits / temperature).softmax(-1)
    student_log = (student_logits / temperature).log_softmax(-1)
    kl = (teacher * (teacher.clamp_min(1e-300).log() - student_log)).sum(-1)
    if not valid.any():
        raise ValueError("at least one assistant target is required")
    return temperature ** 2 * kl[valid].mean()


def verify_distillation():
    torch.manual_seed(13)
    valid = reply_mask(OWNERS)
    student = torch.randn(len(valid), len(TOKENS), dtype=torch.float64, requires_grad=True)
    teacher = torch.randn_like(student)
    distillation_loss(student, teacher, valid).backward()
    expected = student.detach().softmax(-1) - teacher.softmax(-1)
    expected[~valid] = 0
    expected /= valid.sum()
    torch.testing.assert_close(student.grad, expected, atol=1e-12, rtol=1e-12)
    one_hot = torch.full_like(student, -torch.inf).scatter(1, STREAM[1:, None], 0.)
    torch.testing.assert_close(distillation_loss(student, one_hot, valid),
                               F.cross_entropy(student[valid], STREAM[1:][valid]),
                               atol=1e-12, rtol=1e-12)
    student.grad = None
    distillation_loss(student, teacher, valid, temperature=4.).backward()
    soft = (student.detach() / 4).softmax(-1) - (teacher / 4).softmax(-1)
    soft[~valid] = 0
    torch.testing.assert_close(student.grad, 4 * soft / valid.sum(), atol=1e-12, rtol=1e-12)
    assert distillation_loss(teacher, teacher, valid).abs() < 1e-12
    print("distillation: four assistant targets, zero prompt logits gradient, one-hot equals SFT")


if __name__ == "__main__":
    verify()
    verify_distillation()
