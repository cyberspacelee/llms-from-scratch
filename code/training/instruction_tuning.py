"""Assistant target masks and prompt gradients in a causal model."""

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "principles"))
from decoder import Decoder
from language_model import sequence_loss


def reply_mask(roles):
    # Ownership refers to each target token, including assistant end-of-turn.
    return torch.tensor([role == "assistant" for role in roles[1:]], dtype=torch.bool)


def verify():
    torch.manual_seed(17)
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    roles = ["system", "system", "user", "user", "assistant_header", "assistant",
             "assistant", "user", "user", "assistant_header", "assistant", "assistant"]
    valid = reply_mask(roles)
    assert valid.nonzero().flatten().tolist() == [4, 5, 9, 10]
    embedding = torch.nn.Embedding(12, 4).double()
    head = torch.nn.Linear(4, 12, bias=False).double()
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
    # Prompt tokens are conditioned upon even though their direct loss is masked.
    labels = ids[1:].clone()
    labels[~valid] = -100
    torch.testing.assert_close(loss, F.cross_entropy(logits, labels, ignore_index=-100))
    print("SFT: assistant prediction rows", valid.nonzero().flatten().tolist())
    print("SFT: masked logits have zero direct gradient; prompt embedding receives gradient")

    torch.set_num_threads(1)
    torch.manual_seed(37)
    model = Decoder(9, width=16, heads=2, ff_width=32, layers=2, max_length=8).double()
    # A tiny format task: two training prompts both request the constant reply [5,7].
    conversations = torch.tensor([[0, 1, 2, 4, 5, 7], [0, 1, 3, 4, 5, 7]])
    targets = conversations[:, 1:]
    assistant = torch.tensor([[False, False, False, True, True]]).expand(2, -1)
    heldout = torch.tensor([[0, 1, 8, 4, 5, 7]])
    heldout_mask = assistant[:1]

    @torch.no_grad()
    def measure():
        model.eval()
        nll = sequence_loss(model(heldout[:, :-1]), heldout[:, 1:], heldout_mask).item()
        generated = heldout[:, :4].clone()
        for _ in range(2):
            following = model(generated)[:, -1].argmax(-1, keepdim=True)
            generated = torch.cat((generated, following), dim=1)
        return nll, generated[0, -2:].tolist()

    before, before_reply = measure()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0, foreach=False)
    for _ in range(40):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        objective = sequence_loss(model(conversations[:, :-1]), targets, assistant)
        objective.backward()
        optimizer.step()
    after, after_reply = measure()
    assert after < before
    print(f"SFT format task: heldout reply NLL {before:.6f} -> {after:.6f}")
    print("SFT format task: generated reply", before_reply, "->", after_reply, "; target [5, 7]")


def distillation_loss(student_logits, teacher_logits, temperature=1.0):
    """tau^2 * KL(teacher_tau || student_tau), averaged over positions."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    teacher = (teacher_logits / temperature).softmax(-1)
    student_log = (student_logits / temperature).log_softmax(-1)
    kl = (teacher * (teacher.clamp_min(1e-300).log() - student_log)).sum(-1)
    return temperature ** 2 * kl.mean()


def verify_distillation():
    torch.manual_seed(13)
    student = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(3, 5, dtype=torch.float64)
    # tau = 1: the gradient on each student logit is (p_student - p_teacher) / positions.
    distillation_loss(student, teacher).backward()
    expected = (student.detach().softmax(-1) - teacher.softmax(-1)) / 3
    torch.testing.assert_close(student.grad, expected, atol=1e-12, rtol=1e-12)
    # A one-hot teacher turns the KL into ordinary cross-entropy on its label.
    labels = torch.tensor([4, 0, 2])
    one_hot = torch.full((3, 5), -torch.inf, dtype=torch.float64).scatter(1, labels[:, None], 0.)
    torch.testing.assert_close(distillation_loss(student, one_hot),
                               torch.nn.functional.cross_entropy(student, labels), atol=1e-12, rtol=1e-12)
    # The tau^2 factor keeps gradient size comparable: at tau = 4 it rescales the 1/tau^2 shrinkage.
    student.grad = None
    distillation_loss(student, teacher, temperature=4.).backward()
    soft = (student.detach() / 4).softmax(-1) - (teacher / 4).softmax(-1)
    torch.testing.assert_close(student.grad, 4 * soft / 3, atol=1e-12, rtol=1e-12)
    assert distillation_loss(teacher, teacher).abs() < 1e-12
    print("distillation: KL gradient p_s - p_t, one-hot teacher equals CE, tau^2 scaling verified")


if __name__ == "__main__":
    verify()
    verify_distillation()
