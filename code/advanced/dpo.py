"""Recompute a DPO preference pair from response tokens and a frozen reference."""
import math
import torch
from torch.nn import functional as F


def response_log_prob(logits, labels, mask):
    if logits.shape[:-1] != labels.shape or labels.shape != mask.shape or mask.dtype != torch.bool:
        raise ValueError("expected matching logits, labels and boolean response mask")
    if not mask.any(-1).all():
        raise ValueError("every response must have at least one scored token")
    safe_labels = labels.masked_fill(~mask, 0)
    selected = logits.log_softmax(-1).gather(-1, safe_labels[..., None]).squeeze(-1)
    return selected.masked_fill(~mask, 0).sum(-1)


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, coefficient):
    if coefficient <= 0:
        raise ValueError("positive DPO coefficient required")
    margin = coefficient * ((chosen - rejected) - (ref_chosen.detach() - ref_rejected.detach()))
    return F.softplus(-margin).mean()


def verify():
    dtype = torch.float64
    # Vocabulary IDs: 4, 5, EOS, ?, other. Both candidates share prompt logits.
    probabilities = torch.tensor([
        [[.1, .1, .1, .6, .1], [.55, .30, .05, .05, .05], [.05, .05, .8, .05, .05], [.2] * 5],
        [[.1, .1, .1, .6, .1], [.55, .30, .05, .05, .05], [.1, .1, .6, .1, .1], [.2] * 5],
    ], dtype=dtype)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones((2, 4), dtype=dtype))
    torch.testing.assert_close(probabilities[0, 1], probabilities[1, 1])
    logits = probabilities.log().requires_grad_()
    labels = torch.tensor([[3, 0, 2, -999], [3, 1, 2, -999]])
    mask = torch.tensor([[False, True, True, False]] * 2)
    lp = response_log_prob(logits, labels, mask)
    torch.testing.assert_close(lp, torch.tensor([math.log(.44), math.log(.18)], dtype=dtype))
    ref_probabilities = torch.tensor([
        [[.1, .1, .1, .6, .1], [.4, .4, .05, .05, .1], [.0625, .0625, .75, .0625, .0625], [.2] * 5],
        [[.1, .1, .1, .6, .1], [.4, .4, .05, .05, .1], [.125, .125, .5, .125, .125], [.2] * 5],
    ], dtype=dtype)
    torch.testing.assert_close(ref_probabilities[0, 1], ref_probabilities[1, 1])
    ref_logits = ref_probabilities.log().requires_grad_()
    ref = response_log_prob(ref_logits, labels, mask)
    torch.testing.assert_close(ref, torch.tensor([math.log(.30), math.log(.20)], dtype=dtype))
    loss = dpo_loss(lp[:1], lp[1:], ref[:1], ref[1:], .2)
    margin = .2 * math.log((.44 / .18) / (.30 / .20))
    assert abs(loss.item() - math.log1p(math.exp(-margin))) < 1e-12
    lp_grad = torch.autograd.grad(loss, lp, retain_graph=True)[0]
    expected = .2 / (1 + math.exp(margin))
    torch.testing.assert_close(lp_grad, torch.tensor([-expected, expected], dtype=dtype))
    loss.backward()
    assert ref_logits.grad is None
    assert torch.equal(logits.grad[:, [0, 3]], torch.zeros_like(logits.grad[:, [0, 3]]))
    assert logits.grad[0, 1, 0] < 0 and logits.grad[1, 1, 1] > 0
    print(f"DPO: margin={margin:.6f}, loss={loss.item():.6f}; reference frozen")


if __name__ == "__main__":
    verify()
