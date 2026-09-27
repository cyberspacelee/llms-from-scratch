"""Response-only sequence likelihood and a frozen-reference DPO objective."""
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
    probabilities = torch.tensor([[[.5, .5], [.8, .2], [.7, .3], [.2, .8]],
                                  [[.5, .5], [.6, .4], [.4, .6], [.3, .7]]], dtype=dtype)
    logits = probabilities.log().requires_grad_()
    labels = torch.tensor([[0, 0, 0, -999], [0, 0, 0, -999]])
    mask = torch.tensor([[False, True, True, False]] * 2)
    lp = response_log_prob(logits, labels, mask)
    torch.testing.assert_close(lp, torch.tensor([math.log(.56), math.log(.24)], dtype=dtype))
    ref = torch.tensor([math.log(.4), math.log(.3)], dtype=dtype, requires_grad=True)
    loss = dpo_loss(lp[:1], lp[1:], ref[:1], ref[1:], .2)
    margin = .2 * math.log((.56 / .24) / (.4 / .3))
    assert abs(loss.item() - math.log1p(math.exp(-margin))) < 1e-12
    loss.backward()
    assert ref.grad is None
    assert torch.equal(logits.grad[:, [0, 3]], torch.zeros_like(logits.grad[:, [0, 3]]))
    assert logits.grad[0, 1, 0] < 0 and logits.grad[1, 1, 0] > 0
    print(f"DPO: margin={margin:.6f}, loss={loss.item():.6f}; reference frozen")


if __name__ == "__main__":
    verify()
