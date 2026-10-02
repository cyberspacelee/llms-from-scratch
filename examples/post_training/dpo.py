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
    probabilities = torch.tensor(
        [
            [
                [0.1, 0.1, 0.1, 0.6, 0.1],
                [0.55, 0.30, 0.05, 0.05, 0.05],
                [0.05, 0.05, 0.8, 0.05, 0.05],
                [0.2] * 5,
            ],
            [
                [0.1, 0.1, 0.1, 0.6, 0.1],
                [0.55, 0.30, 0.05, 0.05, 0.05],
                [0.1, 0.1, 0.6, 0.1, 0.1],
                [0.2] * 5,
            ],
        ],
        dtype=dtype,
    )
    torch.testing.assert_close(probabilities.sum(-1), torch.ones((2, 4), dtype=dtype))
    torch.testing.assert_close(probabilities[0, 1], probabilities[1, 1])
    logits = probabilities.log().requires_grad_()
    labels = torch.tensor([[3, 0, 2, -999], [3, 1, 2, -999]])
    mask = torch.tensor([[False, True, True, False]] * 2)
    lp = response_log_prob(logits, labels, mask)
    torch.testing.assert_close(lp, torch.tensor([math.log(0.44), math.log(0.18)], dtype=dtype))
    ref_probabilities = torch.tensor(
        [
            [
                [0.1, 0.1, 0.1, 0.6, 0.1],
                [0.4, 0.4, 0.05, 0.05, 0.1],
                [0.0625, 0.0625, 0.75, 0.0625, 0.0625],
                [0.2] * 5,
            ],
            [
                [0.1, 0.1, 0.1, 0.6, 0.1],
                [0.4, 0.4, 0.05, 0.05, 0.1],
                [0.125, 0.125, 0.5, 0.125, 0.125],
                [0.2] * 5,
            ],
        ],
        dtype=dtype,
    )
    torch.testing.assert_close(ref_probabilities[0, 1], ref_probabilities[1, 1])
    ref_logits = ref_probabilities.log().requires_grad_()
    ref = response_log_prob(ref_logits, labels, mask)
    torch.testing.assert_close(ref, torch.tensor([math.log(0.30), math.log(0.20)], dtype=dtype))
    loss = dpo_loss(lp[:1], lp[1:], ref[:1], ref[1:], 0.2)
    margin = 0.2 * math.log((0.44 / 0.18) / (0.30 / 0.20))
    assert abs(loss.item() - math.log1p(math.exp(-margin))) < 1e-12
    lp_grad = torch.autograd.grad(loss, lp, retain_graph=True)[0]
    expected = 0.2 / (1 + math.exp(margin))
    torch.testing.assert_close(lp_grad, torch.tensor([-expected, expected], dtype=dtype))
    loss.backward()
    assert ref_logits.grad is None
    assert torch.equal(logits.grad[:, [0, 3]], torch.zeros_like(logits.grad[:, [0, 3]]))
    assert logits.grad[0, 1, 0] < 0 and logits.grad[1, 1, 1] > 0
    print(f"DPO: margin={margin:.6f}, loss={loss.item():.6f}; reference frozen")


def verify_model_update():
    """Score the same two response candidates using a trainable canonical LM."""
    import copy

    from llms_from_scratch import Transformer, decoder_config

    torch.set_num_threads(1)
    torch.manual_seed(47)
    model = Transformer(
        decoder_config(8, dim=8, ff_dim=16, layers=1, heads=2, kv_heads=1, head_dim=4, max_length=8)
    ).double()
    reference = copy.deepcopy(model).eval().requires_grad_(False)
    # Shared [BOS, prompt], candidates [chosen/rejected, EOS]; only response targets count.
    tokens = torch.tensor([[0, 1, 2, 4], [0, 1, 3, 4]])
    labels, valid = tokens[:, 1:], torch.tensor([[False, True, True]] * 2)
    with torch.no_grad():
        ref = response_log_prob(reference(tokens[:, :-1]).logits, labels, valid)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.0)
    initial = None
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        logp = response_log_prob(model(tokens[:, :-1]).logits, labels, valid)
        loss = dpo_loss(logp[:1], logp[1:], ref[:1], ref[1:], 0.2)
        initial = loss.item() if initial is None else initial
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert loss.item() < initial
    assert all(p.grad is None for p in reference.parameters())
    print(
        f"DPO canonical LM: two reply masks and frozen reference; objective {initial:.6f} -> {loss.item():.6f}"
    )


if __name__ == "__main__":
    verify()
    verify_model_update()
