"""Next-token targets, valid-token losses, and a trained bigram baseline."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def sequence_loss(logits, targets, valid=None):
    if logits.ndim != 3 or targets.shape != logits.shape[:2]:
        raise ValueError("expected logits=[B,T,V] and targets=[B,T]")
    if targets.dtype != torch.long:
        raise ValueError("targets must be integer token IDs")
    if valid is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    if valid.shape != targets.shape or valid.dtype != torch.bool:
        raise ValueError("valid must be a boolean mask with the target shape")
    if not valid.any():
        raise ValueError("a loss needs at least one valid prediction position")
    selected = targets[valid]
    if (selected < 0).any() or (selected >= logits.shape[-1]).any():
        raise ValueError("valid targets must belong to the vocabulary")
    safe_targets = targets.masked_fill(~valid, -100)
    losses = F.cross_entropy(logits.transpose(1, 2), safe_targets,
                             reduction="none", ignore_index=-100)
    return losses.sum() / valid.sum()


class BigramLM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        if vocab_size < 2:
            raise ValueError("a vocabulary needs at least two entries")
        self.table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, token_ids):
        return self.table(token_ids)


def verify():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    close = torch.testing.assert_close
    probabilities = torch.tensor([
        [.1, .6, .2, .1], [.1, .1, .7, .1],
        [.1, .1, .2, .6], [.1, .1, .1, .7],
    ], dtype=torch.float64)
    document = torch.tensor([[0, 1, 2, 3]])
    inputs, targets = document[:, :-1], document[:, 1:]
    logits = probabilities.log()[inputs].detach().requires_grad_()
    expected = torch.tensor(-(math.log(.6) + math.log(.7) + math.log(.6)) / 3, dtype=torch.float64)
    close(sequence_loss(logits, targets), expected.double(), rtol=1e-7, atol=1e-8)
    true_probabilities = probabilities[inputs, targets]
    joint_probability = true_probabilities.prod().item()
    assert math.isclose(joint_probability, .252)
    assert math.isclose(-math.log(joint_probability), 3 * expected.item())
    perplexity = math.exp(expected.item())
    assert math.isclose(perplexity, joint_probability ** (-1 / targets.numel()))
    assert math.isclose(perplexity, 1 / true_probabilities.log().mean().exp().item())
    assert not math.isclose(perplexity, 1 / true_probabilities.mean().item())
    assert math.isclose(math.exp(math.log(4)), 4)
    assert math.isclose(math.exp((math.log(2) + 3 * math.log(8)) / 4), math.sqrt(32))
    assert sequence_loss(logits, inputs) > sequence_loss(logits, targets)
    padded_logits = F.pad(logits.detach(), (0, 0, 0, 2)).requires_grad_()
    padded_targets = torch.tensor([[1, 2, 3, -999, -999]])
    valid = torch.tensor([[True, True, True, False, False]])
    padded_loss = sequence_loss(padded_logits, padded_targets, valid)
    close(padded_loss, sequence_loss(logits, targets))
    padded_loss.backward()
    close(padded_logits.grad[:, 3:], torch.zeros(1, 2, 4, dtype=torch.float64))
    probs = logits.softmax(-1)
    expected_gradient = (probs - F.one_hot(targets, 4).double()) / 3
    sequence_loss(logits, targets).backward()
    close(logits.grad, expected_gradient)

    hand_model = BigramLM(4).double()
    with torch.no_grad():
        hand_model.table.weight.zero_()
    hand_optimizer = torch.optim.SGD(hand_model.parameters(), lr=1.0)
    hand_loss = sequence_loss(hand_model(inputs), targets)
    close(hand_loss, torch.tensor(math.log(4), dtype=torch.float64))
    hand_loss.backward()
    close(hand_model.table.weight.grad[0],
          torch.tensor([1 / 12, -1 / 4, 1 / 12, 1 / 12],
                       dtype=torch.float64))
    hand_optimizer.step()
    close(hand_model.table.weight[0].detach(),
          torch.tensor([-1 / 12, 1 / 4, -1 / 12, -1 / 12],
                       dtype=torch.float64))
    close(hand_model.table.weight[1].detach(),
          torch.tensor([-1 / 12, -1 / 12, 1 / 4, -1 / 12],
                       dtype=torch.float64))
    close(hand_model.table.weight[2].detach(),
          torch.tensor([-1 / 12, -1 / 12, -1 / 12, 1 / 4],
                       dtype=torch.float64))
    close(hand_model.table.weight[3].detach(), torch.zeros(4, dtype=torch.float64))
    expected_probability = math.exp(1 / 3) / (math.exp(1 / 3) + 3)
    close(hand_model(inputs).softmax(-1)[0, 0, 1].detach(),
          torch.tensor(expected_probability, dtype=torch.float64))
    assert sequence_loss(hand_model(inputs), targets) < hand_loss

    model = BigramLM(4).double()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    initial = sequence_loss(model(inputs), targets).item()
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        sequence_loss(model(inputs), targets).backward()
        optimizer.step()
    final = sequence_loss(model(inputs), targets).item()
    assert final < .03 and final < initial
    assert torch.equal(model(inputs).argmax(-1), targets)
    try:
        sequence_loss(logits, targets, torch.zeros_like(targets, dtype=torch.bool))
    except ValueError:
        pass
    else:
        raise AssertionError("an empty loss mask was silently accepted")
    print(f"PASS: sequence probability {joint_probability:.6f}, token PPL {perplexity:.6f}")
    print("PASS: shifted labels, hand NLL, padding invariance, CE gradients, one SGD step")
    print(f"PASS: bigram fitted one document; NLL {initial:.6f} -> {final:.6f}")
    print("This fit checks the training loop, not held-out language quality.")


if __name__ == "__main__":
    verify()
