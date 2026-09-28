"""F5: one small Module, from registered parameters to masked token loss."""

import math

import torch
from torch import nn
import torch.nn.functional as F


class TinyWords(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 2, dtype=torch.float64)
        self.norm = nn.LayerNorm(2, eps=1.0, dtype=torch.float64)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(2, 4, bias=False, dtype=torch.float64)

    def forward(self, ids):
        x = self.embedding(ids)
        x = self.norm(x)
        x = self.dropout(x)
        return self.head(x)


def token_loss(logits, targets):
    """Mean over valid index targets; an all-ignored batch gives connected zero."""
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or logits.shape[-1] == 0:
        raise ValueError("Expected (B,T,V) logits and (B,T) targets")
    if targets.dtype != torch.long or targets.device != logits.device:
        raise ValueError("Targets must be same-device long indices")
    valid = targets != -100
    if ((targets[valid] < 0) | (targets[valid] >= logits.shape[-1])).any():
        raise ValueError("Valid target outside vocabulary")
    total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                            targets.reshape(-1), ignore_index=-100, reduction="sum")
    return total / valid.sum().clamp_min(1)


def verify():
    model = TinyWords()
    with torch.no_grad():
        model.embedding.weight.copy_(torch.tensor([[2., 0.], [0., 2.],
                                                   [2., 0.], [0., 0.]], dtype=torch.float64))
        model.head.weight.copy_(torch.tensor([[1., 0.], [0., 1.],
                                              [0., 0.], [0., 0.]], dtype=torch.float64))
    ids = torch.tensor([[0, 1, 2], [1, 0, 0]], dtype=torch.long)
    targets = torch.tensor([[0, 1, -100], [1, 0, -100]], dtype=torch.long)

    names = dict(model.named_parameters())
    assert set(names) == {"embedding.weight", "norm.weight", "norm.bias", "head.weight"}
    assert sum(p.numel() for p in model.parameters()) == 20
    assert set(model.state_dict()) == set(names)
    assert model.head.weight.shape == (4, 2)

    model.eval()
    assert not model.dropout.training
    logits = model(ids)
    s = 1 / math.sqrt(2)
    expected = torch.tensor([[[s, -s, 0., 0.], [-s, s, 0., 0.], [s, -s, 0., 0.]],
                             [[-s, s, 0., 0.], [s, -s, 0., 0.], [s, -s, 0., 0.]]],
                            dtype=torch.float64)
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(model.norm(model.embedding(ids))[0, 0],
                               torch.tensor([s, -s], dtype=torch.float64))
    per_position = F.cross_entropy(logits.transpose(1, 2), targets,
                                   ignore_index=-100, reduction="none")
    assert per_position.shape == (2, 3)
    torch.testing.assert_close(per_position[targets != -100],
                               torch.full((4,), math.log(math.exp(s) + math.exp(-s) + 2) - s,
                                          dtype=torch.float64))
    assert torch.equal(per_position[targets == -100], torch.zeros(2, dtype=torch.float64))
    loss = token_loss(logits, targets)
    torch.testing.assert_close(loss, per_position.sum() / 4)
    torch.testing.assert_close(loss, F.cross_entropy(logits.transpose(1, 2), targets))
    loss.backward()
    assert model.head.weight.grad is not None  # eval does not disable autograd.
    assert model.embedding.weight.grad[0].abs().sum() > 0
    torch.testing.assert_close(model.embedding.weight.grad[2], torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(model.embedding.weight.grad[3], torch.zeros(2, dtype=torch.float64))
    try:
        F.cross_entropy(logits, targets)
    except RuntimeError:
        pass  # (B,T,V) was mistaken for (N,C,...) and target shape no longer matches.
    else:
        raise AssertionError("Unmoved class axis should fail for T=3, V=4")

    model.zero_grad(set_to_none=True)
    zero = token_loss(model(ids), torch.full_like(targets, -100))
    assert zero.item() == 0 and zero.requires_grad
    zero.backward()
    torch.testing.assert_close(model.head.weight.grad, torch.zeros_like(model.head.weight))

    model.train()
    assert model.dropout.training
    with torch.random.fork_rng():
        torch.manual_seed(7)
        kept = model.dropout(torch.ones(20, dtype=torch.float64))
        assert ((kept == 0) | (kept == 2)).all()

    class HiddenLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(2, 2, dtype=torch.float64)]

        def forward(self, x):
            return self.layers[0](x)

    hidden = HiddenLayer()
    hidden(torch.ones(1, 2, dtype=torch.float64)).sum().backward()
    assert hidden.layers[0].weight.grad is not None
    assert not list(hidden.parameters()) and not hidden.state_dict()
    hidden.eval()
    assert hidden.layers[0].training
    print(f"modules_and_losses: module, logits and masked CE passed; torch {torch.__version__}")


if __name__ == "__main__":
    verify()
