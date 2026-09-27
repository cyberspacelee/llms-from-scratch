"""F5: CPU float64 checks for registration, layers and index-target losses."""
import torch
from torch import nn
import torch.nn.functional as F


class TinyWords(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 3, dtype=torch.float64)
        self.norm = nn.LayerNorm(3, dtype=torch.float64)
        self.blocks = nn.ModuleList([nn.Linear(3, 3, dtype=torch.float64) for _ in range(2)])
        self.dropout = nn.Dropout(0.25)
        self.head = nn.Linear(3, 4, dtype=torch.float64)
        self.gain = nn.Parameter(torch.ones((), dtype=torch.float64))
        self.register_buffer("offset", torch.zeros((), dtype=torch.float64))

    def forward(self, ids):
        x = self.norm(self.embedding(ids))
        for layer in self.blocks:
            x = x + F.relu(layer(x))
        return self.head(self.dropout(x)) * self.gain + self.offset


def token_cross_entropy(logits, targets, ignore_index=-100):
    """Index targets, no class weights/smoothing; all ignored returns connected zero."""
    if (logits.ndim != 3 or logits.shape[-1] < 1 or targets.shape != logits.shape[:2]
            or targets.dtype != torch.long or targets.device != logits.device
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError("Expected finite (B,T,V) logits and same-device long (B,T) targets")
    valid = targets != ignore_index
    if ((targets[valid] < 0) | (targets[valid] >= logits.shape[-1])).any():
        raise ValueError("Valid class index out of range")
    total = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
                            ignore_index=ignore_index, reduction="sum")
    return total / valid.sum().clamp_min(1)


def verify():
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = TinyWords()
        names = dict(model.named_parameters())
        assert "blocks.0.weight" in names and "gain" in names
        assert sum(p.numel() for p in model.parameters()) == 59
        assert "offset" in model.state_dict() and "offset" not in names
        assert "dropout" not in model.state_dict()
        ids = torch.tensor([[0, 1, 1], [3, 2, 0]], dtype=torch.long)
        assert model(ids).shape == (2, 3, 4)
        model.eval()
        assert all(not child.training for child in model.modules())
        model(ids).sum().backward()
        assert model.head.weight.grad is not None  # eval does not disable autograd.

        class PlainList(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = [nn.Linear(3, 3, dtype=torch.float64)]
                self.raw = torch.ones((), dtype=torch.float64, requires_grad=True)

            def forward(self, x):
                return self.layers[0](x) * self.raw

        plain = PlainList()
        assert list(plain.parameters()) == [] and len(plain.state_dict()) == 0
        plain(torch.ones(1, 3, dtype=torch.float64)).sum().backward()
        assert plain.layers[0].weight.grad is not None and plain.raw.grad is not None
        plain.eval()
        assert plain.layers[0].training  # The unregistered child is not traversed.
        plain.float()
        assert plain.layers[0].weight.dtype == torch.float64

        layer = nn.Linear(3, 2, dtype=torch.float64)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1., 2., 3.], [-1., 0., 1.]], dtype=torch.float64))
            layer.bias.copy_(torch.tensor([.5, -.5], dtype=torch.float64))
        x = torch.tensor([[1., 0., -1.], [2., 1., 0.]], dtype=torch.float64)
        torch.testing.assert_close(layer(x), x @ layer.weight.T + layer.bias)
        torch.testing.assert_close(layer(x), torch.tensor([[-1.5, -2.5], [4.5, -2.5]], dtype=torch.float64))
        sequence = nn.Sequential(layer, nn.ReLU())
        torch.testing.assert_close(sequence(x), F.relu(layer(x)))

        embedding = nn.Embedding(4, 3, dtype=torch.float64)
        selected = torch.tensor([1, 1, 3], dtype=torch.long)
        embedding(selected).sum().backward()
        expected = torch.tensor([[0., 0., 0.], [2., 2., 2.], [0., 0., 0.], [1., 1., 1.]], dtype=torch.float64)
        torch.testing.assert_close(embedding.weight.grad, expected)
        padding = nn.Embedding(4, 3, padding_idx=0, dtype=torch.float64)
        padding(torch.tensor([0, 1, 0], dtype=torch.long)).sum().backward()
        assert torch.equal(padding.weight.grad[0], torch.zeros(3, dtype=torch.float64))

        norm = nn.LayerNorm(3, eps=1e-5, dtype=torch.float64)
        mean, variance = x.mean(-1, keepdim=True), x.var(-1, correction=0, keepdim=True)
        torch.testing.assert_close(norm(x), (x - mean) / torch.sqrt(variance + norm.eps))
        dropout = nn.Dropout(0.5)
        ones = torch.ones(64, dtype=torch.float64, requires_grad=True)
        dropped = dropout(ones)
        assert ((dropped == 0) | (dropped == 2)).all()
        assert (dropped == 0).any() and (dropped == 2).any()
        dropout.eval()
        torch.testing.assert_close(dropout(ones), ones)
        assert dropout(ones).requires_grad
        torch.testing.assert_close(F.gelu(x), 0.5 * x * (1 + torch.erf(x / 2**0.5)))

        logits = torch.tensor([[[2., 1., 0.], [0., 1., 2.], [1., 1., 1.]]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[0, 2, -100]], dtype=torch.long)
        flat = logits.reshape(-1, 3)
        manual = -F.log_softmax(flat, dim=-1)[torch.arange(2), torch.tensor([0, 2])].sum() / 2
        loss = token_cross_entropy(logits, targets)
        torch.testing.assert_close(loss, manual)
        torch.testing.assert_close(loss, F.cross_entropy(logits.transpose(1, 2), targets))
        none = F.cross_entropy(flat, targets.flatten(), ignore_index=-100, reduction="none")
        assert none[-1].item() == 0
        torch.testing.assert_close(none.sum() / 2, loss)
        probability = F.softmax(flat, dim=-1)
        torch.testing.assert_close(probability.sum(-1), torch.ones(3, dtype=torch.float64))
        assert not torch.allclose(F.cross_entropy(probability, targets.flatten()), loss)
        loss.backward()
        torch.testing.assert_close(logits.grad[0, 2], torch.zeros(3, dtype=torch.float64))
        all_ignored = torch.full_like(targets, -100)
        assert torch.isnan(F.cross_entropy(flat, all_ignored.flatten(), reduction="mean"))
        logits.grad = None
        zero = token_cross_entropy(logits, all_ignored)
        assert zero.item() == 0 and zero.requires_grad
        zero.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
        logits.grad = None
        empty_loss = token_cross_entropy(logits[:, :0], targets[:, :0])
        assert empty_loss.item() == 0
        try:
            token_cross_entropy(logits, targets.double())
        except ValueError:
            pass
        else:
            raise AssertionError("Floating-point index targets must be rejected")
        print(f"modules_and_losses: registration, layer shapes, repeat gradients, modes and masked CE passed; torch {torch.__version__}")


if __name__ == "__main__":
    verify()
