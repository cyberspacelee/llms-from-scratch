"""One supervised token through a frozen full-rank 6-by-4 layer."""

import io

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, base, rank=2, alpha=2):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, base.in_features, dtype=base.weight.dtype,
                                          device=base.weight.device) * 0.1)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=base.weight.dtype,
                                          device=base.weight.device))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T

    def merged_weight(self):
        return self.base.weight.detach() + self.scale * (self.B @ self.A)


def verify():
    torch.manual_seed(23)
    base = nn.Linear(4, 6, bias=False).double()
    with torch.no_grad():
        base.weight.copy_(torch.tensor([
            [1., 0., 0., 0.],
            [0., 0.5, 0., 0.],
            [0., 0., 1. / 3., 0.],
            [0., 0., 0., 0.25],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.],
        ], dtype=torch.float64))
    layer = LoRALinear(base)
    with torch.no_grad():
        layer.A.copy_(torch.tensor([
            [1., 0., 1., 0.],
            [0., 1., 0., 1.],
        ], dtype=torch.float64))
    x = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    target = torch.tensor([2])
    frozen = layer.base.weight.detach().clone()
    initial_A = layer.A.detach().clone()
    assert torch.linalg.matrix_rank(frozen) == 4
    torch.testing.assert_close(layer.A @ x, torch.tensor([4., 6.], dtype=torch.float64))
    torch.testing.assert_close(layer(x), layer.base(x))
    torch.testing.assert_close(layer(x), torch.tensor([1., 1., 1., 1., 0., 0.], dtype=torch.float64))
    initial_loss = F.cross_entropy(layer(x).unsqueeze(0), target)
    assert abs(initial_loss.item() - 1.555142) < 1e-6
    optimizer = torch.optim.SGD([layer.A, layer.B], lr=0.01)
    initial_loss.backward()
    g = torch.softmax(layer.base(x).detach(), dim=0)
    g[2] -= 1
    torch.testing.assert_close(layer.B.grad, torch.outer(g, layer.A.detach() @ x))
    torch.testing.assert_close(layer.A.grad, torch.zeros_like(layer.A))
    assert layer.base.weight.grad is None
    optimizer.step()
    torch.testing.assert_close(layer.B, -0.01 * torch.outer(g, initial_A @ x))
    torch.testing.assert_close(layer.A, initial_A)
    torch.testing.assert_close(layer(x), layer.base(x) - 0.52 * g)
    first_loss = F.cross_entropy(layer(x).unsqueeze(0), target)
    assert abs(first_loss.item() - 1.179401) < 1e-6
    assert first_loss < initial_loss
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(layer(x).unsqueeze(0), target).backward()
    assert layer.A.grad.abs().sum() > 0
    optimizer.step()
    torch.testing.assert_close(layer.base.weight, frozen, atol=0, rtol=0)
    torch.testing.assert_close(layer(x), x @ layer.merged_weight().T, atol=1e-12, rtol=1e-12)
    batch = torch.stack((x, 2 * x))
    torch.testing.assert_close(layer(batch), batch @ layer.merged_weight().T,
                               atol=1e-12, rtol=1e-12)
    assert torch.linalg.matrix_rank(layer.B @ layer.A) <= 2
    assert layer.A.numel() + layer.B.numel() == 20
    torch.testing.assert_close(layer.B @ (layer.A @ x), (layer.B @ layer.A) @ x)
    payload = {"A": layer.A.detach().clone(), "B": layer.B.detach().clone(),
               "rank": 2, "alpha": 2, "base_id": "teaching-linear-seed23"}
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=True)
    assert loaded["base_id"] == "teaching-linear-seed23"
    restored = LoRALinear(nn.Linear(4, 6, bias=False).double(), loaded["rank"], loaded["alpha"])
    with torch.no_grad():
        restored.base.weight.copy_(frozen)
        restored.A.copy_(loaded["A"])
        restored.B.copy_(loaded["B"])
    torch.testing.assert_close(restored(x), layer(x), atol=0, rtol=0)
    print(f"LoRA: initial CE {initial_loss.item():.6f}, after one step {first_loss.item():.6f}")
    print("LoRA: 24 frozen / 20 trainable; gradients, rank, merge and restore verified")


if __name__ == "__main__":
    verify()
