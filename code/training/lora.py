"""A frozen linear base with a low-rank update and exact merge check."""

import io

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base, rank=2, alpha=2):
        super().__init__()
        if rank <= 0 or alpha <= 0:
            raise ValueError("Rank and alpha must be positive")
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
    layer = LoRALinear(nn.Linear(4, 6, bias=False).double())
    x = torch.randn(5, 4, dtype=torch.float64)
    target = torch.randn(5, 6, dtype=torch.float64)
    frozen = layer.base.weight.detach().clone()
    torch.testing.assert_close(layer(x), layer.base(x))
    optimizer = torch.optim.SGD([layer.A, layer.B], lr=0.05)
    ((layer(x) - target).square().mean()).backward()
    assert layer.A.grad.count_nonzero() == 0
    assert layer.B.grad.abs().sum() > 0 and layer.base.weight.grad is None
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ((layer(x) - target).square().mean()).backward()
    assert layer.A.grad.abs().sum() > 0
    optimizer.step()
    torch.testing.assert_close(layer.base.weight, frozen, atol=0, rtol=0)
    torch.testing.assert_close(layer(x), x @ layer.merged_weight().T, atol=1e-12, rtol=1e-12)
    assert torch.linalg.matrix_rank(layer.B @ layer.A) <= 2
    assert layer.A.numel() + layer.B.numel() == 20
    hand_A = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]], dtype=torch.float64)
    hand_B = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 0.], [0., -1.], [1., -1.]], dtype=torch.float64)
    hand_x = torch.tensor([1., 2., 3., 4.], dtype=torch.float64)
    torch.testing.assert_close(hand_B @ (hand_A @ hand_x),
                               torch.tensor([4., 6., 10., -4., -6., -2.], dtype=torch.float64))
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
    print("LoRA: 24 frozen / 20 trainable parameters; initial and trained gradients verified")
    print("LoRA: merged outputs, hand matrix example and adapter-only serialization verified")


if __name__ == "__main__":
    verify()
