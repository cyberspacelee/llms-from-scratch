"""Explicit and absorbed implementations of the same one-head MLA operator."""
import math
import torch


def rotation(angle):
    return torch.tensor([[math.cos(angle), -math.sin(angle)],
                         [math.sin(angle), math.cos(angle)]], dtype=torch.float64)


def verify():
    torch.manual_seed(11)
    dtype = torch.float64
    latent = torch.randn(5, 3, dtype=dtype)
    up_k, up_v = torch.randn(2, 3, dtype=dtype), torch.randn(4, 3, dtype=dtype)
    out = torch.randn(6, 4, dtype=dtype)
    q_content = torch.randn(2, dtype=dtype)
    q_rope = rotation(4 * .3) @ torch.tensor([1., 2.], dtype=dtype)
    raw_rope = torch.randn(5, 2, dtype=dtype)
    keys_rope = torch.stack([rotation(j * .3) @ key for j, key in enumerate(raw_rope)])
    keys, values = latent @ up_k.T, latent @ up_v.T
    scores = (keys @ q_content + keys_rope @ q_rope) / math.sqrt(4)
    weights = scores.softmax(0)
    explicit = out @ (weights @ values)
    q_latent = up_k.T @ q_content
    absorbed_scores = (latent @ q_latent + keys_rope @ q_rope) / math.sqrt(4)
    absorbed = (out @ up_v) @ (absorbed_scores.softmax(0) @ latent)
    torch.testing.assert_close(scores, absorbed_scores)
    torch.testing.assert_close(explicit, absorbed)
    # Position-dependent rotation cannot be absorbed into one fixed query map.
    square_up = torch.tensor([[1., 2.], [0., 1.]], dtype=dtype)
    assert not torch.allclose(rotation(.3) @ square_up, square_up @ rotation(.3))
    assert 5 * (3 + 2) == 25  # Shared latent plus positional keys, per layer.
    print("MLA: explicit K/V = absorbed latent computation; cached elements = 25")


if __name__ == "__main__":
    verify()
