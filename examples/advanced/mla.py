"""Three-token, two-head MLA: explicit K/V and absorbed decode agree."""

import math

import torch

DTYPE = torch.float64


def rotate(vector: torch.Tensor, position: int) -> torch.Tensor:
    angle = position * math.pi / 2
    matrix = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]], dtype=DTYPE
    )
    return matrix @ vector


def verify() -> None:
    # The first two coordinates are the joint KV latent; the third feeds RoPE.
    hidden = torch.tensor([[1, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=DTYPE)
    down = torch.tensor([[1, 0, 0], [0, 1, 0]], dtype=DTYPE)
    key_position = torch.tensor([[0, 0, 1], [0, 0, 0]], dtype=DTYPE)
    latent = hidden @ down.T
    rope_keys = torch.stack([rotate(key_position @ x, j) for j, x in enumerate(hidden)])

    up_keys = torch.tensor([[[1, 1], [0, 1]], [[1, 0], [1, 1]]], dtype=DTYPE)
    up_values = torch.tensor([[[1, 0], [0, 2]], [[0, 1], [1, 1]]], dtype=DTYPE)
    content_queries = torch.tensor([[2, 1], [1, 2]], dtype=DTYPE)
    rope_queries = torch.stack(
        [
            rotate(torch.tensor([1.0, 0.0], dtype=DTYPE), 2),
            rotate(torch.tensor([0.0, 1.0], dtype=DTYPE), 2),
        ]
    )
    output = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=DTYPE)

    # region absorbed_decode
    explicit_heads = []
    absorbed_heads = []
    for h in range(2):
        keys = latent @ up_keys[h].T
        values = latent @ up_values[h].T
        explicit_scores = (keys @ content_queries[h] + rope_keys @ rope_queries[h]) / 2
        absorbed_query = up_keys[h].T @ content_queries[h]
        absorbed_scores = (latent @ absorbed_query + rope_keys @ rope_queries[h]) / 2
        torch.testing.assert_close(explicit_scores, absorbed_scores)
        weights = explicit_scores.softmax(0)
        explicit_heads.append(weights @ values)
        absorbed_heads.append(up_values[h] @ (weights @ latent))
        expected = ([1, 3, 6], [3, 1, 5])[h]
        torch.testing.assert_close(explicit_scores * 2, torch.tensor(expected, dtype=DTYPE))
        print(f"head {h}: unscaled scores={expected}, weights={weights.tolist()}")

    explicit = output @ torch.cat(explicit_heads)
    absorbed = output @ torch.cat(absorbed_heads)
    torch.testing.assert_close(explicit, absorbed)
    assert latent.shape == (3, 2) and rope_keys.shape == (3, 2)
    assert 3 * (2 + 2) == 12  # Only latent and shared RoPE key persist per layer.

    # endregion absorbed_decode
    # An ordinary up-projection cannot generally pass through position rotation.
    nonsymmetric = torch.tensor([[1.0, 2.0], [0.0, 1.0]], dtype=DTYPE)
    basis = torch.tensor([1.0, 0.0], dtype=DTYPE)
    assert not torch.allclose(rotate(nonsymmetric @ basis, 1), nonsymmetric @ rotate(basis, 1))
    print(f"output={explicit.tolist()}, cached elements=12; explicit=absorbed")


if __name__ == "__main__":
    verify()
