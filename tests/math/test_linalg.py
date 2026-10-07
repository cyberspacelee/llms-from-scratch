import pytest
import torch

from llms_from_scratch.math.linalg import (
    attention_scores_einsum,
    attention_scores_matmul,
    broadcast_shape,
    cosine_similarity,
    linear,
    matmul_columns,
    matmul_dots,
    matmul_outer,
    matmul_rows,
    two_layer_mlp,
)

VIEWS = [matmul_dots, matmul_columns, matmul_outer, matmul_rows]


def test_cosine_similarity_matches_torch():
    u = torch.tensor([1.0, 2.0, 2.0])
    v = torch.tensor([2.0, 0.0, 1.0])
    expected = torch.nn.functional.cosine_similarity(u, v, dim=0)
    torch.testing.assert_close(cosine_similarity(u, v), expected)
    torch.testing.assert_close(cosine_similarity(u, 3 * u), torch.tensor(1.0))
    torch.testing.assert_close(cosine_similarity(u, -u), torch.tensor(-1.0))


@pytest.mark.parametrize("view", VIEWS)
def test_four_views_of_matmul_agree(view):
    torch.manual_seed(0)
    a = torch.randn(3, 4, dtype=torch.float64)
    b = torch.randn(4, 5, dtype=torch.float64)
    torch.testing.assert_close(view(a, b), a @ b)


def test_hand_example():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    expected = torch.tensor([[19.0, 22.0], [43.0, 50.0]])
    for view in VIEWS:
        torch.testing.assert_close(view(a, b), expected)


@pytest.mark.parametrize(
    "shapes",
    [
        ((8, 1, 6, 1), (7, 1, 5)),
        ((5, 4), (1,)),
        ((5, 4), (4,)),
        ((15, 3, 5), (15, 1, 5)),
        ((3,), ()),
    ],
)
def test_broadcast_shape_matches_torch(shapes):
    assert broadcast_shape(*shapes) == tuple(torch.broadcast_shapes(*shapes))


def test_broadcast_shape_rejects_mismatch():
    with pytest.raises(ValueError):
        broadcast_shape((2, 3), (4,))


def test_linear_and_mlp_match_nn():
    torch.manual_seed(0)
    x = torch.randn(2, 7, 4)  # (B, T, d_in)：前面的维度都当作批
    layer1, layer2 = torch.nn.Linear(4, 8), torch.nn.Linear(8, 3)
    torch.testing.assert_close(linear(x, layer1.weight, layer1.bias), layer1(x))
    reference = layer2(torch.relu(layer1(x)))
    out = two_layer_mlp(x, layer1.weight, layer1.bias, layer2.weight, layer2.bias)
    torch.testing.assert_close(out, reference)


def test_einsum_equals_batched_matmul():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 5, 4)
    k = torch.randn(2, 3, 5, 4)
    s = attention_scores_einsum(q, k)
    assert s.shape == (2, 3, 5, 5)
    torch.testing.assert_close(s, attention_scores_matmul(q, k))
