import pytest
import torch

from llms_from_scratch.pytorch.tensors import (
    as_strided_reference,
    attention_scores_einsum,
    broadcast_shape,
    contiguous_strides,
    element_offset,
    is_contiguous,
    merge_heads,
    rearrange,
    shares_storage,
    split_heads,
)


def test_offset_matches_torch_layout():
    x = torch.arange(24.0).reshape(2, 3, 4)
    storage = x.reshape(-1)
    for view in (x, x.transpose(0, 2), x[:, 1:, ::2], x.permute(1, 2, 0)):
        index = tuple(n - 1 for n in view.shape)
        pos = element_offset(index, view.stride(), view.storage_offset())
        assert storage[pos] == view[index]
        rebuilt = as_strided_reference(storage, view.shape, view.stride(), view.storage_offset())
        torch.testing.assert_close(rebuilt, view)
        strided = torch.as_strided(storage, view.shape, view.stride(), view.storage_offset())
        assert torch.equal(strided, view)


def test_contiguity_rules():
    x = torch.arange(12.0).reshape(3, 4)
    assert contiguous_strides(x.shape) == x.stride() == (4, 1)
    for view in (x, x.T, x[:, :2], x[1:], x.unsqueeze(1), x.T.contiguous()):
        assert is_contiguous(view.shape, view.stride()) == view.is_contiguous()


def test_views_share_storage_and_copies_do_not():
    x = torch.arange(12.0).reshape(3, 4)
    assert shares_storage(x, x.view(4, 3))
    assert shares_storage(x, x.T)
    assert shares_storage(x, x[1:, ::2])  # 基本切片是视图
    assert not shares_storage(x, x[[0, 2]])  # 高级索引会复制
    assert not shares_storage(x, x[x > 3])  # 布尔掩码也会复制
    assert not shares_storage(x, x.T.reshape(-1))  # 不连续时 reshape 必须复制
    assert shares_storage(x, x.reshape(-1))  # 连续时 reshape 就是 view
    with pytest.raises(RuntimeError):
        x.T.view(-1)
    y = x.T
    y[0, 1] = -1.0  # 通过视图写入会改到原张量
    assert x[1, 0] == -1.0


@pytest.mark.parametrize(
    "a,b", [((3, 1), (1, 4)), ((2, 3, 4), (4,)), ((5, 1, 3), (2, 1)), ((1,), (7, 1))]
)
def test_broadcast_shape_matches_torch(a, b):
    assert broadcast_shape(a, b) == tuple(torch.broadcast_shapes(a, b))


def test_broadcast_rejects_mismatch():
    with pytest.raises(ValueError):
        broadcast_shape((3, 4), (3,))


def test_broadcast_bug_column_vs_row():
    # 经典 bug：y 形状 (N,)，pred 形状 (N, 1)，相减广播成 (N, N)
    pred, y = torch.zeros(5, 1), torch.arange(5.0)
    assert (pred - y).shape == (5, 5)
    assert (pred.squeeze(-1) - y).shape == (5,)


def test_rearrange_split_merge_heads():
    x = torch.randn(2, 5, 12)
    heads = rearrange(x, "b t (h d) -> b h t d", h=3)
    torch.testing.assert_close(heads, split_heads(x, 3))
    assert shares_storage(heads, x)
    back = rearrange(heads, "b h t d -> b t (h d)")
    torch.testing.assert_close(back, x)
    torch.testing.assert_close(merge_heads(heads), x)
    torch.testing.assert_close(rearrange(x, "b t c -> (b t) c"), x.reshape(10, 12))


def test_rearrange_errors():
    x = torch.randn(2, 6)
    with pytest.raises(ValueError):
        rearrange(x, "a (b c) -> a b c")
    with pytest.raises(ValueError):
        rearrange(x, "a b -> a")


def test_einsum_scores_match_matmul():
    q, k = torch.randn(2, 3, 4, 8), torch.randn(2, 3, 5, 8)
    torch.testing.assert_close(attention_scores_einsum(q, k), q @ k.transpose(-2, -1))


def test_dtype_conversion_and_promotion():
    x = torch.arange(4, dtype=torch.float32)
    assert x.to(torch.float32) is x  # 目标相同时不复制
    h = x.to(torch.bfloat16)
    assert h.element_size() == 2 and x.element_size() == 4
    assert (torch.tensor([1, 2]) / 2).dtype == torch.float32  # 整数真除法提升为浮点
    assert (torch.ones(2, dtype=torch.bfloat16) + torch.ones(2)).dtype == torch.float32
