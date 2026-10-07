import torch

from llms_from_scratch.architecture.vision import (
    MLPProjector,
    PatchEmbed,
    PatchMerger,
    mrope_frequencies,
    mrope_position_ids,
    patchify,
    splice_image_tokens,
)
from llms_from_scratch.transformer.model import rope_frequencies


def test_patch_embed_is_linear_on_patches():
    torch.manual_seed(0)
    pe = PatchEmbed(3, 4, 16)
    img = torch.randn(2, 3, 8, 12)
    out = pe(img)
    assert out.shape == (2, 6, 16)  # (8/4)·(12/4) = 6 个 patch
    W = pe.proj.weight.reshape(16, -1)
    torch.testing.assert_close(out, patchify(img, 4) @ W.t() + pe.proj.bias, atol=1e-5, rtol=1e-5)


def test_patchify_row_major_order():
    img = torch.arange(16.0).view(1, 1, 4, 4)
    p = patchify(img, 2)
    assert p[0, 1].tolist() == [2.0, 3.0, 6.0, 7.0]  # 第 0 行第 1 列的 patch


def test_merger_and_projector_shapes():
    x = torch.randn(2, 6 * 8, 32)
    assert PatchMerger(32, 64)(x, 6, 8).shape == (2, 12, 64)
    assert MLPProjector(32, 64)(x).shape == (2, 48, 64)


def test_merger_groups_spatial_neighbours():
    merger = PatchMerger(1, 4)
    x = torch.arange(16.0).view(1, 16, 1)  # 4×4 网格，值 = 行优先编号
    captured = {}
    merger.mlp.register_forward_pre_hook(lambda m, a: captured.setdefault("x", a[0]))
    merger(x, 4, 4)
    assert captured["x"][0, 0].tolist() == [0.0, 1.0, 4.0, 5.0]  # 左上角 2×2 邻域
    assert captured["x"][0, 1].tolist() == [2.0, 3.0, 6.0, 7.0]


def test_mrope_position_ids():
    ids = mrope_position_ids([("text", 2), ("image", 2, 3), ("text", 2)])
    assert ids.shape == (3, 10)
    assert ids[:, :2].tolist() == [[0, 1]] * 3
    assert ids[0, 2:8].tolist() == [2] * 6  # 图像：时间分量不变
    assert ids[1, 2:8].tolist() == [2, 2, 2, 3, 3, 3]  # 高 = 起点 + 行号
    assert ids[2, 2:8].tolist() == [2, 3, 4, 2, 3, 4]  # 宽 = 起点 + 列号
    assert ids[:, 8:].tolist() == [[5, 6]] * 3  # 之后的文本从 max + 1 = 5 继续


def test_mrope_reduces_to_rope_for_text():
    ids = mrope_position_ids([("text", 12)])
    torch.testing.assert_close(mrope_frequencies(ids, 32, (4, 6, 6)), rope_frequencies(32, 12))


def test_splice():
    emb = torch.zeros(5, 2)
    mask = torch.tensor([False, True, True, False, False])
    out = splice_image_tokens(emb, mask, torch.ones(2, 2))
    assert out[:, 0].tolist() == [0, 1, 1, 0, 0]
