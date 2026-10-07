import torch

from llms_from_scratch.architecture.mla import (
    MLACache,
    MLAConfig,
    MultiHeadLatentAttention,
    mla_cache_elems_per_token,
)


def test_absorbed_equals_naive_full_sequence():
    torch.manual_seed(0)
    mla = MultiHeadLatentAttention(MLAConfig())
    x = torch.randn(2, 12, 64)
    torch.testing.assert_close(mla(x, absorb=True), mla(x, absorb=False), atol=1e-5, rtol=1e-4)


def test_absorbed_decode_with_latent_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = MLAConfig(q_rank=0)
    mla = MultiHeadLatentAttention(cfg)
    x = torch.randn(2, 10, 64)
    full = mla(x, absorb=False)
    cache = MLACache(cfg, 2)
    outs = [mla(x[:, :6], cache, 0, absorb=False)]
    outs += [mla(x[:, t:t + 1], cache, t, absorb=True) for t in range(6, 10)]
    torch.testing.assert_close(torch.cat(outs, 1), full, atol=1e-5, rtol=1e-4)
    assert cache.c_kv.shape[-1] + cache.k_rope.shape[-1] == mla_cache_elems_per_token(cfg)


def test_rope_on_upprojected_keys_would_break_absorption():
    # 若把 RoPE 加在 W_UK c 上，分数里出现依赖位置 j 的旋转 R_j，
    # 它夹在 q^T 与 W_UK 之间，一般与 W_UK 不交换，因此无法预先合并
    torch.manual_seed(0)
    W = torch.randn(4, 4)
    angle = torch.tensor(0.7)
    R = torch.eye(4)
    R[:2, :2] = torch.tensor([[angle.cos(), -angle.sin()], [angle.sin(), angle.cos()]])
    assert not torch.allclose(R @ W, W @ R)
