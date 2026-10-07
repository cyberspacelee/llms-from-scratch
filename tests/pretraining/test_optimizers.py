import pytest
import torch
import torch.nn.functional as F

from llms_from_scratch.pretraining.optimizers import (
    CUBIC_COEFFS,
    AdamW,
    HybridOptimizer,
    Muon,
    newton_schulz,
    ns_scalar_map,
    optimizer_state_bytes,
    polar_factor,
    split_muon_params,
)
from llms_from_scratch.pretraining.stability import markov_sampler
from llms_from_scratch.transformer.model import GPT, GPTConfig


def _run(opt_cls, steps=5, **kw):
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(4, 3))
    opt = opt_cls([w], **kw)
    for t in range(steps):
        opt.zero_grad()
        loss = ((w - t) ** 2).sum() + w.pow(3).sum()
        loss.backward()
        opt.step()
    return w.detach()


def test_adamw_matches_torch():
    ours = _run(AdamW, lr=0.1, weight_decay=0.1)
    ref = _run(torch.optim.AdamW, lr=0.1, weight_decay=0.1)
    torch.testing.assert_close(ours, ref)


def test_l2_variant_matches_torch_adam_weight_decay():
    ours = _run(AdamW, lr=0.1, weight_decay=0.1, decoupled=False)
    ref = _run(torch.optim.Adam, lr=0.1, weight_decay=0.1)
    torch.testing.assert_close(ours, ref)
    assert not torch.allclose(ours, _run(AdamW, lr=0.1, weight_decay=0.1))


def test_newton_schulz_approximates_polar_factor():
    torch.manual_seed(0)
    G = torch.randn(64, 32)
    ortho = newton_schulz(G)
    s = torch.linalg.svdvals(ortho)
    assert s.min() > 0.6 and s.max() < 1.25          # 五次系数只保证奇异值落在 ~[0.7, 1.2]
    # 奇异向量方向与精确的 UVᵀ 一致：O 与 UVᵀ 的余弦相似度接近 1
    P = polar_factor(G)
    cos = (ortho * P).sum() / (ortho.norm() * P.norm())
    assert cos > 0.97


def test_cubic_newton_schulz_converges_exactly_but_slowly():
    torch.manual_seed(0)
    G = torch.randn(16, 16)
    P = polar_factor(G)
    err5 = (newton_schulz(G, steps=5, coeffs=CUBIC_COEFFS) - P).norm() / P.norm()
    err60 = (newton_schulz(G, steps=60, coeffs=CUBIC_COEFFS) - P).norm() / P.norm()
    assert err60 < 1e-3 < err5


def test_scalar_map_pushes_small_singular_values_up():
    assert ns_scalar_map(1e-3) > 0.4          # 斜率 a≈3.44：5 步把 1e-3 放大约 3.44^5 ≈ 480 倍
    assert 0.6 < ns_scalar_map(1e-2) < 1.25
    assert 0.6 < ns_scalar_map(0.5) < 1.25
    assert ns_scalar_map(1e-3, coeffs=CUBIC_COEFFS) < 0.01


def test_muon_rejects_vectors_and_matches_rms():
    with pytest.raises(ValueError):
        Muon([torch.nn.Parameter(torch.zeros(3))])
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(128, 64))
    before = w.detach().clone()
    opt = Muon([w], lr=1.0, nesterov=False)
    w.grad = torch.randn(128, 64)
    opt.step()
    rms = (w.detach() - before).pow(2).mean().sqrt().item()
    assert 0.14 < rms < 0.26                    # RMS 匹配：更新的 RMS ≈ 0.2


def test_split_params():
    model = GPT(GPTConfig(vocab_size=32, context_length=8, d_model=32, n_layers=1, n_heads=2))
    muon, adamw = split_muon_params(model)
    assert all(p.ndim == 2 for p in muon)
    assert any(p is model.embed.weight for p in adamw)
    assert len(muon) + len(adamw) == len(list(model.parameters()))


def test_hybrid_muon_trains_small_gpt():
    torch.manual_seed(0)
    torch.set_num_threads(1)
    cfg = GPTConfig(vocab_size=32, context_length=16, d_model=32, n_layers=2, n_heads=2)
    model = GPT(cfg)
    opt = HybridOptimizer(model, lr=3e-3, weight_decay=0.0)
    sample = markov_sampler(cfg.vocab_size, seed=0)
    losses = []
    for _ in range(40):
        x = sample(16, cfg.context_length)
        loss = F.cross_entropy(model(x[:, :-1]).flatten(0, 1), x[:, 1:].flatten())
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert sum(losses[-5:]) / 5 < losses[0] - 0.3


def test_optimizer_state_bytes():
    assert optimizer_state_bytes(7_000_000_000, "adamw") == 56_000_000_000
    assert optimizer_state_bytes(10, "muon") == 40
    assert optimizer_state_bytes(10, "sgd") == 0
