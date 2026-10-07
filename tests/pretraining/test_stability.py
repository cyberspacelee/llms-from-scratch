import math

import torch

from llms_from_scratch.pretraining.stability import (
    StabilityConfig,
    lm_loss,
    make_stable,
    qk_clip_,
    run_stability_experiment,
    soft_cap,
    spike_score,
    z_loss,
)
from llms_from_scratch.transformer.model import GPT, GPTConfig

torch.set_num_threads(1)
CFG = GPTConfig(vocab_size=32, context_length=16, d_model=32, n_layers=2, n_heads=4)


def test_z_loss_is_zero_when_normalized():
    logp = torch.randn(4, 10).log_softmax(-1)          # 已归一化：log Z = 0
    assert z_loss(logp).item() < 1e-10
    shifted = logp + 3.0                                # log Z = 3
    assert math.isclose(z_loss(shifted).item(), 9.0, rel_tol=1e-5)
    # 交叉熵对整体平移不变，只有 z-loss 能“看见”这个平移
    t = torch.randint(0, 10, (4,))
    assert torch.isclose(lm_loss(logp, t)[0], lm_loss(shifted, t)[0])


def test_soft_cap():
    x = torch.tensor([0.1, 10.0, 1e4])
    y = soft_cap(x, 30.0)
    assert abs(y[0] - 0.1) < 1e-4 and y[2] <= 30.0 and y[1] < 10.0


def test_stable_attention_matches_original():
    torch.manual_seed(0)
    model = GPT(CFG)
    x = torch.randint(0, 32, (2, 12))
    ref = model(x)
    torch.testing.assert_close(make_stable(model)(x), ref, atol=1e-5, rtol=1e-4)


def test_qk_clip_caps_max_logit_exactly():
    torch.manual_seed(0)
    model = make_stable(GPT(CFG))
    with torch.no_grad():
        for b in model.blocks:                       # 人为放大 W_q，制造大 logit
            b.attn.q_proj.weight *= 40
            b.attn.k_proj.weight *= 40
    x = torch.randint(0, 32, (2, 16))
    model(x)
    before = max(b.attn.max_logit.max().item() for b in model.blocks)
    assert before > 5.0
    assert qk_clip_(model, tau=5.0) > 0
    model(x)
    after = max(b.attn.max_logit.max().item() for b in model.blocks)
    assert after <= 5.0 + 1e-4                       # 同一输入：S_max 恰好被压到 τ


def test_high_lr_logit_growth_and_fixes():
    base = dict(lr=3e-2, steps=40, model=CFG)
    unstable = run_stability_experiment(StabilityConfig(**base))
    qk_norm = run_stability_experiment(StabilityConfig(qk_norm=True, **base))
    clipped = run_stability_experiment(StabilityConfig(qk_clip_tau=10.0, **base))
    assert unstable["max_logit"][-1] > 15           # 无保护：注意力 logit 一路增长
    assert unstable["max_logit"][-1] > 2 * qk_norm["max_logit"][-1]
    assert max(clipped["max_logit"][20:]) < 15      # QK-Clip 把它钉在 τ = 10 附近
    assert all(math.isfinite(v) for v in unstable["loss"])


def test_z_loss_keeps_log_partition_small():
    base = dict(lr=1e-2, steps=40, model=CFG)
    plain = run_stability_experiment(StabilityConfig(**base))
    zl = run_stability_experiment(StabilityConfig(z_coef=1e-2, **base))
    assert zl["log_z"][-1] < plain["log_z"][-1]


def test_spike_score():
    flat = [1.0 + 0.01 * ((-1) ** i) for i in range(200)]
    assert spike_score(flat, window=50) == 0.0
    flat[150] = 5.0
    assert 0 < spike_score(flat, window=50) < 0.02
