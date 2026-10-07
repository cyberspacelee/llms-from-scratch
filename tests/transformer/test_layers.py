import torch
import torch.nn.functional as F

from llms_from_scratch.transformer import model
from llms_from_scratch.transformer.layers import (
    GELUFeedForward,
    LayerNorm,
    RMSNorm,
    SwiGLUFeedForward,
    TransformerBlock,
    residual_gradient_norms,
    swiglu_hidden_dim,
)


def test_norms_match_reference():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 16) * 4 + 2
    ln = LayerNorm(16)
    torch.testing.assert_close(ln(x), F.layer_norm(x, (16,), eps=1e-5))
    rms = RMSNorm(16)
    torch.testing.assert_close(rms(x), F.rms_norm(x, (16,), eps=1e-5))
    torch.testing.assert_close(rms(x), model.RMSNorm(16)(x))
    torch.testing.assert_close(ln(x).mean(-1), torch.zeros(3, 5), atol=1e-5, rtol=0)


def test_swiglu_matches_model_and_param_budget():
    torch.manual_seed(0)
    d = 96
    ours = SwiGLUFeedForward(d)
    ref = model.SwiGLU(d, swiglu_hidden_dim(d))
    ref.load_state_dict(ours.state_dict())
    x = torch.randn(2, 4, d)
    torch.testing.assert_close(ours(x), ref(x))
    n_swiglu = sum(p.numel() for p in ours.parameters())
    n_gelu = sum(p.numel() for p in GELUFeedForward(d).parameters())
    assert abs(n_swiglu / n_gelu - 1) < 0.05  # 8/3 倍宽度让参数量与 4d 的 GELU FFN 持平


def test_blocks_preserve_shape():
    x = torch.randn(2, 6, 32)
    for arrangement in ("pre", "post", "plain"):
        assert TransformerBlock(32, 4, arrangement)(x).shape == x.shape


def test_residual_keeps_gradients_flowing():
    pre = residual_gradient_norms(6, "pre", std=0.02, seq_len=16)
    plain = residual_gradient_norms(6, "plain", seq_len=16)
    assert max(pre) / min(pre) < 1.1  # 残差：每层梯度量级相同
    assert max(plain) / min(plain) > 10  # 无残差：逐层连乘，量级相差悬殊
