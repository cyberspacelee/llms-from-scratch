import torch

from llms_from_scratch.architecture.hyper_connections import (
    HyperConnection,
    ManifoldHyperConnection,
    collapse_streams,
    composite_gain,
    expand_streams,
    sinkhorn,
)
from llms_from_scratch.transformer.model import RMSNorm, SwiGLU


def test_sinkhorn_is_doubly_stochastic():
    torch.manual_seed(0)
    M = sinkhorn(torch.randn(5, 4, 4) * 3, n_iter=300)
    assert torch.all(M >= 0)
    torch.testing.assert_close(M.sum(-1), torch.ones(5, 4), atol=1e-5, rtol=0)
    torch.testing.assert_close(M.sum(-2), torch.ones(5, 4), atol=1e-5, rtol=0)


def test_doubly_stochastic_closure_and_norm():
    torch.manual_seed(0)
    mats = list(sinkhorn(torch.randn(60, 4, 4) * 2, n_iter=100))
    P = torch.linalg.multi_dot(mats)
    torch.testing.assert_close(P.sum(-1), torch.ones(4), atol=1e-4, rtol=0)  # 乘积仍是双随机
    assert torch.linalg.matrix_norm(mats[0], ord=2) <= 1 + 1e-4  # 谱范数不超过 1
    assert composite_gain(mats) < 1 + 1e-3


def test_unconstrained_mixing_explodes_with_depth():
    torch.manual_seed(0)
    mats = [torch.eye(4) + 0.2 * torch.randn(4, 4) for _ in range(60)]
    assert composite_gain(mats) > 50  # 每层只偏离单位阵一点，60 层后就放大了几十倍以上


def test_residual_mixing_preserves_stream_sum():
    torch.manual_seed(0)
    H = sinkhorn(torch.randn(4, 4))
    X = torch.randn(4, 8)
    torch.testing.assert_close((H @ X).sum(0), X.sum(0), atol=1e-5, rtol=1e-5)


def test_hc_with_one_stream_is_plain_residual():
    torch.manual_seed(0)
    norm, ffn = RMSNorm(16), SwiGLU(16, 32)
    f = lambda x: ffn(norm(x))  # noqa: E731
    hc = HyperConnection(n=1, layer=0, fn=f)
    x = torch.randn(2, 5, 16)
    out = collapse_streams(hc(expand_streams(x, 1)))
    torch.testing.assert_close(out, x + f(x))


def test_hc_init_equals_prenorm_residual_on_each_stream():
    torch.manual_seed(0)
    f = SwiGLU(16, 32)
    hc = HyperConnection(n=4, layer=2, fn=f)
    X = torch.randn(3, 4, 16)
    torch.testing.assert_close(hc(X), X + f(X[:, 2]).unsqueeze(1))


def test_mhc_coefficients_and_shapes():
    torch.manual_seed(0)
    norm, ffn = RMSNorm(16), SwiGLU(16, 32)
    mhc = ManifoldHyperConnection(4, 16, lambda x: ffn(norm(x)))
    with torch.no_grad():
        mhc.w.weight.normal_()
    X = expand_streams(torch.randn(2, 5, 16), 4)
    h_pre, h_post, h_res = mhc.coefficients(X)
    assert h_pre.shape == (2, 5, 4) and h_res.shape == (2, 5, 4, 4)
    assert torch.all((h_pre > 0) & (h_pre < 1)) and torch.all((h_post > 0) & (h_post < 2))
    # 20 次迭代后列和精确为 1（最后一步是列归一化），行和只是近似为 1
    torch.testing.assert_close(h_res.sum(-2), torch.ones(2, 5, 4), atol=1e-5, rtol=0)
    torch.testing.assert_close(h_res.sum(-1), torch.ones(2, 5, 4), atol=1e-2, rtol=0)
    assert mhc(X).shape == X.shape
