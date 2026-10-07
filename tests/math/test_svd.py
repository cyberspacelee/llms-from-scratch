import math

import torch

from llms_from_scratch.math.svd import (
    condition_number,
    eckart_young_error,
    low_rank_approx,
    newton_schulz,
    polar_factor,
    power_iteration,
    spectral_norm,
)

D = torch.float64


def rand(*shape, seed=0):
    return torch.randn(*shape, dtype=D, generator=torch.Generator().manual_seed(seed))


def test_power_iteration_finds_top_eigenpair():
    a = torch.tensor([[2.0, 1.0], [1.0, 2.0]], dtype=D)  # 特征值 3 与 1
    lam, v = power_iteration(a)
    assert math.isclose(lam, 3.0, rel_tol=1e-10)
    torch.testing.assert_close(v.abs(), torch.full((2,), 2**-0.5, dtype=D))


def test_eigh_and_svd_reconstruct():
    m = rand(5, 5)
    s = m + m.T
    lam, q = torch.linalg.eigh(s)
    torch.testing.assert_close(q @ torch.diag(lam) @ q.T, s)
    torch.testing.assert_close(q.T @ q, torch.eye(5, dtype=D))
    a = rand(6, 4, seed=1)
    u, sig, vh = torch.linalg.svd(a, full_matrices=False)
    torch.testing.assert_close(u @ torch.diag(sig) @ vh, a)
    # AᵀA 的特征值是奇异值的平方
    torch.testing.assert_close(torch.linalg.eigvalsh(a.T @ a).flip(0), sig**2)


def test_spectral_norm():
    a = rand(7, 4)
    assert math.isclose(spectral_norm(a, steps=500), torch.linalg.matrix_norm(a, 2).item(), rel_tol=1e-8)


def test_eckart_young():
    a = rand(8, 6)
    for k in range(1, 6):
        ak = low_rank_approx(a, k)
        assert torch.linalg.matrix_rank(ak).item() == k
        fro, spec = eckart_young_error(a, k)
        assert math.isclose(torch.linalg.matrix_norm(a - ak).item(), fro, rel_tol=1e-10)
        assert math.isclose(torch.linalg.matrix_norm(a - ak, 2).item(), spec, rel_tol=1e-10)
        # 随机的秩 k 近似都不会更好
        for seed in range(5):
            b = rand(8, k, seed=10 + seed) @ rand(k, 6, seed=20 + seed)
            best_scale = (a * b).sum() / (b * b).sum()  # 即使允许最优缩放
            assert torch.linalg.matrix_norm(a - best_scale * b).item() >= fro


def test_condition_number():
    a = torch.diag(torch.tensor([10.0, 1.0, 0.1], dtype=D))
    assert math.isclose(condition_number(a), 100.0)
    q, _ = torch.linalg.qr(rand(4, 4))
    assert math.isclose(condition_number(q), 1.0, rel_tol=1e-10)


def test_polar_factor_and_newton_schulz():
    g = rand(6, 4)
    o = polar_factor(g)
    torch.testing.assert_close(o.T @ o, torch.eye(4, dtype=D))
    # UVᵀ 是离 G 最近的半正交矩阵，且与 G 的内积最大
    torch.testing.assert_close(torch.linalg.svdvals(o), torch.ones(4, dtype=D))
    torch.testing.assert_close(newton_schulz(g, steps=40), o, atol=1e-8, rtol=0)
    torch.testing.assert_close(newton_schulz(g.T, steps=40), o.T, atol=1e-8, rtol=0)
