import numpy as np
import pytest

from llms_from_scratch.pretraining.scaling import (
    EPOCH_REFIT,
    HOFFMANN,
    ChinchillaParams,
    chinchilla_loss,
    compute_optimal,
    effective_data,
    fit_chinchilla,
    fit_power_law,
    isoflop_analysis,
    training_flops,
)


def test_compute_optimal_satisfies_budget_and_is_minimum():
    C = 5.76e23                                           # Chinchilla 的训练预算
    N, D = compute_optimal(C, EPOCH_REFIT)
    assert training_flops(N, D) == pytest.approx(C)
    best = chinchilla_loss(N, D)
    for f in (0.8, 1.25):
        assert chinchilla_loss(N * f, C / (6 * N * f)) > best
    assert 15 < D / N < 30                                # 修正后的拟合给出约 20 token/参数


def test_hoffmann_fit_implies_more_tokens_per_param():
    N, D = compute_optimal(5.76e23, HOFFMANN)
    assert D / N > 40


def test_fit_power_law_recovers_parameters():
    x = np.logspace(1, 5, 20)
    k, a = fit_power_law(x, 3.0 * x**-0.25)
    assert k == pytest.approx(3.0) and a == pytest.approx(-0.25)


def test_fit_chinchilla_recovers_known_parameters():
    true = ChinchillaParams(E=1.8, A=500.0, B=2000.0, alpha=0.35, beta=0.37)
    N, D = np.meshgrid(np.logspace(7, 10, 6), np.logspace(9, 12, 6))
    N, D = N.ravel(), D.ravel()
    fit = fit_chinchilla(N, D, chinchilla_loss(N, D, true))
    assert fit.alpha == pytest.approx(true.alpha, abs=0.02)
    assert fit.beta == pytest.approx(true.beta, abs=0.02)
    assert fit.E == pytest.approx(true.E, abs=0.05)


def test_isoflop_recovers_exponent():
    budgets = np.logspace(18, 22, 5)
    _, (_, a) = isoflop_analysis(budgets, EPOCH_REFIT, noise=0.002)
    expected = EPOCH_REFIT.beta / (EPOCH_REFIT.alpha + EPOCH_REFIT.beta)
    assert a == pytest.approx(expected, abs=0.03)


def test_effective_data_diminishing_returns():
    U = 1.0
    assert effective_data(U, 1) == U
    assert effective_data(U, 4) > 3.6                      # 4 轮几乎和 4 倍独特数据一样
    assert effective_data(U, 100) < 16.5                   # 上限 U·(1 + R*)
