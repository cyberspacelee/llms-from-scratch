import math

import pytest
import torch

from llms_from_scratch.architecture.rope_scaling import (
    inv_freq,
    ntk_inv_freq,
    pi_inv_freq,
    rotary_factors,
    wavelength,
    yarn_inv_freq,
    yarn_mscale,
)
from llms_from_scratch.transformer.model import rope_frequencies


def test_rotary_factors_match_model():
    torch.testing.assert_close(rotary_factors(inv_freq(16), 20), rope_frequencies(16, 20),
                               atol=1e-5, rtol=1e-5)


def test_pi_maps_long_positions_into_trained_range():
    s = 4
    base, pi = inv_freq(64), pi_inv_freq(64, s)
    # 新位置 s·m 的旋转角 = 原位置 m 的旋转角
    torch.testing.assert_close(4000 * pi, 1000 * base)


def test_ntk_keeps_highest_and_interpolates_lowest():
    d, s = 64, 8
    base, ntk = inv_freq(d), ntk_inv_freq(d, s)
    assert ntk[0] == pytest.approx(base[0])
    assert ntk[-1] == pytest.approx(base[-1] / s)
    ratio = base / ntk
    assert torch.all(ratio[1:] >= ratio[:-1])  # 越低频插值越多


def test_yarn_piecewise():
    d, s, L = 128, 16, 4096
    base, yarn = inv_freq(d, 10000.0), yarn_inv_freq(d, s, L)
    r = L / wavelength(base)
    high, low = r > 32, r < 1
    assert high.any() and low.any()
    torch.testing.assert_close(yarn[high], base[high])
    torch.testing.assert_close(yarn[low], base[low] / s)
    mid = ~(high | low)
    assert torch.all((yarn[mid] <= base[mid]) & (yarn[mid] >= base[mid] / s))


def test_yarn_mscale():
    assert yarn_mscale(1.0) == 1.0
    assert yarn_mscale(32.0) == pytest.approx(0.1 * math.log(32) + 1)
