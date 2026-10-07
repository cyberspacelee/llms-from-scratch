import math

import numpy as np
import torch

from llms_from_scratch.gpu.online_softmax import (
    merge_states,
    online_max_sum,
    row_softmax_block_sim,
    softmax_online,
    softmax_three_pass,
)


def test_online_equals_three_pass():
    x = np.random.default_rng(0).standard_normal(50) * 10
    ref = torch.softmax(torch.tensor(x), 0).numpy()
    assert np.allclose(softmax_three_pass(x), ref)
    assert np.allclose(softmax_online(x), ref)


def test_large_values_do_not_overflow():
    x = np.array([1000.0, 1001.0, 999.0])
    assert np.allclose(softmax_online(x), torch.softmax(torch.tensor(x), 0).numpy())


def test_merge_matches_whole_and_is_associative():
    x = np.random.default_rng(1).standard_normal(30) * 5
    parts = [online_max_sum(x[i : i + 10]) for i in (0, 10, 20)]
    left = merge_states(*merge_states(*parts[0], *parts[1]), *parts[2])
    right = merge_states(*parts[0], *merge_states(*parts[1], *parts[2]))
    m, ell = online_max_sum(x)
    assert left[0] == right[0] == m
    assert math.isclose(left[1], ell) and math.isclose(right[1], ell)


def test_merge_with_empty_state():
    assert merge_states(-math.inf, 0.0, 2.0, 3.0) == (2.0, 3.0)
    assert online_max_sum(np.array([-math.inf, 1.0]))[1] == 1.0


def test_row_softmax_block_sim():
    x = np.random.default_rng(2).standard_normal((4, 100)) * 3
    ref = torch.softmax(torch.tensor(x), -1).numpy()
    assert np.allclose(row_softmax_block_sim(x, block_dim=32), ref)
