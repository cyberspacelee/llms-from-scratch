"""One four-document comparison: token NLL, windows, and paired task scores."""

import math

import torch


def paired_bootstrap(first, second, repeats=2000, seed=17):
    first, second = (
        torch.as_tensor(first, dtype=torch.float64),
        torch.as_tensor(second, dtype=torch.float64),
    )
    if first.ndim != 1 or first.shape != second.shape or first.numel() < 2 or repeats < 1:
        raise ValueError("need two matching vectors and a positive repeat count")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("paired observations must be finite")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(first.numel(), (repeats, first.numel()), generator=generator)
    differences = (second - first)[indices].mean(-1)
    lower, upper = differences.quantile(torch.tensor([0.025, 0.975], dtype=torch.float64))
    return dict(
        mean=(second - first).mean().item(),
        lower=lower.item(),
        upper=upper.item(),
        seed=seed,
        repeats=repeats,
    )


def target_windows(token_count, context_length, stride):
    if token_count < 2 or not 1 <= stride <= context_length:
        raise ValueError("Need tokens and 1 <= stride <= context length")
    # Target index j uses a prefix ending at j-1. Targets are counted once.
    for first_target in range(1, token_count, stride):
        end = min(first_target + stride, token_count)
        begin = max(0, end - 1 - context_length)
        yield begin, end - 1, first_target, end


def verify():
    documents = [
        ([1.0, 1.0], [0.8, 0.8]),
        ([3.0] * 6, [2.8] * 6),
        ([1.0, 1.0], [1.2, 1.2]),
        ([3.0] * 6, [2.4] * 6),
    ]
    a = [torch.tensor(pair[0], dtype=torch.float64) for pair in documents]
    b = [torch.tensor(pair[1], dtype=torch.float64) for pair in documents]
    assert [item.numel() for item in a] == [2, 6, 2, 6]
    a_sum, b_sum = sum(item.sum() for item in a), sum(item.sum() for item in b)
    assert math.isclose(a_sum.item(), 40.0) and math.isclose(b_sum.item(), 35.2)
    a_nll, b_nll = a_sum / 16, b_sum / 16
    torch.testing.assert_close(a_nll, torch.tensor(2.5, dtype=torch.float64))
    torch.testing.assert_close(b_nll, torch.tensor(2.2, dtype=torch.float64))
    assert math.isclose(a_nll.exp().item(), 12.1825, abs_tol=1e-4)
    assert math.isclose(b_nll.exp().item(), 9.0250, abs_tol=1e-4)
    assert torch.stack([item.mean() for item in a]).mean() == 2.0
    torch.testing.assert_close(
        torch.stack([item.mean() for item in b]).mean(), torch.tensor(1.8, dtype=torch.float64)
    )
    for losses in (a, b):
        for item in losses:
            torch.testing.assert_close(-(-item).exp().log(), item)
    for item in a:
        assert item.numel() in (2, 6)
    for length in (3, 7, 3, 7):
        counted = []
        for begin, input_end, target_begin, target_end in target_windows(length, 4, 2):
            assert input_end - begin <= 4
            assert begin <= target_begin - 1 < input_end
            counted.extend(range(target_begin, target_end))
        assert counted == list(range(1, length))
    assert list(target_windows(7, 4, 2)) == [(0, 2, 1, 3), (0, 4, 3, 5), (2, 6, 5, 7)]
    assert [tuple(window[2:]) for window in target_windows(7, 4, 3)] == [(1, 4), (4, 7)]
    # A task score is paired by source, not by overlapping token window.
    first, second = [1, 0, 1, 0], [1, 1, 0, 1]
    scores = paired_bootstrap(first, second)
    assert scores["mean"] == 0.25
    assert scores["lower"] == -0.5 and scores["upper"] == 1.0
    assert scores == paired_bootstrap(first, second)
    # Wilson interval is conditional on independent, equal-probability trials.
    successes, count, z = 3, 4, 1.96
    rate = successes / count
    center = (rate + z * z / (2 * count)) / (1 + z * z / count)
    radius = (
        z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count)) / (1 + z * z / count)
    )
    assert 0 < center - radius < rate < center + radius < 1
    assert math.isclose(center - radius, 0.3006, abs_tol=0.001)
    assert math.isclose(center + radius, 0.9544, abs_tol=0.001)
    print(f"A: NLL={a_nll:.4f}, PPL={a_nll.exp():.4f}; B: NLL={b_nll:.4f}, PPL={b_nll.exp():.4f}")
    print(
        f"task A=2/4, B=3/4; B Wilson 95% interval [{center - radius:.3f}, {center + radius:.3f}]"
    )
    identical = paired_bootstrap([1, 0, 1, 1], [1, 0, 1, 1])
    assert identical["mean"] == identical["lower"] == identical["upper"] == 0
    shifted = paired_bootstrap([1.0, 2.0, 3.0], [2.0, 3.0, 4.0])
    assert shifted["mean"] == shifted["lower"] == shifted["upper"] == 1
    print(
        "PASS: paired bootstrap preserves sample identity, constant shift and reproducibility",
        scores,
    )


if __name__ == "__main__":
    verify()
