"""Token-weighted evaluation and non-overlapping target windows."""

import math

import torch


def paired_bootstrap(first, second, repeats=2000, seed=17):
    first, second = torch.as_tensor(first, dtype=torch.float64), torch.as_tensor(second, dtype=torch.float64)
    if first.ndim != 1 or first.shape != second.shape or first.numel() < 2 or repeats < 1:
        raise ValueError("need two matching vectors and a positive repeat count")
    if not torch.isfinite(first).all() or not torch.isfinite(second).all():
        raise ValueError("paired observations must be finite")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(first.numel(), (repeats, first.numel()), generator=generator)
    differences = (second - first)[indices].mean(-1)
    lower, upper = differences.quantile(torch.tensor([.025, .975], dtype=torch.float64))
    return dict(mean=(second - first).mean().item(), lower=lower.item(), upper=upper.item(),
                seed=seed, repeats=repeats)


def target_windows(token_count, context_length, stride):
    if token_count < 2 or not 1 <= stride <= context_length:
        raise ValueError("Need tokens and 1 <= stride <= context length")
    # Target index j uses a prefix ending at j-1. Targets are counted once.
    for first_target in range(1, token_count, stride):
        end = min(first_target + stride, token_count)
        begin = max(0, end - 1 - context_length)
        yield begin, end - 1, first_target, end


def verify():
    probabilities = torch.tensor([0.5, 0.25, 0.8], dtype=torch.float64)
    nll = -probabilities.log()
    ppl = nll.mean().exp()
    torch.testing.assert_close(ppl, torch.tensor(10.0 ** (1 / 3), dtype=torch.float64))
    losses = [torch.tensor([1.0, 1.0], dtype=torch.float64),
              torch.tensor([3.0] * 6, dtype=torch.float64)]
    token_mean = sum(x.sum() for x in losses) / sum(x.numel() for x in losses)
    sequence_mean = torch.stack([x.mean() for x in losses]).mean()
    assert token_mean.item() == 2.5 and sequence_mean.item() == 2.0
    for context, stride in [(4, 1), (4, 2), (4, 4), (8, 3)]:
        counted = []
        for begin, input_end, target_begin, target_end in target_windows(15, context, stride):
            assert input_end - begin <= context
            assert begin <= target_begin - 1 < input_end
            counted.extend(range(target_begin, target_end))
        assert counted == list(range(1, 15))
    # Wilson interval for a binomial accuracy, without a normal approximation at the edges.
    successes, count, z = 8, 10, 1.96
    rate = successes / count
    center = (rate + z * z / (2 * count)) / (1 + z * z / count)
    radius = z * math.sqrt(rate * (1-rate) / count + z*z / (4*count*count)) / (1+z*z/count)
    assert 0 < center-radius < rate < center+radius < 1
    print(f"evaluation: NLL={nll.mean():.6f}, PPL={ppl:.6f}, weighted NLL={token_mean:.2f}")
    print(f"8/10 accuracy: Wilson 95% interval [{center-radius:.3f}, {center+radius:.3f}]")
    identical = paired_bootstrap([1, 0, 1, 1], [1, 0, 1, 1])
    assert identical["mean"] == identical["lower"] == identical["upper"] == 0
    shifted = paired_bootstrap([1., 2., 3.], [2., 3., 4.])
    assert shifted["mean"] == shifted["lower"] == shifted["upper"] == 1
    scores = paired_bootstrap([0, 0, 1, 0, 1, 0], [1, 0, 1, 1, 1, 0])
    assert scores == paired_bootstrap([0, 0, 1, 0, 1, 0], [1, 0, 1, 1, 1, 0])
    print("PASS: paired bootstrap preserves sample identity, constant shift and reproducibility", scores)


if __name__ == "__main__":
    verify()
