"""Joint scaling-law fit, compute-optimal allocation and an optional CPU pilot."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def predict(n, d, fit):
    n, d = np.asarray(n, dtype=float), np.asarray(d, dtype=float)
    if not np.isfinite(n).all() or not np.isfinite(d).all() or (n <= 0).any() or (d <= 0).any():
        raise ValueError("model and data counts must be finite and positive")
    return fit["E"] + fit["A"] / n ** fit["alpha"] + fit["B"] / d ** fit["beta"]


def fit_joint(n, d, losses):
    n, d, losses = [np.asarray(x, dtype=float) for x in (n, d, losses)]
    if n.ndim != 1 or n.shape != d.shape or n.shape != losses.shape or n.size < 6:
        raise ValueError("need at least six matching one-dimensional observations")
    if any(not np.isfinite(x).all() or (x <= 0).any() for x in (n, d, losses)):
        raise ValueError("all observations must be finite and positive")
    if np.ptp(n) == 0 or np.ptp(d) == 0:
        raise ValueError("both N and D must vary to fit the joint law")
    # ponytail: bounded exponent grid, refine around its best point; use robust nonlinear fitting for research.
    best, center, radius = None, (.7, .7), .5
    for resolution in (21, 11, 11):
        for alpha in np.linspace(max(.01, center[0] - radius), center[0] + radius, resolution):
            for beta in np.linspace(max(.01, center[1] - radius), center[1] + radius, resolution):
                design = np.column_stack((np.ones_like(n), n ** -alpha, d ** -beta))
                coefficients, _, rank, _ = np.linalg.lstsq(design, losses, rcond=None)
                if rank != 3 or coefficients[0] < 0 or (coefficients[1:] <= 0).any():
                    continue
                mse = np.mean((design @ coefficients - losses) ** 2)
                if best is None or mse < best["mse"]:
                    best = dict(E=float(coefficients[0]), A=float(coefficients[1]), B=float(coefficients[2]),
                                alpha=float(alpha), beta=float(beta), mse=float(mse))
        if best is None:
            raise ValueError("no positive-coefficient fit in the declared exponent range")
        center = best["alpha"], best["beta"]
        radius /= 5
    return best


def optimal_allocation(compute, fit):
    if not math.isfinite(compute) or compute <= 0 or any(not math.isfinite(fit[k]) or fit[k] <= 0 for k in ("A", "B", "alpha", "beta")):
        raise ValueError("compute, coefficients and exponents must be finite and positive")
    alpha, beta = fit["alpha"], fit["beta"]
    n = (alpha * fit["A"] / (beta * fit["B"]) * (compute / 6) ** beta) ** (1 / (alpha + beta))
    return n, compute / (6 * n)


def cpu_pilot():
    import sys
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "principles"))
    from modern_decoder import ModernDecoder
    from language_model import sequence_loss
    from text_pretraining import TRAIN_TEXT, VALID_TEXT, windows
    from tokenization import PreSplitBPE

    torch.set_num_threads(1)
    tokenizer = PreSplitBPE(["<bos>", "<eos>"]).fit(TRAIN_TEXT, 16)
    samples = windows(TRAIN_TEXT, tokenizer, 16)
    validation = windows(VALID_TEXT, tokenizer, 16)
    records = []
    for width in (16, 24, 32):
        torch.manual_seed(41)
        model = ModernDecoder(tokenizer.vocab_size, width=width, ff_width=2 * width,
                              heads=4, kv_heads=2, head_width=4, max_length=32).double()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.005)
        seen = 0
        for step in range(1, 33):
            document = samples[(step - 1) % len(samples)]
            optimizer.zero_grad(set_to_none=True)
            sequence_loss(model(document[:, :-1]), document[:, 1:]).backward()
            optimizer.step()
            seen += document.shape[1] - 1
            if step in (8, 16, 32):
                with torch.no_grad():
                    count = sum(x.shape[1] - 1 for x in validation)
                    loss = sum(sequence_loss(model(x[:, :-1]), x[:, 1:]).item() * (x.shape[1] - 1)
                               for x in validation) / count
                parameters = model.parameter_count()
                records.append(dict(N=parameters, D=seen, step=step, nll=loss,
                                    approximate_flop=6 * parameters * seen, seed=41,
                                    width=width, validation_targets=count))
    return dict(kind="CPU pilot, repeated short prose; not an empirical large-scale law", records=records)


def verify():
    truth = dict(E=.5, A=4., B=2., alpha=.6, beta=.4)
    n, d = np.meshgrid([1., 4., 16., 64.], [2., 8., 32., 128.])
    n, d = n.ravel(), d.ravel()
    losses = predict(n, d, truth)
    fit = fit_joint(n, d, losses)
    np.testing.assert_allclose(predict(n, d, fit), losses, atol=1e-10)
    np.testing.assert_allclose([fit[k] for k in truth], [truth[k] for k in truth], atol=1e-9)
    held_n, held_d = np.array([3., 10.]), np.array([7., 50.])
    np.testing.assert_allclose(predict(held_n, held_d, fit), predict(held_n, held_d, truth), atol=1e-10)
    optimal_n, optimal_d = optimal_allocation(6e6, fit)
    np.testing.assert_allclose(6 * optimal_n * optimal_d, 6e6)
    grid = np.geomspace(optimal_n / 10, optimal_n * 10, 301)
    assert predict(optimal_n, optimal_d, fit) <= predict(grid, 6e6 / (6 * grid), fit).min() + 1e-12
    n2, d2 = optimal_allocation(24e6, fit)
    np.testing.assert_allclose(n2 / optimal_n, 4 ** .4)
    np.testing.assert_allclose(d2 / optimal_d, 4 ** .6)
    try:
        fit_joint(np.ones(6), np.arange(1, 7), np.ones(6))
    except ValueError:
        pass
    else:
        raise AssertionError("unidentifiable design accepted")
    print("PASS: synthetic E/A/B/alpha/beta recovery, held-out predictions, isoflop optimum and unequal exponents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    verify()
    if args.measure:
        payload = json.dumps(cpu_pilot(), indent=2)
        if args.output:
            args.output.write_text(payload, encoding="utf-8")
        print(payload)
