"""Export fixed curriculum cases from the canonical Python implementation.

Run ``uv run python scripts/export_site_traces.py --check`` to detect stale data.
Interactive controls remain in TypeScript; the check compares their outputs to
these independently evaluated float64 cases.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from examples.frameworks.autograd import forward
from llms_from_scratch.attention import scaled_dot_product_attention
from llms_from_scratch.attention.position import apply_rope
from llms_from_scratch.chapters.ch10_flash_attention import online_attention
from llms_from_scratch.inference.sampling import distribution
from llms_from_scratch.training.losses import token_loss_sum

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "site/src/data/reference-traces.json"
SOURCES = [
    "examples/frameworks/autograd.py",
    "src/llms_from_scratch/attention/position.py",
    "src/llms_from_scratch/attention/softmax.py",
    "src/llms_from_scratch/attention/patterns.py",
    "src/llms_from_scratch/config.py",
    "src/llms_from_scratch/chapters/ch10_flash_attention.py",
    "src/llms_from_scratch/inference/sampling.py",
    "src/llms_from_scratch/training/losses.py",
]


def export():
    dtype = torch.float64
    sampling = []
    logits = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=dtype).log()
    base = dict(temperature=1.0, topK=4, topP=1.0, minP=0.0, penalty=1.0, history=[])
    for overrides in (
        {},
        {"topP": 0.6},
        {"topK": 2},
        {"minP": 0.6},
        {"temperature": 0.5},
        {"penalty": 2.0, "history": [0, 0]},
        {"penalty": 2.0, "history": [0, 1]},
        {"temperature": 0.5, "minP": 0.6},
    ):
        options = base | overrides
        result = distribution(
            logits,
            options["temperature"],
            options["topK"],
            options["topP"],
            options["minP"],
            options["penalty"],
            options["history"],
        )
        sampling.append(
            dict(logits=logits.tolist(), options=options, probabilities=result.tolist())
        )
    # Include the UI's BOS/A distribution, both penalty sign branches and stable ties.
    for values, overrides in (
        (torch.tensor([0.1, 0.4, 0.3, 0.2], dtype=dtype).log(), {"minP": 0.6, "history": [0, 1]}),
        (
            torch.tensor([2.0, 0.0, -1.0, 1.0], dtype=dtype),
            {"penalty": 2.0, "history": [0, 1, 2, 2]},
        ),
        (torch.zeros(4, dtype=dtype), {"topK": 2}),
    ):
        options = base | overrides
        result = distribution(
            values,
            options["temperature"],
            options["topK"],
            options["topP"],
            options["minP"],
            options["penalty"],
            options["history"],
        )
        sampling.append(
            dict(logits=values.tolist(), options=options, probabilities=result.tolist())
        )
    cached = []
    # q/k before RoPE are (1,0); v_j=(j,0). Rectangular masks use absolute positions.
    for past, length in ((0, 3), (2, 1), (2, 3), (3, 2), (4, 2)):
        positions = torch.arange(past + length)
        raw = torch.zeros(1, 1, past + length, 2, dtype=dtype)
        raw[..., 0] = 1
        k = apply_rope(raw, positions)
        q = apply_rope(raw[:, :, past:], positions[past:])
        v = torch.zeros_like(raw)
        v[..., 0] = positions.to(dtype)
        visible = positions[None, :] <= positions[past:, None]
        output = scaled_dot_product_attention(q, k, v, visible)
        for row in range(length):
            cached.append(
                dict(past=past, length=length, row=row, output=output[0, 0, row].tolist())
            )
    online = []
    for position in range(6):
        q = torch.ones(1, 1, 1, 1, dtype=dtype)
        k = torch.arange(position + 1, dtype=dtype).reshape(1, 1, -1, 1)
        v = k + 1
        for block in (1, 2, 3, 4):
            output = online_attention(q, k, v, block).item()
            dense = scaled_dot_product_attention(q, k, v).item()
            torch.testing.assert_close(
                torch.tensor(output, dtype=dtype),
                torch.tensor(dense, dtype=dtype),
                atol=1e-12,
                rtol=0.0,
            )
            online.append(dict(position=position, block=block, output=output))
    s = 2**-0.5
    logits = torch.tensor(
        [[s, -s, 0, 0], [-s, s, 0, 0], [s, -s, 0, 0], [-s, s, 0, 0], [s, -s, 0, 0], [s, -s, 0, 0]],
        dtype=dtype,
    )
    targets = [0, 1, 0, 1, 0, 0]
    losses = []
    for ignored in (
        [False, False, True, False, False, True],
        [False] * 6,
        [True] * 6,
        [True, False, True, True, True, True],
    ):
        result = token_loss_sum(
            logits.reshape(2, 3, 4),
            torch.tensor(targets).reshape(2, 3),
            ~torch.tensor(ignored).reshape(2, 3),
        )
        losses.append(
            dict(
                targets=targets,
                ignored=ignored,
                total=result.loss_sum.item(),
                valid=result.valid_count.item(),
                mean=result.mean().item() if result.valid_count else None,
            )
        )
    branches = []
    for w0, w1, bias, seed in (
        (1.0, 2.0, 1.0, 1.0),
        (1.0, 2.0, 1.0, 0.5),
        (-1.0, 0.5, 2.0, -1.0),
        (0.0, 0.0, 0.0, 1.0),
    ):
        w = torch.tensor([w0, w1], dtype=dtype, requires_grad=True)
        b = torch.tensor(bias, dtype=dtype, requires_grad=True)
        u, loss = forward(w, b)
        u.retain_grad()
        (loss * seed).backward()
        branches.append(
            dict(
                w0=w0,
                w1=w1,
                bias=bias,
                seed=seed,
                u=u.tolist(),
                loss=loss.item(),
                gu=u.grad.tolist(),
                gw=w.grad.tolist(),
                gb=b.grad.item(),
            )
        )
    source_hash = hashlib.sha256()
    for path in SOURCES:
        source_hash.update(path.encode())
        source_hash.update((ROOT / path).read_bytes())
    return dict(
        schemaVersion=1,
        dtype="float64",
        atol=1e-12,
        sourceHash=source_hash.hexdigest(),
        sources=SOURCES,
        sampling=sampling,
        cachedAttention=cached,
        onlineAttention=online,
        maskedLoss=losses,
        vectorBranch=branches,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    data = json.dumps(export(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.check:
        if not DESTINATION.exists() or DESTINATION.read_text() != data:
            raise SystemExit(
                "Stale Python reference traces: uv run python scripts/export_site_traces.py"
            )
        print("Python reference traces: source hash and float64 results are current")
    else:
        DESTINATION.write_text(data)
        print(DESTINATION.relative_to(ROOT))


if __name__ == "__main__":
    main()
