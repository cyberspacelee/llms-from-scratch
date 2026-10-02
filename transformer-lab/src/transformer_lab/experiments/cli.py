"""Transformer Lab 命令行入口，默认仅执行指定实验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .presets import PRESETS, preset
from .runners import benchmark, consistency, ledger, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("lesson", "check", "train", "ledger", "benchmark"))
    parser.add_argument(
        "--step", type=int, choices=range(14), default=0, help="lesson 演进步号 00–13"
    )
    parser.add_argument("--preset", choices=PRESETS, default="deepseek")
    parser.add_argument("--architecture", choices=("encoder", "decoder", "encoder_decoder"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", help="JSON report, or .pt checkpoint for train")
    args = parser.parse_args()
    if args.steps < 1 or args.length < 2 or args.repeats < 1:
        parser.error("steps/repeats must be positive, length >= 2")
    torch.set_num_threads(1)
    torch.manual_seed(7)
    device = torch.device(args.device)
    if args.experiment == "lesson":
        from ..tutorials.evolution import run_lesson

        report = run_lesson(args.step, device)
    elif args.experiment == "ledger":
        report = ledger(args.length)
    elif args.experiment == "benchmark":
        report = benchmark(device, args.length, args.repeats)
    elif args.experiment == "train":
        report = train(preset(args.preset, args.architecture), args.steps, device, args.output)
    else:
        report = consistency(preset(args.preset, args.architecture), device)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output and args.experiment != "train":
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized + "\n")


if __name__ == "__main__":
    main()
