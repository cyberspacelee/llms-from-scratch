"""Runnable CPU experiments: cache equivalence, training and honest benchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from functools import partial
from pathlib import Path

import torch

from .analysis import attention_cost, parameter_count
from .config import AttentionConfig, ModelConfig, PositionConfig
from .model import Transformer, make_attention
from .objectives import language_model_loss, teacher_forcing, token_loss
from .presets import PRESETS, preset


def consistency(config: ModelConfig, device: torch.device) -> dict[str, object]:
    model = Transformer(config).to(device).double().eval()
    ids = torch.arange(7, device=device)[None].repeat(2, 1)
    source = torch.flip(ids[:, :5], (1,)) if config.architecture == "encoder_decoder" else None
    with torch.no_grad():
        full = model(ids, source_ids=source)
        if config.architecture == "encoder":
            return {
                "shape": list(full.logits.shape),
                "parameters": parameter_count(model),
                "cache": "bidirectional: not appendable",
            }
        cache, outputs, start = None, [], 0
        for size in (3, 1, 3):
            result = model(
                ids[:, start : start + size],
                source_ids=source if cache is None else None,
                cache=cache,
                use_cache=True,
            )
            outputs.append(result.logits)
            cache, start = result.cache, start + size
        error = (full.logits - torch.cat(outputs, 1)).abs().max().item()
        torch.testing.assert_close(full.logits, torch.cat(outputs, 1), atol=1e-9, rtol=1e-7)
        cached = model.generate(ids[:, :2], 4, source_ids=source)
        naive = model.generate(ids[:, :2], 4, source_ids=source, cached=False)
        assert torch.equal(cached, naive)
    return {
        "shape": list(full.logits.shape),
        "parameters": parameter_count(model),
        "max_cache_error": error,
        "kv_cache_bytes_float64": cache.kv_nbytes,
        "greedy_tokens": cached.tolist(),
    }


def train(
    config: ModelConfig, steps: int, device: torch.device, output_path: str | None = None
) -> dict[str, object]:
    """Learn a deterministic cyclic sequence / copy task, no external tokenizer/data."""
    model = Transformer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    tokens = torch.arange(1, 9, device=device)[None].repeat(4, 1)
    if config.architecture == "encoder":
        raise ValueError("training demo requires a causal decoder")
    initial = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if config.architecture == "encoder_decoder":
            inputs, targets, valid = teacher_forcing(tokens, 0)
            output = model(inputs, source_ids=tokens, valid=valid)
            loss = token_loss(output.logits, targets, valid) + 0.01 * output.auxiliary_loss
            routing = output.routing
        else:
            output = model(tokens)
            loss, _, routing = language_model_loss(model, output, tokens)
        initial = loss.item() if initial is None else initial
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        model.update_router_bias(routing)
    with torch.no_grad():
        if config.architecture == "encoder_decoder":
            final = token_loss(model(inputs, source_ids=tokens).logits, targets).item()
        else:
            final = language_model_loss(model, model(tokens), tokens)[0].item()
    report = {
        "initial_loss": initial,
        "final_loss": final,
        "steps": steps,
        "parameters": parameter_count(model),
    }
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": config,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "report": report,
            },
            path,
        )
        report["checkpoint"] = str(path)
    return report


def ledger(length: int) -> dict[str, dict[str, int]]:
    return {
        kind: attention_cost(
            AttentionConfig(
                kind=kind,
                position=PositionConfig(kind="none")
                if kind in {"linear", "delta", "gated_delta"}
                else PositionConfig(),
            ),
            32,
            key_tokens=length,
        ).to_dict()
        for kind in ("mha", "mqa", "gqa", "mla", "linear", "delta", "gated_delta")
    }


def benchmark(device: torch.device, length: int, repeats: int) -> dict[str, object]:
    x = torch.randn(1, length, 32, device=device)
    result = {}
    for kind in ("mha", "gqa", "mla"):
        for backend in ("manual", "sdpa") if kind != "mla" else ("naive", "absorbed"):
            c = AttentionConfig(
                kind=kind,
                backend=backend if kind != "mla" else "manual",
                mla_impl=backend if kind == "mla" else "absorbed",
            )
            module = make_attention(32, c).to(device).eval()
            with torch.no_grad():
                _, prefix = module(x[:, :-1], causal=True, use_cache=True)
                run = partial(
                    module,
                    x[:, -1:],
                    cache=prefix,
                    query_offset=length - 1,
                    causal=True,
                    use_cache=True,
                )
                for _ in range(3):
                    run()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                times = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    run()
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    times.append((time.perf_counter() - start) * 1000)
            result[f"{kind}/{backend}"] = {
                "decode_median_ms": statistics.median(times),
                "cache_bytes": prefix.nbytes,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
            }
    return {
        "device": str(device),
        "torch": torch.__version__,
        "length": length,
        "note": "wall-clock with synchronization; independent random weights; SDPA backend selected by PyTorch, not necessarily Flash",
        "results": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=("check", "train", "ledger", "benchmark"))
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
    if args.experiment == "ledger":
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
