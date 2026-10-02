"""小型功能实验、理论账本与用户主动选择的性能基准。"""

from __future__ import annotations

import statistics
import time
from functools import partial
from pathlib import Path

import torch

from ..analysis import attention_cost, parameter_count
from ..config import AttentionConfig, ModelConfig, PositionConfig
from ..models import Transformer, make_attention
from ..training import language_model_loss, teacher_forcing, token_loss


def consistency(config: ModelConfig, device: torch.device) -> dict[str, object]:
    """用小张量核对完整、分块缓存前向和 greedy生成一致。

    Args:
        config: 本模块的显式配置对象。
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 报告：logits shape、参数、cache误差/字节和 greedy tokens。
    """
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
    """Learn a deterministic cyclic sequence / copy task, no external tokenizer/data.

    Args:
        config: 本模块的显式配置对象。
        steps: optimizer 更新次数，正整数。
        device: 执行 torch.device，默认由 CLI 选择 CPU。
        output_path: 可选 .pt checkpoint 路径；创建父目录并保存当前状态。

    Returns:
        dict 训练报告；可选将配置、模型/optimizer权重保存至 output_path。
    """
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
    """生成不同 Attention 配方的理论成本表。

    Args:
        length: 待分析/计时的 sequence length T。

    Returns:
        dict[str,dict[str,int]]，各种 Attention的理论成本。
    """
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
    """显式计时 decode，CUDA 时同步；不推断 SDPA 后端。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。
        length: 待分析/计时的 sequence length T。
        repeats: 计时重复次数，正整数。

    Returns:
        dict 设备/torch版本与各实现 median_ms、cache bytes、CUDA峰值。
    """
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
