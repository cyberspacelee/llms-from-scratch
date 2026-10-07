"""从零训练一个小 GPT：BPE 分词 → token 流 → 随机窗口 → AdamW + 余弦学习率 + 梯度裁剪。

    uv run python -m llms_from_scratch.transformer.train            # 合成故事，CPU 约数分钟
    uv run python -m llms_from_scratch.transformer.train --data TinyStoriesV2-GPT4-valid.txt \\
        --vocab-size 2048 --steps 3000

模型就是全书共用的 ``model.GPT``；这里的优化器、调度与数据加载都是本部分手写的版本。
"""

from __future__ import annotations

import argparse
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .bpe import Tokenizer
from .data import EOT, encode_documents, get_batch, load_documents
from .model import GPT, GPTConfig
from .optim import AdamW, clip_grad_norm_, cosine_lr
from .sampling import generate


# region loss
def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """平均负对数似然：ℓ = logsumexp(o) − o_y。logits [..., V]，targets [...]。

    先减去每行最大值再求 logsumexp，避免 exp 上溢；不显式计算 softmax，避免 log(0)。
    """
    logits = logits.float()
    m = logits.amax(-1, keepdim=True)
    lse = (logits - m).exp().sum(-1).log() + m.squeeze(-1)
    target_logit = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (lse - target_logit).mean()
# endregion


@dataclass
class TrainConfig:
    data: str | None = None  # 以 <|endoftext|> 分隔的文本文件；None 用合成故事
    n_stories: int = 6000
    vocab_size: int = 512
    context_length: int = 128
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    batch_size: int = 16
    steps: int = 1500
    max_lr: float = 3e-3
    min_lr: float = 3e-4
    warmup_steps: int = 50
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    eval_every: int = 100
    eval_batches: int = 10
    seed: int = 0
    out: str | None = None  # 检查点路径


@torch.no_grad()
def evaluate(model: GPT, tokens: np.ndarray, cfg: TrainConfig, gen: torch.Generator) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(tokens, cfg.batch_size, cfg.context_length, generator=gen)
        losses.append(float(cross_entropy(model(x), y)))
    model.train()
    return sum(losses) / len(losses)


def param_groups(model: GPT, weight_decay: float) -> list[dict]:
    """只对矩阵（二维参数）做权重衰减；RMSNorm 增益等一维参数不衰减。"""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


# region train
def train(cfg: TrainConfig, log: Callable[[str], None] = print) -> dict:
    torch.manual_seed(cfg.seed)
    docs = load_documents(cfg.data, cfg.n_stories, cfg.seed)
    split = int(0.95 * len(docs))
    tokenizer = Tokenizer.train(f"{EOT}".join(docs[:split]), cfg.vocab_size, [EOT])
    train_ids = encode_documents(tokenizer, docs[:split])
    val_ids = encode_documents(tokenizer, docs[split:])
    log(f"词表 {len(tokenizer)}，训练 {len(train_ids):,} tokens，验证 {len(val_ids):,} tokens")

    config = GPTConfig(vocab_size=len(tokenizer), context_length=cfg.context_length,
                       d_model=cfg.d_model, n_layers=cfg.n_layers, n_heads=cfg.n_heads)
    model = GPT(config)
    log(f"参数量 {model.num_params(non_embedding=False):,}（非嵌入 {model.num_params():,}）")
    opt = AdamW(param_groups(model, cfg.weight_decay), lr=cfg.max_lr, betas=cfg.betas)
    gen = torch.Generator().manual_seed(cfg.seed)
    eval_gen = torch.Generator().manual_seed(cfg.seed + 1)
    history: dict[str, list] = {"step": [], "train": [], "lr": [], "grad_norm": [], "eval": []}
    start = time.time()
    for step in range(cfg.steps):
        lr = cosine_lr(step, cfg.max_lr, cfg.min_lr, cfg.warmup_steps, cfg.steps)
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_ids, cfg.batch_size, cfg.context_length, generator=gen)
        loss = cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        norm = clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        history["step"].append(step)
        history["train"].append(loss.item())
        history["lr"].append(lr)
        history["grad_norm"].append(norm)
        if (step + 1) % cfg.eval_every == 0 or step == cfg.steps - 1:
            val = evaluate(model, val_ids, cfg, eval_gen)
            history["eval"].append((step + 1, val))
            log(f"step {step + 1:5d}  train {loss.item():.3f}  val {val:.3f}  "
                f"ppl {math.exp(val):6.2f}  lr {lr:.2e}  |g| {norm:.2f}  {time.time() - start:.0f}s")
    if cfg.out:
        save_checkpoint(cfg.out, model, opt, cfg.steps, tokenizer, cfg)
    return {"model": model, "tokenizer": tokenizer, "history": history}
# endregion


def save_checkpoint(path: str, model: GPT, opt: torch.optim.Optimizer, step: int,
                    tokenizer: Tokenizer, cfg: TrainConfig) -> None:
    torch.save({
        "model": model.state_dict(), "optimizer": opt.state_dict(), "step": step,
        "config": asdict(model.config), "train_config": asdict(cfg),
        "vocab": tokenizer.vocab, "merges": tokenizer.merges, "special_tokens": tokenizer.special_tokens,
    }, path)


def load_checkpoint(path: str) -> tuple[GPT, Tokenizer]:
    state = torch.load(path, weights_only=False)
    model = GPT(GPTConfig(**state["config"]))
    model.load_state_dict(state["model"])
    return model, Tokenizer(state["vocab"], state["merges"], state["special_tokens"])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    defaults = TrainConfig()
    for name, value in asdict(defaults).items():
        if name == "betas":
            continue
        kind = type(value) if value is not None else str
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=value)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = vars(parser.parse_args(argv))
    prompt, temperature, top_p = args.pop("prompt"), args.pop("temperature"), args.pop("top_p")
    result = train(TrainConfig(**args))
    model, tokenizer = result["model"], result["tokenizer"]
    idx = torch.tensor([tokenizer.encode(prompt)])
    eot = tokenizer.encode(EOT)[0]
    for i in range(3):
        out = generate(model, idx, 200, temperature=temperature, top_p=top_p, eos_id=eot,
                       generator=torch.Generator().manual_seed(i))
        print(f"\n--- 样本 {i + 1} ---\n" + tokenizer.decode(out[0].tolist()).replace(EOT, ""))


if __name__ == "__main__":
    main()
