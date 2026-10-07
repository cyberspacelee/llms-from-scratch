"""训练稳定性：z-loss、logit 软截断、QK-Norm、QK-Clip 与监控指标，以及一个可复现的小实验。

`StableAttention` 是共用 GPT 注意力层的替身：权重布局完全相同，额外支持 QK-Norm、
注意力 logit 软截断，并记录每个头本步的最大注意力 logit（QK-Clip 与监控都要用它）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..transformer.model import GPT, CausalSelfAttention, GPTConfig, KVCache, RMSNorm, apply_rope
from .optimizers import AdamW


# region zloss
def z_loss(logits: torch.Tensor) -> torch.Tensor:
    """(log Z)² 的平均，其中 log Z = logsumexp(logits)。把 softmax 的归一化常数拉向 1。"""
    return torch.logsumexp(logits.float(), dim=-1).pow(2).mean()


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, z_coef: float = 0.0,
            softcap: float | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """交叉熵 + z_coef·z-loss，返回（总损失，纯交叉熵）。softcap 不为 None 时先软截断输出 logit。"""
    if softcap is not None:
        logits = soft_cap(logits, softcap)
    flat = logits.reshape(-1, logits.size(-1)).float()
    ce = F.cross_entropy(flat, targets.reshape(-1))
    total = ce + z_coef * z_loss(flat) if z_coef else ce
    return total, ce.detach()


def soft_cap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """cap·tanh(x / cap)：|x| ≪ cap 时近似恒等，|x| → ∞ 时平滑地饱和在 ±cap。"""
    return cap * torch.tanh(x / cap)
# endregion zloss


# region stable_attention
class StableAttention(CausalSelfAttention):
    """与 CausalSelfAttention 同权重布局，增加 QK-Norm 与注意力 logit 软截断。"""

    def __init__(self, config: GPTConfig, layer: int, qk_norm: bool = False,
                 softcap: float | None = None) -> None:
        super().__init__(config, layer)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else None
        self.softcap = softcap
        self.register_buffer("max_logit", torch.zeros(self.n_heads), persistent=False)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor,
                cache: KVCache | None = None, start_pos: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:            # QK-Norm：在 RoPE 之前对每个头的 q、k 做 RMSNorm
            q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        if cache is not None:
            k, v = cache.update(self.layer, start_pos, k, v)
        repeat = self.n_heads // self.n_kv_heads
        if repeat > 1:
            k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        S = k.shape[2]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)   # [B, h, T, S]
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        # 记录每个头在可见位置上的最大 logit：S_max^h = max q·k / √d
        self.max_logit = scores.detach().masked_fill(~mask, -torch.inf).amax(dim=(0, 2, 3))
        if self.softcap is not None:
            scores = soft_cap(scores, self.softcap)
        scores = scores.masked_fill(~mask, -torch.inf)
        out = scores.softmax(-1) @ v
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


def make_stable(model: GPT, qk_norm: bool = False, softcap: float | None = None) -> GPT:
    """把模型每层的注意力换成 StableAttention，并原样拷贝投影权重。"""
    for block in model.blocks:
        old = block.attn
        new = StableAttention(model.config, old.layer, qk_norm, softcap)
        new.load_state_dict(old.state_dict(), strict=False)
        new.to(next(old.parameters()).device)
        block.attn = new
    return model
# endregion stable_attention


# region qk_clip
@torch.no_grad()
def qk_clip_(model: GPT, tau: float = 100.0) -> int:
    """MuonClip 中的 QK-Clip：对 S_max^h > τ 的头，按 γ = τ / S_max^h 缩小 W_q、W_k。

    q·k 对 W_q、W_k 都是线性的，把两者各乘 √γ 恰好让该头的 logit 乘 γ。
    在优化器 step 之后、用上一步前向记录的 S_max 调用。返回被截断的头数。
    """
    clipped = 0
    for block in model.blocks:
        attn = block.attn
        if not isinstance(attn, StableAttention):
            raise TypeError("先用 make_stable 替换注意力层以记录 S_max")
        repeat = attn.n_heads // attn.n_kv_heads
        d = attn.head_dim
        for h, s_max in enumerate(attn.max_logit.tolist()):
            if s_max <= tau:
                continue
            gamma = tau / s_max
            rows = slice(h * d, (h + 1) * d)
            if repeat == 1:
                attn.q_proj.weight[rows] *= math.sqrt(gamma)
                attn.k_proj.weight[rows] *= math.sqrt(gamma)
            else:
                # GQA 的 K 头被多个查询头共享，只缩放本头的 W_q，避免误伤同组其他头
                attn.q_proj.weight[rows] *= gamma
            clipped += 1
    return clipped
# endregion qk_clip


# region monitor
def global_grad_norm(params) -> float:
    """全部参数梯度拼成一个向量后的 L2 范数。"""
    norms = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return torch.stack(norms).norm().item() if norms else 0.0


def update_to_param_ratio(before: dict[str, torch.Tensor], model: nn.Module) -> dict[str, float]:
    """每个参数的 ‖Δθ‖ / ‖θ‖。健康训练中通常在 1e-3 量级（随学习率调度变化）。"""
    out = {}
    for name, p in model.named_parameters():
        out[name] = ((p.detach() - before[name]).norm() / (before[name].norm() + 1e-12)).item()
    return out


def spike_score(values: list[float], window: int = 1000, n_sigma: float = 7.0) -> float:
    """OLMo 2 的 spike score：偏离前 window 步滚动均值超过 n_sigma 个标准差的点所占比例。"""
    x = torch.tensor(values, dtype=torch.float64)
    spikes = 0
    for t in range(1, len(x)):
        hist = x[max(0, t - window):t]
        if len(hist) < 2:
            continue
        if (x[t] - hist.mean()).abs() > n_sigma * hist.std():
            spikes += 1
    return spikes / max(1, len(x) - 1)
# endregion monitor


# region experiment
def markov_sampler(vocab: int, seed: int = 0, concentration: float = 0.3):
    """一阶马尔可夫链“语言”：转移矩阵每行是稀疏的狄利克雷分布，模型能学到但需要几百步。"""
    g = torch.Generator().manual_seed(seed)
    rng = np.random.default_rng(seed)
    probs = torch.from_numpy(rng.dirichlet(np.full(vocab, concentration), size=vocab)).float()

    def sample(batch: int, length: int) -> torch.Tensor:
        x = torch.empty(batch, length + 1, dtype=torch.long)
        x[:, 0] = torch.randint(0, vocab, (batch,), generator=g)
        for t in range(length):
            x[:, t + 1] = torch.multinomial(probs[x[:, t]], 1, generator=g).squeeze(1)
        return x

    return sample


@dataclass
class StabilityConfig:
    lr: float = 3e-3
    steps: int = 150
    warmup: int = 0
    qk_norm: bool = False
    z_coef: float = 0.0
    clip: float | None = None
    softcap: float | None = None
    qk_clip_tau: float | None = None
    weight_decay: float = 0.0
    seed: int = 0
    model: GPTConfig = field(default_factory=lambda: GPTConfig(
        vocab_size=64, context_length=32, d_model=64, n_layers=2, n_heads=4))


def run_stability_experiment(cfg: StabilityConfig, batch: int = 16) -> dict[str, list[float]]:
    """训练一个小 GPT，逐步记录损失、梯度范数、最大注意力 logit 与输出 log Z。"""
    torch.manual_seed(cfg.seed)
    model = make_stable(GPT(cfg.model), qk_norm=cfg.qk_norm)
    opt = AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    sample = markov_sampler(cfg.model.vocab_size, seed=cfg.seed)
    hist: dict[str, list[float]] = {"loss": [], "grad_norm": [], "max_logit": [], "log_z": []}
    T = cfg.model.context_length
    for step in range(cfg.steps):
        lr = cfg.lr * min(1.0, (step + 1) / cfg.warmup) if cfg.warmup else cfg.lr
        for group in opt.param_groups:
            group["lr"] = lr
        x = sample(batch, T)
        logits = model(x[:, :-1])
        loss, ce = lm_loss(logits, x[:, 1:], cfg.z_coef, cfg.softcap)
        opt.zero_grad()
        loss.backward()
        hist["grad_norm"].append(global_grad_norm(model.parameters()))
        if cfg.clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
        opt.step()
        if cfg.qk_clip_tau is not None:
            qk_clip_(model, cfg.qk_clip_tau)
        hist["loss"].append(ce.item())
        hist["max_logit"].append(max(b.attn.max_logit.max().item() for b in model.blocks))
        hist["log_z"].append(torch.logsumexp(logits.detach().float(), -1).abs().mean().item())
    return hist
# endregion experiment
