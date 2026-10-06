"""全书共用的 GPT 式 Decoder：RMSNorm + RoPE + GQA + SwiGLU，Pre-Norm 结构。

公开接口（其他部分依赖，修改时保持兼容）：
    GPTConfig, GPT, KVCache, RMSNorm, SwiGLU, CausalSelfAttention, Block,
    rope_frequencies, apply_rope
    GPT.forward(idx, cache=None, start_pos=0) -> logits[B, T, V]
    GPT.generate(idx, max_new_tokens, temperature=1.0, top_k=None)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPTConfig:
    vocab_size: int = 256
    context_length: int = 256
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int | None = None  # None 表示与 n_heads 相同（标准多头注意力）
    d_ff: int | None = None  # None 时取约 8/3 * d_model 并对齐到 64 的倍数
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.d_model % self.n_heads:
            raise ValueError("d_model 必须能被 n_heads 整除")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads 必须是 n_kv_heads 的整数倍")
        if self.d_ff is None:
            self.d_ff = 64 * math.ceil(8 * self.d_model / 3 / 64)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * g，在 float32 中计算以保证低精度下的稳定。"""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + self.eps)
        return (normed * self.weight).to(x.dtype)


def rope_frequencies(head_dim: int, length: int, theta: float = 10000.0) -> torch.Tensor:
    """返回复数旋转因子 e^{i·m·θ_k}，形状 [length, head_dim // 2]。"""
    k = torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
    inv_freq = theta ** (-k)
    angles = torch.outer(torch.arange(length, dtype=torch.float32), inv_freq)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """把 x[..., T, head_dim] 的相邻两维看作复数并乘以旋转因子 freqs[T, head_dim/2]。"""
    pairs = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated = torch.view_as_real(pairs * freqs)
    return rotated.flatten(-2).to(x.dtype)


class KVCache:
    """按层预分配的 K/V 缓存，形状 [B, n_kv_heads, max_len, head_dim]。"""

    def __init__(self, config: GPTConfig, batch_size: int, max_len: int | None = None,
                 device: torch.device | str | None = None, dtype: torch.dtype = torch.float32) -> None:
        max_len = max_len or config.context_length
        shape = (batch_size, config.n_kv_heads, max_len, config.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(config.n_layers)]
        self.length = 0  # 已写入的位置数，由 GPT.forward 在所有层写完后推进

    def update(self, layer: int, start: int, k: torch.Tensor, v: torch.Tensor):
        end = start + k.shape[2]
        self.k[layer][:, :, start:end] = k
        self.v[layer][:, :, start:end] = v
        return self.k[layer][:, :, :end], self.v[layer][:, :, :end]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.n_heads, self.n_kv_heads, self.head_dim = config.n_heads, config.n_kv_heads, config.head_dim
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor,
                cache: KVCache | None = None, start_pos: int = 0) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        if cache is not None:
            k, v = cache.update(self.layer, start_pos, k, v)
        # GQA：每组 n_heads / n_kv_heads 个查询头共享同一个 K/V 头
        repeat = self.n_heads // self.n_kv_heads
        if repeat > 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        S = k.shape[2]
        # 第 t 个新 token 的绝对位置是 start_pos + t，只能看到位置 <= 它的键
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    """FFN(x) = W2 (SiLU(W1 x) ⊙ W3 x)。"""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """Pre-Norm 块：x + Attn(Norm(x))，再 x + FFN(Norm(x))。"""

    def __init__(self, config: GPTConfig, layer: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attn = CausalSelfAttention(config, layer)
        self.ffn_norm = RMSNorm(config.d_model, config.norm_eps)
        self.ffn = SwiGLU(config.d_model, config.d_ff)

    def forward(self, x, freqs, cache=None, start_pos=0):
        x = x + self.attn(self.attn_norm(x), freqs, cache, start_pos)
        return x + self.ffn(self.ffn_norm(x))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config, i) for i in range(config.n_layers))
        self.norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.register_buffer(
            "freqs", rope_frequencies(config.head_dim, config.context_length, config.rope_theta),
            persistent=False,
        )
        self.apply(self._init)
        # 残差分支的输出投影按 1/sqrt(2L) 缩放，保持残差流方差随深度不增长
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "w2.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * config.n_layers))

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, idx: torch.Tensor, cache: KVCache | None = None, start_pos: int = 0) -> torch.Tensor:
        T = idx.shape[1]
        if start_pos + T > self.config.context_length:
            raise ValueError("序列超过 context_length")
        x = self.embed(idx)
        freqs = self.freqs[start_pos:start_pos + T]
        for block in self.blocks:
            x = block(x, freqs, cache, start_pos)
        if cache is not None:
            cache.length = start_pos + T
        return self.lm_head(self.norm(x))

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        return n - self.embed.weight.numel() if non_embedding else n

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
        """带 KV Cache 的自回归生成：先对整段提示做一次 prefill，再逐 token decode。"""
        cache = KVCache(self.config, idx.shape[0], device=idx.device, dtype=self.embed.weight.dtype)
        logits = self(idx, cache, 0)[:, -1]
        for _ in range(max_new_tokens):
            if temperature == 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.shape[-1])).values[:, -1:]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                nxt = torch.multinomial(F.softmax(logits, -1), 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
            if idx.shape[1] >= self.config.context_length:
                break
            logits = self(nxt, cache, cache.length)[:, -1]
        return idx
