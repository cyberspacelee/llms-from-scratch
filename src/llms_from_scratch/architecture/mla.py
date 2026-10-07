"""多头潜在注意力（MLA，DeepSeek-V2/V3）：低秩 KV 压缩、解耦 RoPE 与矩阵吸收。

记号与 DeepSeek-V3 技术报告 §2.1.1 一致：
    c_kv = W_DKV h            潜向量，d_c 维，所有头共享，进入缓存
    k_C  = W_UK c_kv          每头内容键（n_heads × d_nope）
    v    = W_UV c_kv          每头值（n_heads × d_v）
    k_R  = RoPE(W_KR h)       解耦位置键，d_R 维，所有头共享，进入缓存
    q    = [W_UQ c_q ; RoPE(W_QR c_q)]，c_q = W_DQ h（查询也做低秩压缩）

``forward`` 有两条等价路径：``absorb=False`` 时显式还原每个头的 K/V（训练与 prefill 的
写法），``absorb=True`` 时把 W_UK 并入查询、把 W_UV 推迟到注意力加权之后（decode 的写法），
全程只读缓存中的 c_kv 与 k_R。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from llms_from_scratch.transformer.model import RMSNorm, apply_rope, rope_frequencies


@dataclass
class MLAConfig:
    d_model: int = 64
    n_heads: int = 4
    kv_rank: int = 16  # d_c
    q_rank: int = 24  # d_c'，0 表示查询不压缩
    nope_dim: int = 16  # 每头内容维 d_h
    rope_dim: int = 8  # 解耦 RoPE 维 d_R
    v_dim: int = 16
    max_len: int = 128
    rope_theta: float = 10000.0


class MLACache:
    """每层每个 token 只缓存 kv_rank + rope_dim 个数。"""

    def __init__(self, config: MLAConfig, batch_size: int) -> None:
        self.c_kv = torch.zeros(batch_size, config.max_len, config.kv_rank)
        self.k_rope = torch.zeros(batch_size, config.max_len, config.rope_dim)

    def update(self, start: int, c_kv: torch.Tensor, k_rope: torch.Tensor):
        end = start + c_kv.shape[1]
        self.c_kv[:, start:end] = c_kv
        self.k_rope[:, start:end] = k_rope
        return self.c_kv[:, :end], self.k_rope[:, :end]


class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.cfg = c = config
        H = c.n_heads
        if c.q_rank:
            self.wq_down = nn.Linear(c.d_model, c.q_rank, bias=False)
            self.q_norm = RMSNorm(c.q_rank)
            q_in = c.q_rank
        else:
            q_in = c.d_model
        self.wq_up = nn.Linear(q_in, H * (c.nope_dim + c.rope_dim), bias=False)
        self.wkv_down = nn.Linear(c.d_model, c.kv_rank, bias=False)  # W_DKV
        self.kv_norm = RMSNorm(c.kv_rank)
        self.wk_rope = nn.Linear(c.d_model, c.rope_dim, bias=False)  # W_KR，所有头共享
        self.wk_up = nn.Linear(c.kv_rank, H * c.nope_dim, bias=False)  # W_UK
        self.wv_up = nn.Linear(c.kv_rank, H * c.v_dim, bias=False)  # W_UV
        self.wo = nn.Linear(H * c.v_dim, c.d_model, bias=False)
        self.scale = 1.0 / math.sqrt(c.nope_dim + c.rope_dim)
        self.register_buffer("freqs", rope_frequencies(c.rope_dim, c.max_len, c.rope_theta),
                             persistent=False)

    def _queries(self, x: torch.Tensor, freqs: torch.Tensor):
        c = self.cfg
        B, T, _ = x.shape
        cq = self.q_norm(self.wq_down(x)) if c.q_rank else x
        q = self.wq_up(cq).view(B, T, c.n_heads, c.nope_dim + c.rope_dim).transpose(1, 2)
        q_nope, q_rope = q.split([c.nope_dim, c.rope_dim], dim=-1)
        return q_nope, apply_rope(q_rope, freqs)  # [B, H, T, ·]

    def forward(self, x: torch.Tensor, cache: MLACache | None = None, start_pos: int = 0,
                absorb: bool = False) -> torch.Tensor:
        B, T, _ = x.shape
        freqs = self.freqs[start_pos:start_pos + T]
        q_nope, q_rope = self._queries(x, freqs)
        c_kv = self.kv_norm(self.wkv_down(x))  # [B, T, d_c]
        k_rope = apply_rope(self.wk_rope(x), freqs)  # [B, T, d_R]
        if cache is not None:
            c_kv, k_rope = cache.update(start_pos, c_kv, k_rope)
        S = c_kv.shape[1]
        mask = torch.ones(T, S, dtype=torch.bool, device=x.device).tril(diagonal=S - T)
        attend = self._absorbed if absorb else self._naive
        out = attend(q_nope, q_rope, c_kv, k_rope, mask)  # [B, H, T, d_v]
        return self.wo(out.transpose(1, 2).reshape(B, T, -1))

    # region naive
    def _naive(self, q_nope, q_rope, c_kv, k_rope, mask):
        """显式路径：先把潜向量上投影成每头的 K 与 V，再做标准注意力。"""
        c = self.cfg
        B, S, _ = c_kv.shape
        k_nope = self.wk_up(c_kv).view(B, S, c.n_heads, c.nope_dim).transpose(1, 2)
        v = self.wv_up(c_kv).view(B, S, c.n_heads, c.v_dim).transpose(1, 2)
        k_rope = k_rope.unsqueeze(1).expand(B, c.n_heads, S, c.rope_dim)  # 所有头共用一份
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)
        scores = (q @ k.transpose(-1, -2)) * self.scale
        probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
        return probs @ v
    # endregion

    # region absorbed
    def _absorbed(self, q_nope, q_rope, c_kv, k_rope, mask):
        """吸收路径：q_C^T (W_UK c) = (W_UK^T q_C)^T c，Σ a_j W_UV c_j = W_UV Σ a_j c_j。"""
        c = self.cfg
        w_uk = self.wk_up.weight.view(c.n_heads, c.nope_dim, c.kv_rank)  # 每头 [d_h, d_c]
        w_uv = self.wv_up.weight.view(c.n_heads, c.v_dim, c.kv_rank)  # 每头 [d_v, d_c]
        q_latent = torch.einsum("bhtd,hdc->bhtc", q_nope, w_uk)  # 查询搬进潜空间
        scores = torch.einsum("bhtc,bsc->bhts", q_latent, c_kv)  # 内容项：直接点积缓存
        scores = scores + torch.einsum("bhtr,bsr->bhts", q_rope, k_rope)  # 位置项
        probs = (scores * self.scale).masked_fill(~mask, float("-inf")).softmax(-1)
        z = torch.einsum("bhts,bsc->bhtc", probs, c_kv)  # 在潜空间里加权求和
        return torch.einsum("bhtc,hvc->bhtv", z, w_uv)  # 最后才上投影成值
    # endregion


def mla_cache_elems_per_token(config: MLAConfig) -> int:
    """MLA 每层每 token 的缓存元素数 d_c + d_R；对比 MHA 的 2·H·d_h。"""
    return config.kv_rank + config.rope_dim
