"""Explicit softmax attention. No nn.MultiheadAttention/Transformer wrappers."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .cache import KVCache
from .config import AttentionConfig
from .layers import RMSNorm
from .masks import attention_mask, masked_softmax
from .position import apply_rope


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    visible: torch.Tensor | None = None,
    dropout: float = 0.0,
    training: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Q [B,H,Q,Dk], K [B,H,K,Dk], V [B,H,K,Dv] -> [B,H,Q,Dv]."""
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or q.shape[:2] != k.shape[:2]
        or k.shape[:3] != v.shape[:3]
        or q.shape[-1] != k.shape[-1]
    ):
        raise ValueError("incompatible attention shapes")
    if visible is None:
        visible = torch.ones((q.shape[-2], k.shape[-2]), dtype=torch.bool, device=q.device)
    if visible.dtype != torch.bool:
        raise ValueError("attention masks use True=visible")
    scores = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5 if scale is None else scale)
    weights = masked_softmax(scores, visible)
    return F.dropout(weights, dropout, training) @ v


def gathered_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Actually gather selected keys: indices [Q,M], no [Q,K] score allocation.

    Same key selection for all batch/head entries; valid [Q,M] masks fillers.
    """
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or q.shape[:2] != k.shape[:2]
        or k.shape[:3] != v.shape[:3]
        or q.shape[-1] != k.shape[-1]
    ):
        raise ValueError("incompatible gathered attention shapes")
    if (
        indices.ndim != 2
        or indices.dtype != torch.long
        or indices.shape[0] != q.shape[-2]
        or indices.shape[1] < 1
    ):
        raise ValueError("indices must be nonempty long [Q,M]")
    if indices.device != q.device or indices.min() < 0 or indices.max() >= k.shape[-2]:
        raise ValueError("sparse index outside key sequence")
    if valid is None:
        valid = torch.ones_like(indices, dtype=torch.bool)
    if valid.shape != indices.shape or valid.dtype != torch.bool or valid.device != q.device:
        raise ValueError("valid must be boolean [Q,M]")
    # Duplicate real indices change normalization; filler duplicates are allowed.
    for row, mask in zip(indices, valid, strict=True):
        if row[mask].unique().numel() != mask.sum():
            raise ValueError("duplicate selected keys")
    selected_k, selected_v = k[:, :, indices, :], v[:, :, indices, :]
    scores = (q.unsqueeze(-2) * selected_k).sum(-1) * (
        q.shape[-1] ** -0.5 if scale is None else scale
    )
    return (masked_softmax(scores, valid).unsqueeze(-1) * selected_v).sum(-2)


class MultiHeadAttention(nn.Module):
    """One implementation for MHA/MQA/GQA, self/cross, full/prefill/decode."""

    def __init__(
        self, dim: int, config: AttentionConfig, dropout: float = 0.0, cross: bool = False
    ) -> None:
        super().__init__()
        self.config, self.dim, self.cross, self.dropout = config, dim, cross, dropout
        h, g, d = config.heads, config.effective_kv_heads, config.head_dim
        self.q = nn.Linear(dim, h * d, bias=False)
        self.k = nn.Linear(dim, g * d, bias=False)
        self.v = nn.Linear(dim, g * d, bias=False)
        self.output = nn.Linear(h * d, dim, bias=False)
        self.q_norm = RMSNorm(d) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(d) if config.qk_norm else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        cache: KVCache | None = None,
        use_cache: bool = False,
        query_offset: int = 0,
        causal: bool = False,
        key_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        c = self.config
        b, t, _ = x.shape
        h, g, d = c.heads, c.effective_kv_heads, c.head_dim
        if not self.cross and cache is None and query_offset != 0:
            raise ValueError("nonzero self-attention position requires a prefix cache")
        if cache is not None:
            if not isinstance(cache, KVCache):
                raise ValueError("expected a KVCache")
            cache.validate(b, g, d, d, x, self.cross)
            if not self.cross and query_offset != cache.length:
                raise ValueError("append position must equal cached prefix length")
        q = self.q_norm(self.q(x).reshape(b, t, h, d).transpose(1, 2))
        qp = torch.arange(query_offset, query_offset + t, device=x.device)
        if c.position.kind == "rope":
            q = apply_rope(q, qp, c.position)
        if self.cross and cache is not None:
            next_cache = cache
        else:
            source = memory if self.cross else x
            if (
                source is None
                or source.ndim != 3
                or source.shape[0] != b
                or source.shape[-1] != self.dim
            ):
                raise ValueError("cross attention requires compatible encoder memory")
            s = source.shape[1]
            k = self.k_norm(self.k(source).reshape(b, s, g, d).transpose(1, 2))
            v = self.v(source).reshape(b, s, g, d).transpose(1, 2)
            kp = torch.arange(
                0 if self.cross else query_offset,
                (0 if self.cross else query_offset) + s,
                device=x.device,
            )
            if c.position.kind == "rope":
                k = apply_rope(k, kp, c.position)
            next_cache = KVCache(k, v, self.cross) if cache is None else cache.append(k, v)
        k, v = next_cache.key, next_cache.value
        visible = attention_mask(
            qp, torch.arange(next_cache.length, device=x.device), c, causal, key_valid
        )
        # ponytail: expanded KV is temporary; grouped GPU kernels avoid the H/G replication.
        k, v = (a.repeat_interleave(h // g, 1) for a in (k, v))
        scale = d**-0.5 * c.position.attention_factor**2
        if c.backend == "sdpa":
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=visible,
                dropout_p=self.dropout if self.training else 0.0,
                scale=scale,
            )
        else:
            y = scaled_dot_product_attention(q, k, v, visible, self.dropout, self.training, scale)
        return self.output(y.transpose(1, 2).reshape(b, t, -1)), next_cache if use_cache else None


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek-style normalized KV latent + shared decoupled rotary key.

    Both paths use the same latent cache. Naive reconstructs K/V; absorbed
    contracts Q with W_UK and absorbs W_UV into W_O, retaining autograd.
    """

    def __init__(
        self, dim: int, config: AttentionConfig, dropout: float = 0.0, cross: bool = False
    ) -> None:
        super().__init__()
        self.config, self.dim, self.cross, self.dropout = config, dim, cross, dropout
        h, d, r, rank = config.heads, config.head_dim, config.rope_dim, config.kv_rank
        if config.q_rank:
            self.q_down = nn.Linear(dim, config.q_rank, bias=False)
            self.q_norm = RMSNorm(config.q_rank)
            self.q_up = nn.Linear(config.q_rank, h * (d + r), bias=False)
        else:
            self.q_down = nn.Identity()
            self.q_norm = nn.Identity()
            self.q_up = nn.Linear(dim, h * (d + r), bias=False)
        self.kv_down = nn.Linear(dim, rank, bias=False)
        self.kv_norm = RMSNorm(rank)
        self.k_rope = nn.Linear(dim, r, bias=False)
        self.k_up = nn.Linear(rank, h * d, bias=False)
        self.v_up = nn.Linear(rank, h * config.value_dim, bias=False)
        self.output = nn.Linear(h * config.value_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        cache: KVCache | None = None,
        use_cache: bool = False,
        query_offset: int = 0,
        causal: bool = False,
        key_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        c = self.config
        b, t, _ = x.shape
        if not self.cross and cache is None and query_offset != 0:
            raise ValueError("nonzero MLA position requires a prefix cache")
        if cache is not None:
            if not isinstance(cache, KVCache):
                raise ValueError("expected a latent KVCache")
            cache.validate(b, 1, c.kv_rank, c.rope_dim, x, self.cross)
            if not self.cross and query_offset != cache.length:
                raise ValueError("append position must equal cached prefix length")
        q = (
            self.q_up(self.q_norm(self.q_down(x)))
            .reshape(b, t, c.heads, c.head_dim + c.rope_dim)
            .transpose(1, 2)
        )
        qc, qr = q.split((c.head_dim, c.rope_dim), -1)
        qp = torch.arange(query_offset, query_offset + t, device=x.device)
        if c.position.kind == "rope":
            qr = apply_rope(qr, qp, c.position)
        if self.cross and cache is not None:
            next_cache = cache
        else:
            source = memory if self.cross else x
            if (
                source is None
                or source.ndim != 3
                or source.shape[0] != b
                or source.shape[-1] != self.dim
            ):
                raise ValueError("cross MLA requires compatible encoder memory")
            latent = self.kv_norm(self.kv_down(source)).unsqueeze(1)
            kr = self.k_rope(source).unsqueeze(1)
            start = 0 if self.cross else query_offset
            kp = torch.arange(start, start + source.shape[1], device=x.device)
            if c.position.kind == "rope":
                kr = apply_rope(kr, kp, c.position)
            next_cache = (
                KVCache(latent, kr, self.cross) if cache is None else cache.append(latent, kr)
            )
        latent, kr = next_cache.key, next_cache.value
        s = next_cache.length
        visible = attention_mask(qp, torch.arange(s, device=x.device), c, causal, key_valid)
        scale = (c.head_dim + c.rope_dim) ** -0.5 * c.position.attention_factor**2
        if c.mla_impl == "naive":
            kc = self.k_up(latent[:, 0]).reshape(b, s, c.heads, c.head_dim).transpose(1, 2)
            v = self.v_up(latent[:, 0]).reshape(b, s, c.heads, c.value_dim).transpose(1, 2)
            scores = (qc @ kc.transpose(-1, -2) + qr @ kr.transpose(-1, -2)) * scale
            p = F.dropout(masked_softmax(scores, visible), self.dropout, self.training)
            y = p @ v
            y = self.output(y.transpose(1, 2).reshape(b, t, -1))
        else:
            wk = self.k_up.weight.reshape(c.heads, c.head_dim, c.kv_rank)
            q_latent = torch.matmul(qc, wk)
            scores = (q_latent @ latent.transpose(-1, -2) + qr @ kr.transpose(-1, -2)) * scale
            p = F.dropout(masked_softmax(scores, visible), self.dropout, self.training)
            context = p @ latent  # [B,H,T,L], no expanded K/V
            wv = self.v_up.weight.reshape(c.heads, c.value_dim, c.kv_rank)
            wo = self.output.weight.T.reshape(c.heads, c.value_dim, self.dim)
            # Recompute from current weights, so optimizer steps cannot leave stale absorbed weights.
            wvo = torch.matmul(wv.transpose(-1, -2), wo)  # [H,L,D]
            y = torch.matmul(context, wvo).sum(1)
        return y, next_cache if use_cache else None
