"""Token-by-token reference for linear attention and the (gated) delta rule."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .cache import RecurrentCache
from .config import AttentionConfig


class RecurrentAttention(nn.Module):
    def __init__(self, dim: int, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        h, d = config.heads, config.head_dim
        self.q = nn.Linear(dim, h * d, bias=False)
        self.k = nn.Linear(dim, h * d, bias=False)
        self.v = nn.Linear(dim, h * config.value_dim, bias=False)
        self.beta = nn.Linear(dim, h) if config.kind != "linear" else None
        self.decay = nn.Linear(dim, h) if config.kind == "gated_delta" else None
        self.output = nn.Linear(h * config.value_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        cache: RecurrentCache | None = None,
        use_cache: bool = False,
        query_offset: int = 0,
        causal: bool = True,
        key_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RecurrentCache | None]:
        if not causal or memory is not None:
            raise ValueError("recurrent reference supports causal self attention only")
        c = self.config
        b, t, _ = x.shape
        shape = (b, c.heads, c.head_dim, c.value_dim)
        if cache is None and query_offset != 0:
            raise ValueError("nonzero recurrent position requires a state cache")
        if cache is not None:
            if not isinstance(cache, RecurrentCache):
                raise ValueError("expected a RecurrentCache")
            if (
                cache.state.shape != shape
                or cache.state.device != x.device
                or cache.state.dtype != x.dtype
                or cache.length != query_offset
            ):
                raise ValueError("invalid recurrent state or position")
            if c.kind == "linear" and (
                cache.normalizer is None
                or cache.normalizer.shape != shape[:-1]
                or cache.normalizer.device != x.device
                or cache.normalizer.dtype != x.dtype
            ):
                raise ValueError("invalid linear normalizer state")
            if c.kind != "linear" and cache.normalizer is not None:
                raise ValueError("delta state has no normalizer")
        state = x.new_zeros(shape) if cache is None else cache.state
        normalizer = x.new_zeros(shape[:-1]) if cache is None else cache.normalizer

        def project(layer: nn.Linear, width: int) -> torch.Tensor:
            return layer(x).reshape(b, t, c.heads, width).transpose(1, 2)

        q, k, v = (
            project(self.q, c.head_dim),
            project(self.k, c.head_dim),
            project(self.v, c.value_dim),
        )
        if c.kind == "linear":
            q, k = F.elu(q) + 1, F.elu(k) + 1
        else:
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        beta = self.beta(x).sigmoid().transpose(1, 2) if self.beta is not None else None
        decay = self.decay(x).sigmoid().transpose(1, 2) if self.decay is not None else None
        if key_valid is not None and (
            key_valid.shape != (b, query_offset + t)
            or key_valid.dtype != torch.bool
            or key_valid.device != x.device
        ):
            raise ValueError("key_valid must cover the complete recurrent prefix")
        outputs = []
        # ponytail: sequential recurrence demonstrates fixed state; chunkwise parallel kernels accelerate training.
        for i in range(t):
            ki, qi, vi = k[:, :, i], q[:, :, i], v[:, :, i]
            active = (
                torch.ones((b, 1, 1, 1), device=x.device, dtype=torch.bool)
                if key_valid is None
                else key_valid[:, query_offset + i, None, None, None]
            )
            if c.kind == "linear":
                updated = state + ki.unsqueeze(-1) * vi.unsqueeze(-2)
                z = normalizer + ki
                normalizer = torch.where(active.squeeze(-1), z, normalizer)
            else:
                decayed = state if decay is None else decay[:, :, i, None, None] * state
                residual = vi - (ki.unsqueeze(-2) @ decayed).squeeze(-2)
                updated = decayed + beta[:, :, i, None, None] * ki.unsqueeze(
                    -1
                ) * residual.unsqueeze(-2)
            state = torch.where(active, updated, state)
            yi = (qi.unsqueeze(-2) @ state).squeeze(-2)
            if c.kind == "linear":
                yi = yi / (qi * normalizer).sum(-1, keepdim=True).clamp_min(1e-6)
            outputs.append(yi * active.squeeze(-1))
        y = torch.stack(outputs, 2).transpose(1, 2).reshape(b, t, -1)
        result_cache = RecurrentCache(
            state, normalizer if c.kind == "linear" else None, query_offset + t
        )
        return self.output(y), result_cache if use_cache else None
