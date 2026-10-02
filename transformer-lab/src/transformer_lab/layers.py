"""Normalization, FFNs, and explicit expert dispatch/combine."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import BlockConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = x if x.dtype == torch.float64 else x.float()
        return (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + self.eps)).to(
            x.dtype
        ) * self.weight


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        work = x if x.dtype == torch.float64 else x.float()
        centered = work - work.mean(-1, keepdim=True)
        return (centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + self.eps)).to(
            x.dtype
        ) * self.weight + self.bias


def make_norm(kind: str, dim: int) -> RMSNorm | LayerNorm:
    return {"rms": RMSNorm, "layer": LayerNorm}[kind](dim)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, activation: str = "swiglu") -> None:
        super().__init__()
        if activation not in {"relu", "gelu", "glu", "geglu", "swiglu"}:
            raise ValueError("unknown activation")
        self.activation = activation
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.gate = nn.Linear(dim, hidden_dim, bias=False) if activation.endswith("glu") else None
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        function = {
            "relu": F.relu,
            "gelu": F.gelu,
            "glu": torch.sigmoid,
            "geglu": F.gelu,
            "swiglu": F.silu,
        }[self.activation]
        hidden = function(self.up(x)) if self.gate is None else function(self.gate(x)) * self.up(x)
        return self.down(hidden)


class MixtureOfExperts(nn.Module):
    """No token dropping; selection bias affects IDs, never combination weights."""

    def __init__(self, dim: int, config: BlockConfig) -> None:
        super().__init__()
        self.config = config
        self.router = nn.Linear(dim, config.experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(dim, config.ff_dim, config.activation) for _ in range(config.experts)]
        )
        self.shared = nn.ModuleList(
            [
                FeedForward(dim, config.ff_dim, config.activation)
                for _ in range(config.shared_experts)
            ]
        )
        self.register_buffer("selection_bias", torch.zeros(config.experts))

    def route(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        work = logits if logits.dtype == torch.float64 else logits.float()
        probabilities = (
            work.softmax(-1) if self.config.router_score == "softmax" else work.sigmoid()
        )
        ids = (probabilities + self.selection_bias).topk(self.config.top_k, -1).indices
        weights = probabilities.gather(-1, ids)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        return ids, weights.to(x.dtype), probabilities

    def forward(
        self, x: torch.Tensor, valid: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = x.reshape(-1, x.shape[-1])
        rows = (
            torch.arange(flat.shape[0], device=x.device)
            if valid is None
            else valid.reshape(-1).nonzero().flatten()
        )
        tokens = flat[rows]
        if tokens.shape[0] == 0:
            return (
                torch.zeros_like(x),
                x.sum() * 0,
                torch.zeros(len(self.experts), device=x.device, dtype=torch.long),
            )
        ids, weights, probabilities = self.route(tokens)
        result = torch.zeros_like(tokens)
        # ponytail: expert loop is a transparent CPU reference; use grouped GEMM when dispatch dominates.
        for expert_id, expert in enumerate(self.experts):
            token, slot = (ids == expert_id).nonzero(as_tuple=True)
            if token.numel():
                result = result.index_add(
                    0, token, expert(tokens[token]) * weights[token, slot, None]
                )
        for expert in self.shared:
            result = result + expert(tokens)
        counts = torch.bincount(ids.flatten(), minlength=len(self.experts))
        fractions = counts.to(probabilities.dtype) / ids.numel()
        auxiliary = len(self.experts) * (fractions * probabilities.mean(0)).sum()
        if self.config.balance != "aux":
            auxiliary = auxiliary * 0
        output = torch.zeros_like(flat).index_copy(0, rows, result).reshape_as(x)
        return output, auxiliary, counts

    @torch.no_grad()
    def update_balance(self, counts: torch.Tensor) -> None:
        """Call once after the optimizer step; sum counts across ranks/microbatches first."""
        if self.config.balance == "bias" and counts.sum() > 0:
            self.selection_bias.add_(
                self.config.balance_rate * (counts.float().mean() - counts).sign()
            )
            self.selection_bias.sub_(self.selection_bias.mean())


Routing = tuple[tuple[MixtureOfExperts, torch.Tensor], ...]
