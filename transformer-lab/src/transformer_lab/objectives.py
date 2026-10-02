"""Teacher forcing, NTP and sequential future-token-conditioned MTP."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .config import BlockConfig
from .layers import Routing

if TYPE_CHECKING:
    from .model import ModelOutput, Transformer
import torch
from torch import nn
from torch.nn import functional as F

from .layers import RMSNorm


def teacher_forcing(
    targets: torch.Tensor, bos_id: int, pad_id: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Targets [B,T] include EOS; output [BOS,y0,...,y(T-2)], same targets."""
    if (
        targets.ndim != 2
        or targets.dtype != torch.long
        or targets.numel() == 0
        or type(bos_id) is not int
        or bos_id < 0
    ):
        raise ValueError("expected nonempty targets and nonnegative BOS")
    inputs = torch.cat((torch.full_like(targets[:, :1], bos_id), targets[:, :-1]), 1)
    valid = torch.ones_like(targets, dtype=torch.bool) if pad_id is None else targets != pad_id
    return inputs, targets, valid


def token_loss(
    logits: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    if logits.ndim != 3 or targets.shape != logits.shape[:2] or targets.dtype != torch.long:
        raise ValueError("expected logits [B,T,V] and long targets [B,T]")
    if valid is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    if valid.shape != targets.shape or valid.dtype != torch.bool or valid.device != targets.device:
        raise ValueError("invalid loss mask")
    selected = targets[valid]
    if selected.numel() and (selected.min() < 0 or selected.max() >= logits.shape[-1]):
        raise ValueError("target outside vocabulary")
    if not selected.numel():
        return logits.sum() * 0
    return F.cross_entropy(logits[valid], selected)


def next_token_loss(
    logits: torch.Tensor, tokens: torch.Tensor, valid: torch.Tensor | None = None
) -> torch.Tensor:
    mask = None if valid is None else valid[:, :-1] & valid[:, 1:]
    return token_loss(logits[:, :-1], tokens[:, 1:], mask)


class MultiTokenPrediction(nn.Module):
    """Depth j consumes the true future token x(t+j), predicts x(t+j+1).

    Shared embedding/output weights live in the parent model, avoiding duplicate
    state_dict entries. Each depth has its own projection and Transformer block.
    """

    def __init__(self, dim: int, config: BlockConfig, depth: int) -> None:
        super().__init__()
        from .model import TransformerBlock

        self.hidden_norms = nn.ModuleList([RMSNorm(dim) for _ in range(depth)])
        self.token_norms = nn.ModuleList([RMSNorm(dim) for _ in range(depth)])
        self.projections = nn.ModuleList(
            [nn.Linear(2 * dim, dim, bias=False) for _ in range(depth)]
        )
        self.blocks = nn.ModuleList([TransformerBlock(dim, config) for _ in range(depth)])
        self.output_norms = nn.ModuleList([RMSNorm(dim) for _ in range(depth)])

    def forward(
        self,
        hidden: torch.Tensor,
        tokens: torch.Tensor,
        embedding: nn.Embedding,
        head: nn.Linear,
        valid: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor, Routing]:
        if hidden.shape[:2] != tokens.shape or tokens.ndim != 2:
            raise ValueError("MTP needs aligned hidden states and tokens")
        valid = torch.ones_like(tokens, dtype=torch.bool) if valid is None else valid
        h, outputs, auxiliary, routing = hidden, [], hidden.new_zeros(()), ()
        active = valid
        for i, block in enumerate(self.blocks):
            if h.shape[1] < 3:
                break
            h = h[:, :-1]
            future = embedding(tokens[:, i + 1 :])
            active = active[:, :-1] & valid[:, i + 1 :]
            h = self.projections[i](
                torch.cat((self.hidden_norms[i](h), self.token_norms[i](future)), -1)
            )
            h, _, aux, stats = block(h, True, active)
            outputs.append(head(self.output_norms[i](h))[:, :-1])
            auxiliary, routing = auxiliary + aux, routing + stats
        return tuple(outputs), auxiliary, routing


def language_model_loss(
    model: Transformer,
    output: ModelOutput,
    tokens: torch.Tensor,
    valid: torch.Tensor | None = None,
    mtp_weight: float = 0.3,
    auxiliary_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], Routing]:
    if mtp_weight < 0 or auxiliary_weight < 0:
        raise ValueError("loss weights must be nonnegative")
    losses = {"ntp": next_token_loss(output.logits, tokens, valid)}
    auxiliary = output.auxiliary_loss
    routing = output.routing
    if model.mtp is not None:
        predictions, aux, stats = model.mtp(
            output.hidden, tokens, model.embedding, model.head, valid
        )
        auxiliary = auxiliary + aux
        routing = routing + stats
        active = torch.ones_like(tokens, dtype=torch.bool) if valid is None else valid
        for depth, logits in enumerate(predictions, 2):
            mask = active[:, : tokens.shape[1] - depth].clone()
            for offset in range(1, depth + 1):
                mask &= active[:, offset : tokens.shape[1] - depth + offset]
            losses[f"mtp_{depth}"] = token_loss(logits, tokens[:, depth:], mask)
    mtp = [value for name, value in losses.items() if name.startswith("mtp_")]
    total = losses["ntp"] + auxiliary_weight * auxiliary
    if mtp:
        total = total + mtp_weight * torch.stack(mtp).mean()
    return total, losses, routing
