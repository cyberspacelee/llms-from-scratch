"""Encoder, decoder and encoder-decoder share the same explicit blocks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .attention import MultiHeadAttention, MultiHeadLatentAttention
from .cache import LayerCache, ModelCache
from .config import AttentionConfig, BlockConfig, ModelConfig
from .layers import FeedForward, MixtureOfExperts, Routing, make_norm
from .position import sinusoidal
from .recurrent import RecurrentAttention


def make_attention(
    dim: int, config: AttentionConfig, dropout: float = 0.0, cross: bool = False
) -> MultiHeadAttention | MultiHeadLatentAttention | RecurrentAttention:
    if config.kind in {"mha", "mqa", "gqa"}:
        return MultiHeadAttention(dim, config, dropout, cross)
    if config.kind == "mla":
        return MultiHeadLatentAttention(dim, config, dropout, cross)
    if cross:
        raise ValueError("cross attention must use softmax")
    return RecurrentAttention(dim, config)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        config: BlockConfig,
        cross_attention: AttentionConfig | None = None,
        cross_causal: bool = False,
    ) -> None:
        super().__init__()
        self.config, self.cross_causal = config, cross_causal
        self.attention = make_attention(dim, config.attention, config.dropout)
        self.cross = (
            make_attention(dim, cross_attention, config.dropout, True) if cross_attention else None
        )
        self.ff = (
            MixtureOfExperts(dim, config)
            if config.experts
            else FeedForward(dim, config.ff_dim, config.activation)
        )
        self.norms = nn.ModuleList(
            [make_norm(config.norm, dim) for _ in range(3 if self.cross else 2)]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.residual_gates = (
            nn.Parameter(torch.zeros(len(self.norms))) if config.residual == "gated" else None
        )

    def branch_input(self, x: torch.Tensor, index: int) -> torch.Tensor:
        return self.norms[index](x) if self.config.norm_order == "pre" else x

    def combine(self, x: torch.Tensor, branch: torch.Tensor, index: int) -> torch.Tensor:
        branch = self.dropout(branch)
        if self.residual_gates is not None:
            branch = branch * self.residual_gates[index].sigmoid()
        x = x + branch
        return x if self.config.norm_order == "pre" else self.norms[index](x)

    def forward(
        self,
        x: torch.Tensor,
        causal: bool,
        valid: torch.Tensor,
        cache: LayerCache | None = None,
        use_cache: bool = False,
        offset: int = 0,
        memory: torch.Tensor | None = None,
        memory_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, LayerCache | None, torch.Tensor, Routing]:
        branch, self_cache = self.attention(
            self.branch_input(x, 0),
            cache=None if cache is None else cache.self_attention,
            use_cache=use_cache,
            query_offset=offset,
            causal=causal,
            key_valid=valid,
        )
        x = self.combine(x, branch, 0)
        cross_cache = None
        if self.cross is not None:
            branch, cross_cache = self.cross(
                self.branch_input(x, 1),
                memory=memory,
                cache=None if cache is None else cache.cross_attention,
                use_cache=use_cache,
                query_offset=offset,
                causal=self.cross_causal,
                key_valid=memory_valid,
            )
            x = self.combine(x, branch, 1)
        index = len(self.norms) - 1
        if isinstance(self.ff, MixtureOfExperts):
            branch, auxiliary, counts = self.ff(
                self.branch_input(x, index), valid[:, offset : offset + x.shape[1]]
            )
            routing = ((self.ff, counts),)
        else:
            branch = self.ff(self.branch_input(x, index))
            auxiliary, routing = x.new_zeros(()), ()
        return (
            self.combine(x, branch, index),
            LayerCache(self_cache, cross_cache) if use_cache else None,
            auxiliary,
            routing,
        )


@dataclass(frozen=True)
class EncoderMemory:
    hidden: torch.Tensor
    valid: torch.Tensor
    layers: tuple[LayerCache, ...] | None = None  # causal encoder streaming only
    auxiliary_loss: torch.Tensor | None = None
    routing: Routing = ()


@dataclass(frozen=True)
class ModelOutput:
    logits: torch.Tensor
    hidden: torch.Tensor
    cache: ModelCache | None
    auxiliary_loss: torch.Tensor
    routing: Routing = ()


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)
        configs = config.layer_blocks or (config.block,) * config.layers
        cross = config.cross_attention or config.block.attention
        if config.architecture == "encoder_decoder" and cross.kind in {
            "linear",
            "delta",
            "gated_delta",
        }:
            cross = AttentionConfig()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.dim,
                    c,
                    cross if config.architecture == "encoder_decoder" else None,
                    config.cross_causal,
                )
                for c in configs
            ]
        )
        self.norm = (
            make_norm(configs[-1].norm, config.dim)
            if configs[-1].norm_order == "pre"
            else nn.Identity()
        )
        if config.architecture == "encoder_decoder":
            ec = config.encoder_block or config.block
            self.encoder_blocks = nn.ModuleList(
                [TransformerBlock(config.dim, ec) for _ in range(config.encoder_layers)]
            )
            self.encoder_norm = (
                make_norm(ec.norm, config.dim) if ec.norm_order == "pre" else nn.Identity()
            )
        else:
            self.encoder_blocks = nn.ModuleList()
        self.head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.mtp_depth:
            from .objectives import MultiTokenPrediction

            self.mtp = MultiTokenPrediction(config.dim, config.block, config.mtp_depth)
        else:
            self.mtp = None
        self.apply(self._initialize)
        if config.tie_embeddings:
            self.head.weight = self.embedding.weight

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def validate_ids(self, ids: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
        if (
            ids.ndim != 2
            or ids.dtype != torch.long
            or ids.numel() == 0
            or ids.device != self.embedding.weight.device
        ):
            raise ValueError("IDs must be nonempty long [B,T] on model device")
        if ids.min() < 0 or ids.max() >= self.config.vocab_size:
            raise ValueError("token outside vocabulary")
        if valid is None:
            return torch.ones_like(ids, dtype=torch.bool)
        if valid.dtype != torch.bool or valid.shape != ids.shape or valid.device != ids.device:
            raise ValueError("valid must be boolean [B,T] on input device")
        return valid

    def embed(self, ids: torch.Tensor, offset: int, block: BlockConfig) -> torch.Tensor:
        if offset + ids.shape[1] > self.config.max_length:
            raise ValueError("sequence exceeds max_length")
        x = self.embedding(ids)
        if block.attention.position.kind == "sinusoidal":
            positions = torch.arange(offset, offset + ids.shape[1], device=ids.device)
            dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
            x = math.sqrt(self.config.dim) * x + sinusoidal(positions, self.config.dim, dtype).to(
                x.dtype
            )
        return nn.functional.dropout(x, block.dropout, self.training)

    def encode(
        self,
        ids: torch.Tensor,
        valid: torch.Tensor | None = None,
        past: EncoderMemory | None = None,
        use_cache: bool = False,
    ) -> EncoderMemory:
        """Reuse bidirectional memory; append is only valid for a causal encoder."""
        if self.config.architecture != "encoder_decoder":
            raise ValueError("encode() belongs to encoder-decoder models")
        valid = self.validate_ids(ids, valid)
        ec = self.config.encoder_block or self.config.block
        if past is not None and (not self.config.encoder_causal or past.layers is None):
            raise ValueError("bidirectional encoders cannot append cached hidden states")
        offset = 0 if past is None else past.hidden.shape[1]
        if past is not None:
            if past.hidden.shape[0] != ids.shape[0] or len(past.layers) != len(self.encoder_blocks):
                raise ValueError("encoder cache batch/layer count mismatch")
            valid = torch.cat((past.valid, valid), 1)
        x = self.embed(ids, offset, ec)
        layers, routing, auxiliary = [], (), x.new_zeros(())
        for i, block in enumerate(self.encoder_blocks):
            x, layer, aux, stats = block(
                x,
                self.config.encoder_causal,
                valid,
                None if past is None else past.layers[i],
                use_cache and self.config.encoder_causal,
                offset,
            )
            layers.append(layer)
            auxiliary, routing = auxiliary + aux, routing + stats
        x = self.encoder_norm(x)
        if past is not None:
            x = torch.cat((past.hidden, x), 1)
        return EncoderMemory(
            x,
            valid,
            tuple(layers) if use_cache and self.config.encoder_causal else None,
            auxiliary,
            routing,
        )

    def forward(
        self,
        ids: torch.Tensor,
        *,
        source_ids: torch.Tensor | None = None,
        source_valid: torch.Tensor | None = None,
        memory: EncoderMemory | None = None,
        valid: torch.Tensor | None = None,
        cache: ModelCache | None = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        c = self.config
        current_valid = self.validate_ids(ids, valid)
        causal = c.architecture != "encoder" or c.encoder_causal
        if not causal and (use_cache or cache is not None):
            raise ValueError("bidirectional encoder forwards cannot append KV cache")
        if cache is not None:
            if (
                not use_cache
                or len(cache.layers) != len(self.blocks)
                or cache.valid.shape[0] != ids.shape[0]
                or cache.valid.dtype != torch.bool
                or cache.valid.device != ids.device
            ):
                raise ValueError("invalid model cache or use_cache=False")
            if any(layer.self_attention.length != cache.length for layer in cache.layers):
                raise ValueError("all layer positions must agree")
            if source_ids is not None or memory is not None or source_valid is not None:
                raise ValueError(
                    "cached session owns encoder memory; start a new session to change source"
                )
        offset = 0 if cache is None else cache.length
        full_valid = current_valid if cache is None else torch.cat((cache.valid, current_valid), 1)
        auxiliary, routing = self.embedding.weight.new_zeros(()), ()
        if c.architecture == "encoder_decoder":
            if cache is not None:
                hidden_memory, memory_valid = cache.memory, cache.memory_valid
            else:
                if source_ids is not None and memory is not None:
                    raise ValueError("provide source_ids or prepared memory, not both")
                if memory is None:
                    if source_ids is None:
                        raise ValueError("encoder-decoder requires source_ids or memory")
                    memory = self.encode(source_ids, source_valid)
                elif source_valid is not None:
                    raise ValueError("prepared memory already contains source validity")
                hidden_memory, memory_valid = memory.hidden, memory.valid
                if memory.auxiliary_loss is not None:
                    auxiliary, routing = auxiliary + memory.auxiliary_loss, routing + memory.routing
            if (
                hidden_memory is None
                or hidden_memory.ndim != 3
                or hidden_memory.shape[0] != ids.shape[0]
                or hidden_memory.shape[-1] != c.dim
                or hidden_memory.device != ids.device
                or hidden_memory.dtype != self.embedding.weight.dtype
            ):
                raise ValueError("encoder memory is incompatible")
            if (
                memory_valid is None
                or memory_valid.shape != hidden_memory.shape[:2]
                or memory_valid.dtype != torch.bool
                or memory_valid.device != ids.device
            ):
                raise ValueError("invalid encoder validity mask")
        else:
            if source_ids is not None or memory is not None or source_valid is not None:
                raise ValueError("source memory is only used by encoder-decoder")
            hidden_memory = memory_valid = None
        x = self.embed(ids, offset, c.block)
        layers = []
        for i, block in enumerate(self.blocks):
            x, layer, aux, stats = block(
                x,
                causal,
                full_valid,
                None if cache is None else cache.layers[i],
                use_cache,
                offset,
                hidden_memory,
                memory_valid,
            )
            layers.append(layer)
            auxiliary, routing = auxiliary + aux, routing + stats
        x = self.norm(x)
        result_cache = (
            ModelCache(tuple(layers), full_valid, hidden_memory, memory_valid)
            if use_cache
            else None
        )
        return ModelOutput(self.head(x), x, result_cache, auxiliary, routing)

    @torch.no_grad()
    def update_router_bias(self, routing: Routing) -> None:
        for expert, counts in routing:
            expert.update_balance(counts)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int,
        *,
        source_ids: torch.Tensor | None = None,
        memory: EncoderMemory | None = None,
        source_valid: torch.Tensor | None = None,
        eos_id: int | None = None,
        cached: bool = True,
    ) -> torch.Tensor:
        """Deterministic greedy reference, equal-length unpadded prompts."""
        if self.config.architecture == "encoder":
            raise ValueError("generation requires a decoder")
        self.validate_ids(prompt, None)
        if (
            type(max_new_tokens) is not int
            or max_new_tokens < 0
            or prompt.shape[1] + max_new_tokens > self.config.max_length
        ):
            raise ValueError("generation length exceeds context")
        if eos_id is not None and (
            type(eos_id) is not int or not 0 <= eos_id < self.config.vocab_size
        ):
            raise ValueError("invalid EOS")
        was_training = self.training
        self.eval()
        result, cache = prompt.clone(), None
        finished = torch.zeros(prompt.shape[0], device=prompt.device, dtype=torch.bool)
        try:
            if self.config.architecture == "encoder_decoder" and memory is None:
                if source_ids is None:
                    raise ValueError("generation requires encoder source")
                memory = self.encode(source_ids, source_valid)
            for _ in range(max_new_tokens):
                current = result[:, -1:] if cached and cache is not None else result
                output = self(
                    current, memory=memory if cache is None else None, cache=cache, use_cache=cached
                )
                cache = output.cache
                token = output.logits[:, -1].argmax(-1)
                if eos_id is not None:
                    token = torch.where(finished, eos_id, token)
                    finished |= token == eos_id
                result = torch.cat((result, token[:, None]), 1)
                if finished.all():
                    break
        finally:
            self.train(was_training)
        return result
