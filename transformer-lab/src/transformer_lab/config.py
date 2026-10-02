"""Small, explicit configs; tuples describe actual per-layer experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PositionConfig:
    kind: str = "rope"  # none, sinusoidal, rope
    base: float = 10000.0
    scaling: str = "none"  # none, linear, ntk, yarn (fixed for the entire session)
    factor: float = 1.0
    original_length: int = 2048
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    attention_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.kind not in {"none", "sinusoidal", "rope"}:
            raise ValueError("unknown position encoding")
        if self.scaling not in {"none", "linear", "ntk", "yarn"}:
            raise ValueError("unknown RoPE scaling")
        for v in (self.base, self.factor, self.beta_fast, self.beta_slow, self.attention_factor):
            if not math.isfinite(v) or v <= 0:
                raise ValueError("position values must be finite and positive")
        if (
            self.base <= 1
            or self.factor < 1
            or self.original_length < 1
            or self.beta_fast <= self.beta_slow
        ):
            raise ValueError("invalid RoPE base, extension factor or correction range")
        if self.kind != "rope" and self.scaling != "none":
            raise ValueError("scaling only applies to RoPE")


@dataclass(frozen=True)
class AttentionConfig:
    kind: str = "gqa"  # mha, mqa, gqa, mla, linear, delta, gated_delta
    heads: int = 4
    kv_heads: int = 2
    head_dim: int = 8
    pattern: str = "global"  # global, sliding, local, block_sparse, token_sparse
    window: int = 8
    block_size: int = 4
    global_stride: int = 8
    qk_norm: bool = False
    position: PositionConfig = field(default_factory=PositionConfig)
    kv_rank: int = 8
    q_rank: int = 8  # 0 means direct Q projection
    rope_dim: int = 4
    value_dim: int = 8
    mla_impl: str = "absorbed"  # naive shares weights and latent cache with absorbed
    backend: str = "manual"  # sdpa only for MHA/MQA/GQA comparisons

    def __post_init__(self) -> None:
        if self.kind not in {"mha", "mqa", "gqa", "mla", "linear", "delta", "gated_delta"}:
            raise ValueError("unknown attention kind")
        for name in (
            "heads",
            "kv_heads",
            "head_dim",
            "window",
            "block_size",
            "global_stride",
            "kv_rank",
            "rope_dim",
            "value_dim",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.q_rank) is not int or self.q_rank < 0:
            raise ValueError("q_rank must be a nonnegative integer")
        if self.kind == "gqa" and self.heads % self.kv_heads:
            raise ValueError("query heads must be divisible by KV heads")
        if self.pattern not in {"global", "sliding", "local", "block_sparse", "token_sparse"}:
            raise ValueError("unknown attention pattern")
        if self.position.kind == "rope" and (self.head_dim % 2 or self.rope_dim % 2):
            raise ValueError("rotary dimensions must be even")
        if self.backend not in {"manual", "sdpa"} or self.mla_impl not in {"naive", "absorbed"}:
            raise ValueError("unknown implementation")
        if self.kind in {"mla", "linear", "delta", "gated_delta"} and self.backend != "manual":
            raise ValueError("this reference path only supports the manual backend")
        if self.kind in {"linear", "delta", "gated_delta"} and (
            self.pattern != "global" or self.position.kind != "none"
        ):
            raise ValueError("recurrent mixers require global pattern and position=none")
        if self.kind == "mla" and self.qk_norm:
            raise ValueError("MLA uses latent RMSNorm; expanded QK-Norm prevents simple absorption")

    @property
    def effective_kv_heads(self) -> int:
        return self.heads if self.kind == "mha" else 1 if self.kind == "mqa" else self.kv_heads


@dataclass(frozen=True)
class BlockConfig:
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    ff_dim: int = 64
    activation: str = "swiglu"  # relu, gelu, glu, geglu, swiglu
    norm: str = "rms"  # layer, rms
    norm_order: str = "pre"  # pre, post
    dropout: float = 0.0
    experts: int = 0  # 0 = dense FFN
    top_k: int = 2
    shared_experts: int = 0
    router_score: str = "softmax"  # softmax, sigmoid
    balance: str = "aux"  # aux, bias, none
    balance_rate: float = 0.01
    residual: str = "standard"  # standard, gated (HC needs a different stream shape)

    def __post_init__(self) -> None:
        if type(self.ff_dim) is not int or self.ff_dim <= 0:
            raise ValueError("ff_dim must be positive")
        if self.activation not in {"relu", "gelu", "glu", "geglu", "swiglu"}:
            raise ValueError("unknown FFN activation")
        if self.norm not in {"layer", "rms"} or self.norm_order not in {"pre", "post"}:
            raise ValueError("unknown normalization")
        if (
            not 0 <= self.dropout < 1
            or not math.isfinite(self.balance_rate)
            or self.balance_rate < 0
        ):
            raise ValueError("invalid dropout or balance rate")
        if any(type(n) is not int or n < 0 for n in (self.experts, self.shared_experts)):
            raise ValueError("expert counts must be nonnegative integers")
        if (
            type(self.top_k) is not int
            or self.top_k < 1
            or (self.experts and self.top_k > self.experts)
        ):
            raise ValueError("invalid top-k")
        if not self.experts and self.shared_experts:
            raise ValueError("shared experts require MoE")
        if self.router_score not in {"softmax", "sigmoid"} or self.balance not in {
            "aux",
            "bias",
            "none",
        }:
            raise ValueError("unknown routing configuration")
        if self.residual not in {"standard", "gated"}:
            raise ValueError("unknown residual connection")


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "decoder"  # encoder, decoder, encoder_decoder
    vocab_size: int = 64
    dim: int = 32
    layers: int = 2
    max_length: int = 256
    block: BlockConfig = field(default_factory=BlockConfig)
    layer_blocks: tuple[BlockConfig, ...] = ()
    encoder_layers: int = 2
    encoder_block: BlockConfig | None = None
    encoder_causal: bool = False
    cross_attention: AttentionConfig | None = None
    cross_causal: bool = False  # aligned source/output timelines, not ordinary translation
    tie_embeddings: bool = True
    mtp_depth: int = 0

    def __post_init__(self) -> None:
        if self.architecture not in {"encoder", "decoder", "encoder_decoder"}:
            raise ValueError("unknown architecture")
        if any(
            type(n) is not int or n < 1
            for n in (self.vocab_size, self.dim, self.layers, self.max_length, self.encoder_layers)
        ):
            raise ValueError("model dimensions must be positive integers")
        if type(self.mtp_depth) is not int or self.mtp_depth < 0:
            raise ValueError("MTP depth must be nonnegative")
        if self.layer_blocks and len(self.layer_blocks) != self.layers:
            raise ValueError("layer_blocks must specify every layer")
        blocks = self.layer_blocks or (self.block,) * self.layers
        if any(b.attention.position.kind != self.block.attention.position.kind for b in blocks):
            # Sinusoidal is added at input; RoPE/NoPE may be mixed per layer.
            if (
                any(b.attention.position.kind == "sinusoidal" for b in blocks)
                or self.block.attention.position.kind == "sinusoidal"
            ):
                raise ValueError("sinusoidal input encoding must be consistent")
        if self.cross_attention and self.cross_attention.kind in {"linear", "delta", "gated_delta"}:
            raise ValueError("cross-attention must be a softmax mixer")
        if self.architecture == "encoder" and self.mtp_depth:
            raise ValueError("MTP requires a causal decoder")
