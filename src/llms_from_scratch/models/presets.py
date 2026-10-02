"""The two course baselines share the same Transformer implementation."""

from ..config import AttentionConfig, BlockConfig, ModelConfig, PositionConfig


def decoder_config(
    vocab_size: int,
    dim: int = 32,
    ff_dim: int = 64,
    layers: int = 2,
    heads: int = 4,
    kv_heads: int = 2,
    head_dim: int = 8,
    max_length: int = 64,
    tie_embeddings: bool = True,
) -> ModelConfig:
    """Modern baseline: Pre-RMSNorm, GQA, adjacent-pair RoPE and SwiGLU."""
    attention = AttentionConfig(heads=heads, kv_heads=kv_heads, head_dim=head_dim)
    return ModelConfig(
        vocab_size=vocab_size,
        dim=dim,
        layers=layers,
        max_length=max_length,
        tie_embeddings=tie_embeddings,
        block=BlockConfig(attention=attention, ff_dim=ff_dim),
    )


def basic_decoder_config(
    vocab_size: int,
    dim: int = 32,
    ff_dim: int = 64,
    layers: int = 2,
    heads: int = 4,
    max_length: int = 32,
) -> ModelConfig:
    """P4 baseline: learned positions, Pre-LayerNorm, MHA/GELU, biased untied head."""
    if type(heads) is not int or heads < 1 or type(dim) is not int or dim < 1 or dim % heads:
        raise ValueError("basic MHA width must divide into heads")
    attention = AttentionConfig(
        kind="mha",
        heads=heads,
        head_dim=dim // heads,
        position=PositionConfig(kind="learned"),
        bias=True,
    )
    return ModelConfig(
        vocab_size=vocab_size,
        dim=dim,
        layers=layers,
        max_length=max_length,
        tie_embeddings=False,
        head_bias=True,
        block=BlockConfig(
            attention=attention, ff_dim=ff_dim, activation="gelu", norm="layer", bias=True
        ),
    )
