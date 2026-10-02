"""Modern Transformer primitives, without high-level attention wrappers."""

from .attention import gathered_attention, scaled_dot_product_attention
from .config import AttentionConfig, BlockConfig, ModelConfig, PositionConfig
from .models import EncoderMemory, ModelOutput, Transformer, TransformerBlock, make_attention
from .training import language_model_loss, next_token_loss, teacher_forcing, token_loss

__all__ = [
    "PositionConfig",
    "AttentionConfig",
    "BlockConfig",
    "ModelConfig",
    "Transformer",
    "TransformerBlock",
    "EncoderMemory",
    "ModelOutput",
    "make_attention",
    "scaled_dot_product_attention",
    "gathered_attention",
    "teacher_forcing",
    "token_loss",
    "next_token_loss",
    "language_model_loss",
]
