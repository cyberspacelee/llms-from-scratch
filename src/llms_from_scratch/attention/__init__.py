"""attention 子包公开入口；具体公式与 Shape 见各实现模块。"""

from .mla import MultiHeadLatentAttention
from .recurrent import RecurrentAttention
from .softmax import MultiHeadAttention, gathered_attention, scaled_dot_product_attention

__all__ = [
    "MultiHeadAttention",
    "gathered_attention",
    "scaled_dot_product_attention",
    "MultiHeadLatentAttention",
    "RecurrentAttention",
]
