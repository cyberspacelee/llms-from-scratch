"""inference 子包公开入口；具体公式与 Shape 见各实现模块。"""

from .compression import compress_sequence, compressed_causal_attention
from .speculative import greedy_speculative_generate

__all__ = ["compress_sequence", "compressed_causal_attention", "greedy_speculative_generate"]
