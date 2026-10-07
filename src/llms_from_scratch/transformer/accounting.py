"""按公式数出 ``model.GPT`` 的参数量与前向 FLOPs，测试用真实模型核对。

约定：一次 [m, k] @ [k, n] 的矩阵乘法计 2mkn FLOPs（每个乘加算 2 次）；
只统计矩阵乘法，忽略归一化、激活、softmax、RoPE 等逐元素运算（它们是低阶项）。
"""

from __future__ import annotations

from .model import GPTConfig


# region params
def count_parameters(config: GPTConfig) -> dict[str, int]:
    d, L, V = config.d_model, config.n_layers, config.vocab_size
    d_kv = config.n_kv_heads * config.head_dim  # K、V 投影的输出宽度（GQA 时小于 d）
    attn = d * d + 2 * d * d_kv + d * d  # W_q, W_k, W_v, W_o
    ffn = 3 * d * config.d_ff  # SwiGLU 的 W1, W3, W2
    norms = 2 * d  # 每块两个 RMSNorm 的增益
    parts = {
        "embedding": V * d,
        "attention": L * attn,
        "ffn": L * ffn,
        "norms": L * norms + d,  # 再加 final norm
        "lm_head": 0 if config.tie_embeddings else V * d,
    }
    parts["total"] = sum(parts.values())
    return parts
# endregion


# region flops
def forward_flops(config: GPTConfig, seq_len: int, batch: int = 1) -> dict[str, int]:
    """一次前向（批中每条序列长 seq_len，带因果掩码但按完整 T×T 计算）的矩阵乘法 FLOPs。"""
    d, L, V, T = config.d_model, config.n_layers, config.vocab_size, seq_len
    d_kv = config.n_kv_heads * config.head_dim
    per_token = {
        "attn_proj": 2 * (d * d + 2 * d * d_kv + d * d),  # Q/K/V/O 四个投影
        "attn_scores": 2 * T * d,  # Q Kᵀ：每个查询与 T 个键做 d 维点积（所有头合计）
        "attn_values": 2 * T * d,  # 权重 @ V
        "ffn": 2 * 3 * d * config.d_ff,
    }
    parts = {k: batch * T * L * v for k, v in per_token.items()}
    parts["lm_head"] = batch * T * 2 * d * V
    parts["total"] = sum(parts.values())
    return parts
# endregion
