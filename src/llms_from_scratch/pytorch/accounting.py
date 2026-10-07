"""资源核算：参数量、FLOPs、显存与 MFU 的公式，以及用 PyTorch 实测的对照工具。

公式针对 ``llms_from_scratch.transformer.model.GPT``（RMSNorm + RoPE + GQA + SwiGLU，无偏置）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.flop_counter import FlopCounterMode

from llms_from_scratch.transformer.model import GPTConfig

# region peaks
# 稠密（非稀疏）峰值算力，单位 FLOP/s。来源：NVIDIA H100 / A100 数据手册；
# 手册中带 * 的 Tensor Core 数字是 2:4 结构化稀疏下的值，稠密值为其一半。
PEAK_FLOPS = {
    "H100-SXM-bf16": 989.5e12,
    "H100-SXM-fp32": 67e12,
    "A100-bf16": 312e12,
    "A100-fp32": 19.5e12,
}
# endregion peaks


# region matmul
def matmul_flops(m: int, k: int, n: int) -> int:
    """[m, k] @ [k, n]：输出 m·n 个元素，每个元素 k 次乘法 + k 次加法，共 2mkn。"""
    return 2 * m * k * n


def linear_flops(tokens: int, d_in: int, d_out: int) -> dict[str, int]:
    """线性层 Y = X Wᵀ（X: [tokens, d_in]）前向与反向的 FLOPs。

    反向要算两个矩阵乘：dX = dY W（对输入）和 dW = dYᵀ X（对权重），各与前向等大。
    """
    fwd = matmul_flops(tokens, d_in, d_out)
    return {"forward": fwd, "backward_input": fwd, "backward_weight": fwd}
# endregion matmul


# region params
def gpt_param_count(cfg: GPTConfig) -> dict[str, int]:
    """按结构逐项写出 GPT 的参数量。"""
    d, dh, L = cfg.d_model, cfg.head_dim, cfg.n_layers
    attn = d * cfg.n_heads * dh * 2 + d * cfg.n_kv_heads * dh * 2  # W_Q, W_O + W_K, W_V
    ffn = 3 * d * cfg.d_ff                                          # SwiGLU 的 w1, w3, w2
    norms = 2 * d                                                   # 两个 RMSNorm 的增益
    embed = cfg.vocab_size * d
    head = 0 if cfg.tie_embeddings else cfg.vocab_size * d
    blocks = L * (attn + ffn + norms)
    return {
        "embedding": embed, "attention": L * attn, "ffn": L * ffn, "norms": L * norms + d,
        "lm_head": head, "total": embed + blocks + d + head,
    }


def matmul_params(cfg: GPTConfig) -> int:
    """参与矩阵乘的权重个数（每个 token 都要乘一遍）：各层投影 + 输出头。嵌入查表不算。"""
    p = gpt_param_count(cfg)
    return p["attention"] + p["ffn"] + cfg.vocab_size * cfg.d_model
# endregion params


# region flops
def gpt_forward_flops(cfg: GPTConfig, batch: int, seq_len: int, causal: bool = False) -> dict[str, int]:
    """一次前向的 FLOPs，分成“权重矩阵乘”与“注意力”两部分。

    - 权重矩阵乘：每个 token 对每个权重做一次乘加，2 · tokens · N_matmul；
    - 注意力：每层 QKᵀ 与 PV 各是 B·h 个 [T, d_h] @ [d_h, T]，共 2 · 2·B·T²·d。
      因果掩码让一半分数无用，causal=True 时按 1/2 计（FlashAttention 等实现可以跳过它们）。
    """
    tokens = batch * seq_len
    linear = 2 * tokens * matmul_params(cfg)
    attention = cfg.n_layers * 2 * matmul_flops(batch * cfg.n_heads * seq_len, cfg.head_dim, seq_len)
    if causal:
        attention //= 2
    return {"linear": linear, "attention": attention, "total": linear + attention}


def training_flops(n_params: int, tokens: int) -> int:
    """6N 法则：前向 2N、反向 4N 每 token。忽略注意力项与逐元素运算。"""
    return 6 * n_params * tokens


def measured_flops(fn, *args) -> int:
    """用 torch.utils.flop_counter 统计 fn(*args) 中矩阵乘类算子的 FLOPs。"""
    with FlopCounterMode(display=False) as counter:
        fn(*args)
    return counter.get_total_flops()
# endregion flops


# region memory
BYTES = {torch.float32: 4, torch.bfloat16: 2, torch.float16: 2, torch.float8_e4m3fn: 1}


@dataclass
class MemoryEstimate:
    params: int
    grads: int
    optimizer: int
    master: int

    @property
    def total(self) -> int:
        return self.params + self.grads + self.optimizer + self.master


def estimate_memory(n_params: int, mixed_precision: bool = False, optimizer: str = "adamw") -> MemoryEstimate:
    """不含激活的训练显存（字节）。

    - FP32：参数 4 + 梯度 4 + Adam 两个矩 8 = 16 字节/参数；
    - 混合精度（ZeRO 论文的记法）：BF16 参数 2 + BF16 梯度 2 + FP32 主权重 4 + FP32 两个矩 8
      = 16 字节/参数——低精度并没有让“静态”显存变少，省下的是激活和带宽。
    """
    state = {"adamw": 8, "sgd-momentum": 4, "sgd": 0}[optimizer]
    if mixed_precision:
        return MemoryEstimate(params=2 * n_params, grads=2 * n_params,
                              optimizer=state * n_params, master=4 * n_params)
    return MemoryEstimate(params=4 * n_params, grads=4 * n_params, optimizer=state * n_params, master=0)


def gpt_activation_elements(cfg: GPTConfig, batch: int, seq_len: int) -> int:
    """粗略估计每层为反向保存的激活元素数（按本书 GPT 的写法逐项数，不含注意力矩阵）。

    每层：两个 RMSNorm 的输入 2·BTd 与输出 2·BTd；Q、K、V（RoPE 前后各一份）；
    注意力输出 BTd；SwiGLU 的 w1/w3 输出与 SiLU 结果 3·BT·d_ff。
    这是数量级估计——确切数字取决于每个算子选择保存什么，用 saved_tensor_bytes 实测。
    """
    d, kv = cfg.d_model, cfg.n_kv_heads * cfg.head_dim
    per_token = 4 * d + 2 * (d + 2 * kv) + d + 3 * cfg.d_ff
    return cfg.n_layers * batch * seq_len * per_token
# endregion memory


# region mfu
def measure_matmul_flops_per_s(n: int = 1024, dtype: torch.dtype = torch.float32,
                               repeats: int = 10) -> float:
    """在本机 CPU 上测 n×n 矩阵乘的实际 FLOP/s：先预热，再取多次计时的中位数。"""
    a, b = torch.randn(n, n, dtype=dtype), torch.randn(n, n, dtype=dtype)
    for _ in range(2):
        a @ b
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        a @ b
        times.append(time.perf_counter() - start)
    times.sort()
    return matmul_flops(n, n, n) / times[len(times) // 2]


def mfu(tokens_per_second: float, flops_per_token: float, peak_flops: float) -> float:
    """模型 FLOPs 利用率：模型理论上需要的 FLOP/s ÷ 硬件峰值。重计算等额外开销不计入分子。"""
    return tokens_per_second * flops_per_token / peak_flops
# endregion mfu


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
