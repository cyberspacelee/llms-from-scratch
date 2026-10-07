"""推理的成本模型：prefill 与 decode 每一步的 FLOPs、显存读写量与 roofline 时间。

只计入主导项：线性层（含 LM head）与注意力的乘加、权重与 KV Cache 的读写；
忽略归一化、激活函数、RoPE 等逐元素运算和激活值的读写（它们在大模型中占比很小）。
所有数字都是理论上界，真实系统还要乘以利用率（MFU / MBU）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Decoder-only Transformer 的形状（LLaMA 式：GQA + SwiGLU）。"""

    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab_size: int
    tie_embeddings: bool = False

    @property
    def attn_params_per_layer(self) -> int:
        q_o = 2 * self.d_model * self.n_heads * self.head_dim
        k_v = 2 * self.d_model * self.n_kv_heads * self.head_dim
        return q_o + k_v

    @property
    def ffn_params_per_layer(self) -> int:
        return 3 * self.d_model * self.d_ff  # W1、W3（门控）与 W2

    @property
    def body_params(self) -> int:
        """所有 Transformer 层中参与矩阵乘法的参数（不含嵌入与 LM head）。"""
        return self.n_layers * (self.attn_params_per_layer + self.ffn_params_per_layer)

    @property
    def lm_head_params(self) -> int:
        return self.vocab_size * self.d_model

    def num_params(self) -> int:
        norms = (2 * self.n_layers + 1) * self.d_model
        embeddings = self.vocab_size * self.d_model * (1 if self.tie_embeddings else 2)
        return self.body_params + norms + embeddings


@dataclass(frozen=True)
class HardwareSpec:
    """单卡的峰值算力（FLOP/s）、显存带宽（B/s）与容量（B）。"""

    name: str
    peak_flops: float
    mem_bandwidth: float
    mem_bytes: float

    @property
    def ridge_point(self) -> float:
        """roofline 拐点：算术强度高于它才可能受算力限制。"""
        return self.peak_flops / self.mem_bandwidth


# 公开配置：Meta Llama 3 的 config.json（8B 与 70B）。
LLAMA3_8B = ModelSpec("Llama-3-8B", n_layers=32, d_model=4096, n_heads=32, n_kv_heads=8,
                      head_dim=128, d_ff=14336, vocab_size=128256)
LLAMA3_70B = ModelSpec("Llama-3-70B", n_layers=80, d_model=8192, n_heads=64, n_kv_heads=8,
                       head_dim=128, d_ff=28672, vocab_size=128256)
# NVIDIA H100 SXM5 数据手册：BF16 稠密 989 TFLOP/s（稀疏 1979），HBM3 3.35 TB/s，80 GB。
H100_SXM = HardwareSpec("H100 SXM", peak_flops=989e12, mem_bandwidth=3.35e12, mem_bytes=80e9)
# NVIDIA A100 SXM4 80GB：BF16 稠密 312 TFLOP/s，HBM2e 2.039 TB/s。
A100_SXM = HardwareSpec("A100 SXM 80GB", peak_flops=312e12, mem_bandwidth=2.039e12,
                        mem_bytes=80e9)


# region kv_bytes
def kv_bytes_per_token(model: ModelSpec, dtype_bytes: float = 2) -> float:
    """每个 token 在所有层中缓存的 K 与 V：2 · L · n_kv_heads · head_dim · 字节数。"""
    return 2 * model.n_layers * model.n_kv_heads * model.head_dim * dtype_bytes
# endregion kv_bytes


@dataclass(frozen=True)
class StepCost:
    flops: float
    bytes: float

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes

    def time(self, hw: HardwareSpec, mfu: float = 1.0, mbu: float = 1.0) -> float:
        """roofline 时间：算力时间与访存时间取大者（假设两者完全重叠）。"""
        return max(self.flops / (hw.peak_flops * mfu), self.bytes / (hw.mem_bandwidth * mbu))


# region step_cost
def step_cost(model: ModelSpec, batch: list[tuple[int, int]], weight_bytes: float = 2,
              kv_dtype_bytes: float = 2) -> StepCost:
    """一次前向的成本。batch 中每项是 (本步新 token 数 q, 已缓存的上下文长度 c)。

    prefill 是 (提示长度, 0)，decode 是 (1, 当前长度)，chunked prefill 是 (块长, 已算长度)。
    """
    n_tokens = sum(q for q, _ in batch)
    n_seqs = len(batch)
    # 线性层：每个 token 对每个参数做一次乘加（2 FLOPs）；LM head 只对每个序列的最后一个 token 算。
    linear = 2 * model.body_params * n_tokens + 2 * model.lm_head_params * n_seqs
    # 注意力：位置 p 的查询看 p+1 个键，QKᵀ 与 PV 各 2·head_dim FLOPs/键/头。
    keys_seen = sum(q * c + q * (q + 1) // 2 for q, c in batch)
    attention = 4 * model.n_layers * model.n_heads * model.head_dim * keys_seen
    # 访存：权重每步读一遍；每个序列的 KV 读一遍（c+q 个位置），新 token 的 KV 写一遍。
    weights = (model.body_params + model.lm_head_params) * weight_bytes
    kv = kv_bytes_per_token(model, kv_dtype_bytes)
    kv_traffic = sum((c + q) * kv + q * kv for q, c in batch)
    return StepCost(flops=linear + attention, bytes=weights + kv_traffic)
# endregion step_cost


def decode_intensity(model: ModelSpec, batch_size: int, context: int,
                     weight_bytes: float = 2, kv_dtype_bytes: float = 2) -> float:
    """batch_size 个序列各 decode 一个 token 时的算术强度（FLOP/B）。"""
    return step_cost(model, [(1, context)] * batch_size, weight_bytes, kv_dtype_bytes).intensity


def prefill_time(model: ModelSpec, hw: HardwareSpec, prompt_len: int, mfu: float = 1.0) -> float:
    return step_cost(model, [(prompt_len, 0)]).time(hw, mfu=mfu)


def decode_step_time(model: ModelSpec, hw: HardwareSpec, batch_size: int, context: int,
                     weight_bytes: float = 2, kv_dtype_bytes: float = 2,
                     mbu: float = 1.0) -> float:
    cost = step_cost(model, [(1, context)] * batch_size, weight_bytes, kv_dtype_bytes)
    return cost.time(hw, mbu=mbu)


# region max_batch
def max_batch_size(model: ModelSpec, hw: HardwareSpec, context: int, weight_bytes: float = 2,
                   kv_dtype_bytes: float = 2, utilization: float = 0.9,
                   reserved_bytes: float = 0.0, n_gpus: int = 1) -> int:
    """显存放得下多少个长度为 context 的序列（权重与 KV 均匀切到 n_gpus 张卡上）。"""
    budget = n_gpus * hw.mem_bytes * utilization - reserved_bytes
    free = budget - model.num_params() * weight_bytes
    per_seq = context * kv_bytes_per_token(model, kv_dtype_bytes)
    return max(0, math.floor(free / per_seq))
# endregion max_batch


@dataclass(frozen=True)
class LatencyReport:
    ttft: float  # 首 token 延迟（只算 prefill，不含排队）
    tpot: float  # 每个输出 token 的平均时间
    e2e: float  # 端到端延迟
    throughput: float  # 整个批的输出 token/s


def batch_latency(model: ModelSpec, hw: HardwareSpec, batch_size: int, prompt_len: int,
                  output_len: int, mfu: float = 1.0, mbu: float = 1.0) -> LatencyReport:
    """一批同时到达、同样长度的请求：一次 prefill 后逐步 decode（解析近似）。"""
    ttft = step_cost(model, [(prompt_len, 0)] * batch_size).time(hw, mfu=mfu)
    steps = [decode_step_time(model, hw, batch_size, prompt_len + t, mbu=mbu)
             for t in range(1, output_len)]
    decode = sum(steps)
    tpot = decode / max(1, output_len - 1)
    e2e = ttft + decode
    return LatencyReport(ttft, tpot, e2e, batch_size * output_len / e2e)
