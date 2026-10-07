"""组合并行的显存与吞吐规划：把前面各章的公式放到一起。

所有数字都是“纸面估算”：激活公式来自 Korthikanti 等（2022）对 GPT 式层的推导，
模型状态公式来自 ZeRO（2020），FLOPs 用 6N + 6·L·s·d 的近似。它们用来在开机之前排除
明显不可行的配置、比较候选方案的数量级；最终仍要在目标硬件上实测峰值显存与吞吐。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from llms_from_scratch.distributed.zero import zero_memory_per_gpu

GB = 1e9

# region activations


def activation_bytes_per_layer(s: int, b: int, h: int, a: int, t: int = 1,
                               sequence_parallel: bool = False,
                               recompute: str = "none") -> float:
    """一个 Transformer 层为反向保存的激活字节数（Korthikanti 等 2022，表 2）。

    s 序列长度、b 微批大小、h 隐藏维、a 头数、t 张量并行度；激活为 16 位，dropout 掩码 1 字节。
    recompute: "none" 不重计算；"selective" 只重算注意力的 softmax/dropout 部分（去掉 5as/h 项）；
    "full" 每层只保存输入 2sbh（序列并行时再除以 t）。
    """
    sbh = s * b * h
    attention_core = 5 * a * s / (h * t)  # QKᵀ 分数、softmax、dropout：随 s² 增长的部分
    if recompute == "full":
        return 2 * sbh / (t if sequence_parallel else 1)
    if sequence_parallel:
        rest = 34 / t
    else:
        rest = 10 + 24 / t  # LayerNorm、dropout 与残差输入共 10sbh 无法被张量并行切开
    return sbh * (rest + (0 if recompute == "selective" else attention_core))


def pipeline_activation_factor(p: int, m: int, interleave: int = 1) -> float:
    """1F1B 下第一个 stage 要同时保存 p 个微批，每个微批 L/p 层 → 恰好相当于 L 层的激活。

    交错调度（每个 rank 有 interleave 个模型块）需要再乘 1 + (p-1)/(p·interleave)。
    返回“相当于多少个完整层的激活 / L”，m < p 时微批数不足以填满流水线。
    """
    in_flight = min(m, p) / p
    return in_flight * (1 + (p - 1) / (p * interleave) if interleave > 1 else 1.0)

# endregion


# region models


@dataclass
class DenseConfig:
    """LLaMA 式稠密模型：GQA 注意力 + SwiGLU 前馈，词嵌入与输出层不共享。"""

    layers: int
    d_model: int
    heads: int
    kv_heads: int
    d_ff: int
    vocab: int

    def params_per_layer(self) -> int:
        head_dim = self.d_model // self.heads
        attn = 2 * self.d_model * self.d_model + 2 * self.d_model * self.kv_heads * head_dim
        return attn + 3 * self.d_model * self.d_ff + 2 * self.d_model

    def params(self) -> int:
        return self.layers * self.params_per_layer() + 2 * self.vocab * self.d_model + self.d_model


LLAMA3_70B = DenseConfig(layers=80, d_model=8192, heads=64, kv_heads=8, d_ff=28672, vocab=128256)


@dataclass
class DeepSeekV3Config:
    """DeepSeek-V3 的主要超参数（技术报告表与 config.json），MLA 注意力 + DeepSeekMoE。"""

    layers: int = 61
    dense_layers: int = 3
    d_model: int = 7168
    heads: int = 128
    q_lora: int = 1536
    kv_lora: int = 512
    qk_nope: int = 128
    qk_rope: int = 64
    v_head: int = 128
    dense_ff: int = 18432
    expert_ff: int = 2048
    routed: int = 256
    shared: int = 1
    top_k: int = 8
    vocab: int = 129280

    def attention_params(self) -> int:
        h, d = self.heads, self.d_model
        q = d * self.q_lora + self.q_lora * h * (self.qk_nope + self.qk_rope)
        kv = d * (self.kv_lora + self.qk_rope) + self.kv_lora * h * (self.qk_nope + self.v_head)
        o = h * self.v_head * d
        norms = self.q_lora + self.kv_lora + 2 * d
        return q + kv + o + norms

    def expert_params(self) -> int:
        return 3 * self.d_model * self.expert_ff

    def params(self) -> dict[str, int]:
        moe_layers = self.layers - self.dense_layers
        return {
            "attention": self.layers * self.attention_params(),
            "dense_ffn": self.dense_layers * 3 * self.d_model * self.dense_ff,
            "routed_experts": moe_layers * self.routed * self.expert_params(),
            "shared_experts": moe_layers * self.shared * self.expert_params(),
            "routers": moe_layers * self.routed * self.d_model,
            "embeddings": 2 * self.vocab * self.d_model + self.d_model,
        }

    def total_params(self) -> int:
        return sum(self.params().values())

    def active_params(self) -> int:
        p = self.params()
        moe_layers = self.layers - self.dense_layers
        active_routed = moe_layers * self.top_k * self.expert_params()
        return p["attention"] + p["dense_ffn"] + active_routed + p["shared_experts"] \
            + p["routers"] + p["embeddings"]

# endregion


# region dense_plan


@dataclass
class DensePlan:
    """稠密模型的一个 TP × PP × DP 配置的每卡显存估算（单位：字节）。"""

    model: DenseConfig
    gpus: int
    tp: int
    pp: int
    seq: int
    micro_batch: int
    global_batch_tokens: int
    zero_stage: int = 1
    sequence_parallel: bool = True
    recompute: str = "selective"
    interleave: int = 1

    @property
    def dp(self) -> int:
        return self.gpus // (self.tp * self.pp)

    @property
    def microbatches(self) -> int:
        """每个数据并行副本每步处理的微批数 m。"""
        return self.global_batch_tokens // (self.seq * self.micro_batch * self.dp)

    def state_bytes(self) -> float:
        """权重、梯度、优化器状态：先按 TP×PP 切模型，再在 DP 维度上按 ZeRO 阶段切。"""
        local = self.model.params() / (self.tp * self.pp)
        return zero_memory_per_gpu(local, self.dp, self.zero_stage)

    def activation_bytes(self) -> float:
        m = self.model
        per_layer = activation_bytes_per_layer(self.seq, self.micro_batch, m.d_model, m.heads,
                                               self.tp, self.sequence_parallel, self.recompute)
        factor = pipeline_activation_factor(self.pp, self.microbatches, self.interleave)
        return per_layer * m.layers * factor

    def bubble(self) -> float:
        """1F1B 的气泡占比 (p-1)/(v·m + p - 1)；v 是交错的模型块数。"""
        p, m, v = self.pp, self.microbatches, self.interleave
        return (p - 1) / (v * m + p - 1)

    def summary(self) -> dict[str, float]:
        return {"dp": self.dp, "m": self.microbatches, "state_gb": self.state_bytes() / GB,
                "activation_gb": self.activation_bytes() / GB,
                "total_gb": (self.state_bytes() + self.activation_bytes()) / GB,
                "bubble": self.bubble()}

# endregion


# region throughput


def training_flops_per_token(params: float, layers: int, seq: int, d_attn: int) -> float:
    """每个训练 token 的 FLOPs：6N（矩阵乘）+ 6·L·s·d（注意力分数与加权求和，按非因果计）。"""
    return 6 * params + 6 * layers * seq * d_attn


def training_days(total_tokens: float, flops_per_token: float, gpus: int,
                  peak_flops: float, mfu: float) -> float:
    return total_tokens * flops_per_token / (gpus * peak_flops * mfu) / 86400


def achieved_flops_per_gpu(total_tokens: float, flops_per_token: float,
                           gpu_hours: float) -> float:
    """由公开的 GPU 小时数反推每卡平均有效算力（FLOP/s）。"""
    return total_tokens * flops_per_token / (gpu_hours * 3600)

# endregion


# region moe_plan


def moe_state_bytes_per_gpu(cfg: DeepSeekV3Config, pp: int, ep: int, dp: int,
                            weight_bytes: float = 2, grad_bytes: float = 4,
                            optim_bytes: float = 8) -> dict[str, float]:
    """PP × EP × DP（ZeRO-1）下 MoE 模型每卡的模型状态。

    路由专家被 EP 切开，每个专家在 dp/ep 个 rank 上有副本；其余参数（注意力、共享专家、
    稠密层、嵌入）只按 PP 切，在全部 dp 个 rank 上复制。ZeRO-1 把优化器状态在各自的
    副本组内切分。默认值按 DeepSeek-V3 报告：BF16 权重、FP32 梯度累积、FP32 主参数 + BF16 两个矩。
    """
    p = cfg.params()
    expert = p["routed_experts"] / (pp * ep)
    other = (cfg.total_params() - p["routed_experts"]) / pp
    weights = (expert + other) * weight_bytes
    grads = (expert + other) * grad_bytes
    optimizer = expert * optim_bytes / (dp // ep) + other * optim_bytes / dp
    return {"expert_params": expert, "other_params": other, "weights": weights,
            "grads": grads, "optimizer": optimizer, "total": weights + grads + optimizer}


def ib_dispatch_bytes(tokens: int, nodes_per_token: float, d_model: int,
                      bytes_per_elem: float) -> float:
    """节点受限路由下跨节点（IB）的 dispatch 字节数：同一 token 去同一节点只发一次，再由 NVLink 转发。"""
    return tokens * nodes_per_token * d_model * bytes_per_elem

# endregion


# region checkpoint


def optimal_checkpoint_interval(checkpoint_seconds: float, mtbf_seconds: float) -> float:
    """Young（1974）的一阶近似：τ* ≈ sqrt(2 · δ · M)。

    δ 是写一次检查点让训练停顿的时间，M 是整个作业的平均无故障时间。间隔太短，
    写检查点的开销 δ/τ 大；太长，故障时平均丢失 τ/2 的进度。两项之和在 τ* 处最小。
    """
    return (2 * checkpoint_seconds * mtbf_seconds) ** 0.5


def job_mtbf(gpu_mtbf_hours: float, gpus: int) -> float:
    """假设各 GPU 独立、故障率恒定，整个作业的平均无故障时间（小时）随卡数反比下降。"""
    return gpu_mtbf_hours / gpus

# endregion


# region measure


def saved_activation_bytes(fn: Callable[[], torch.Tensor]) -> int:
    """运行 fn()（一次前向），统计 autograd 为反向保存的张量总字节数（按存储去重）。"""
    seen: dict[int, int] = {}

    def pack(t: torch.Tensor):
        storage = t.untyped_storage()
        seen[storage.data_ptr()] = storage.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        out = fn()
    out.sum().backward()
    return sum(seen.values())

# endregion
