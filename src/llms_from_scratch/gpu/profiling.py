"""性能分析工具：roofline、正确的计时、PyTorch Profiler，以及 Transformer 层的逐算子估算。"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch

from llms_from_scratch.gpu.hardware import H100, GPUSpec


# region roofline
def attainable_tflops(intensity: float, peak_tflops: float, bandwidth_tb_s: float) -> float:
    """Roofline：可达算力 = min(峰值算力, 算术强度 × 带宽)。intensity 单位 FLOP/byte。"""
    return min(peak_tflops, intensity * bandwidth_tb_s)


def ridge_point(peak_tflops: float, bandwidth_tb_s: float) -> float:
    """两条屋顶线的交点：I* = 峰值算力 / 带宽。"""
    return peak_tflops / bandwidth_tb_s


# endregion roofline


# region benchmark
@dataclass(frozen=True)
class Timing:
    median_ms: float
    min_ms: float
    max_ms: float
    samples: list[float]


def benchmark(fn, warmup: int = 5, repeats: int = 20, device: str = "cpu") -> Timing:
    """预热 → 多次计时 → 取中位数。

    GPU 上 kernel 是异步启动的：主机端计时只量到“启动”所花的时间，
    所以用 CUDA Event 记录设备时间线上的起止点，并在读取前同步。
    """
    for _ in range(warmup):  # 预热：触发编译、cuBLAS 选算法、分配器缓存、时钟升频
        fn()
    samples = []
    if device == "cuda":
        torch.cuda.synchronize()
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()  # 等到 end 事件在 GPU 上真正发生
            samples.append(start.elapsed_time(end))
    else:
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - t0) * 1e3)
    return Timing(statistics.median(samples), min(samples), max(samples), samples)


# endregion benchmark


# region profile_block
def profile_block(batch: int = 2, seq: int = 64, d_model: int = 128, n_heads: int = 4):
    """用 torch.profiler 记录共享 GPT 的一个 Block 的前向 + 反向（CPU 上即可运行）。

    返回 profiler 对象；``prof.key_averages().table(...)`` 给出按算子汇总的表格，
    ``prof.export_chrome_trace(path)`` 导出可在 Perfetto 中查看的时间线。
    """
    from torch.profiler import ProfilerActivity, profile, record_function

    from llms_from_scratch.transformer.model import Block, GPTConfig, rope_frequencies

    cfg = GPTConfig(d_model=d_model, n_heads=n_heads, context_length=seq)
    block = Block(cfg, layer=0)
    freqs = rope_frequencies(cfg.head_dim, seq)
    x = torch.randn(batch, seq, d_model, requires_grad=True)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    block(x, freqs).sum().backward()  # 预热一次
    with profile(activities=activities, record_shapes=True) as prof:
        with record_function("block_forward"):
            y = block(x, freqs)
        with record_function("block_backward"):
            y.sum().backward()
    return prof


# endregion profile_block


# region layer_estimate
@dataclass(frozen=True)
class OpEstimate:
    name: str
    flops: float
    bytes: float

    @property
    def intensity(self) -> float:
        return self.flops / self.bytes

    def time_us(self, spec: GPUSpec, dtype: str = "bf16") -> float:
        """不考虑重叠与效率损失的下界：max(计算时间, 访存时间)。"""
        compute = self.flops / (spec.tflops[dtype] * 1e12)
        memory = self.bytes / (spec.hbm_tb_s * 1e12)
        return max(compute, memory) * 1e6

    def bound(self, spec: GPUSpec, dtype: str = "bf16") -> str:
        return "compute" if self.intensity > spec.ridge_point(dtype) else "memory"


def transformer_layer_ops(
    batch: int, seq: int, d_model: int, n_heads: int, d_ff: int, elem: int = 2,
    flash: bool = True,
) -> list[OpEstimate]:
    """一个 Pre-Norm + SwiGLU 层前向的逐算子 FLOPs 与 HBM 字节（BF16，MHA）。

    矩阵乘 [M,K]@[K,N]：FLOPs = 2MKN，字节 = (MK + KN + MN)·elem。
    逐元素与归一化：只计读写字节，FLOPs 记为每元素若干次，可忽略。
    """
    m = batch * seq  # token 数
    d, f, h = d_model, d_ff, n_heads

    def matmul(name, mm, k, n, count=1):
        return OpEstimate(name, 2 * mm * k * n * count, (mm * k + k * n + mm * n) * elem * count)

    def elementwise(name, n_elems, reads, writes, flops_per=5):
        return OpEstimate(name, flops_per * n_elems, (reads + writes) * n_elems * elem)

    ops = [
        elementwise("rmsnorm (attn)", m * d, 1, 1),
        matmul("qkv proj", m, d, 3 * d),
        elementwise("rope (q,k)", 2 * m * d, 1, 1),
    ]
    hd = d // h
    if flash:  # 只读 Q、K、V 与写 O；因果掩码下 FLOPs 约减半
        ops.append(OpEstimate("flash attention", 2 * 2 * batch * h * seq * seq * hd / 2,
                              4 * m * d * elem))
    else:
        ops += [
            matmul("QKᵀ", seq, hd, seq, count=batch * h),
            elementwise("softmax", batch * h * seq * seq, 1, 1),
            matmul("PV", seq, seq, hd, count=batch * h),
        ]
    ops += [
        matmul("out proj", m, d, d),
        elementwise("residual add", m * d, 2, 1, flops_per=1),
        elementwise("rmsnorm (ffn)", m * d, 1, 1),
        matmul("w1, w3 (up)", m, d, 2 * f),
        elementwise("silu * mul", m * f, 2, 1),
        matmul("w2 (down)", m, f, d),
        elementwise("residual add", m * d, 2, 1, flops_per=1),
    ]
    return ops


# endregion layer_estimate


def h100_layer_report(batch: int = 8, seq: int = 2048, d_model: int = 4096, n_heads: int = 32):
    """对 H100 BF16 打印每个算子的强度、受限类型与时间下界（教学用的“纸面 profile”）。"""
    d_ff = 64 * ((8 * d_model // 3 + 63) // 64)
    rows = []
    for op in transformer_layer_ops(batch, seq, d_model, n_heads, d_ff):
        rows.append((op.name, op.intensity, op.bound(H100), op.time_us(H100)))
    return rows
