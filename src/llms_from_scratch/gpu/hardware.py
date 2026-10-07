"""GPU 硬件参数表、脊点（ridge point）与占用率（occupancy）计算。

所有数字取自 NVIDIA 官方白皮书 / 数据手册（见 ``GPUSpec.source``）；
张量核心峰值一律取 **稠密**（dense）数值，数据手册中带 * 的“稀疏”数值是它的两倍。
未公开或口径不一的字段记为 ``None``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GPUSpec:
    name: str
    sms: int | None
    hbm_gb: float
    hbm_tb_s: float  # HBM 带宽，TB/s
    l2_mb: float | None
    smem_per_sm_kb: int  # 单个 SM 可配置为共享内存的上限
    regs_per_sm: int = 65536  # 32 位寄存器个数（= 256 KB）
    max_warps_per_sm: int = 64
    max_blocks_per_sm: int = 32
    # 稠密峰值，TFLOPS
    tflops: dict[str, float] = field(default_factory=dict)
    source: str = ""

    def ridge_point(self, dtype: str = "bf16") -> float:
        """脊点 = 峰值算力 / 内存带宽，单位 FLOP/byte。"""
        return self.tflops[dtype] * 1e12 / (self.hbm_tb_s * 1e12)


V100 = GPUSpec(
    name="V100 SXM2",
    sms=80,
    hbm_gb=32,
    hbm_tb_s=0.9,
    l2_mb=6,
    smem_per_sm_kb=96,
    max_blocks_per_sm=32,
    tflops={"fp32": 15.7, "fp16": 125.0},
    source="NVIDIA Tesla V100 GPU Architecture whitepaper (2017)",
)

A100 = GPUSpec(
    name="A100 SXM4 80GB",
    sms=108,
    hbm_gb=80,
    hbm_tb_s=2.039,
    l2_mb=40,
    smem_per_sm_kb=164,
    tflops={"fp32": 19.5, "tf32": 156.0, "bf16": 312.0, "fp16": 312.0, "int8": 624.0},
    source="NVIDIA A100 Tensor Core GPU Architecture whitepaper; A100 datasheet",
)

H100 = GPUSpec(
    name="H100 SXM5 80GB",
    sms=132,
    hbm_gb=80,
    hbm_tb_s=3.35,
    l2_mb=50,
    smem_per_sm_kb=228,
    tflops={"fp32": 67.0, "tf32": 494.7, "bf16": 989.4, "fp16": 989.4, "fp8": 1978.9},
    source="NVIDIA H100 Tensor Core GPU Architecture whitepaper; H100 datasheet",
)

B200 = GPUSpec(
    name="B200 (HGX)",
    sms=None,  # 官方技术简报未给出单卡 SM 数，此处不填
    hbm_gb=180,
    hbm_tb_s=8.0,
    l2_mb=None,
    smem_per_sm_kb=228,
    tflops={"tf32": 1100.0, "bf16": 2250.0, "fp16": 2250.0, "fp8": 4500.0, "fp4": 9000.0},
    source="NVIDIA HGX B200 datasheet / Blackwell architecture technical brief",
)

SPECS = {"V100": V100, "A100": A100, "H100": H100, "B200": B200}


# region occupancy
@dataclass(frozen=True)
class Occupancy:
    blocks_per_sm: int
    active_warps: int
    occupancy: float  # 活跃 warp / 每 SM 最大 warp
    limiter: str  # 哪一项资源先耗尽


def occupancy(
    threads_per_block: int,
    regs_per_thread: int,
    smem_per_block: int,
    spec: GPUSpec = H100,
    reg_alloc_unit: int = 256,
    smem_reserved_per_block: int = 1024,
) -> Occupancy:
    """按线程数、寄存器、共享内存三种资源分别求每个 SM 能驻留的 block 数，取最小值。

    寄存器按 warp 分配，粒度为 256 个；sm_80 之后每个 block 额外保留 1 KB 共享内存。
    """
    warps_per_block = math.ceil(threads_per_block / 32)
    regs_per_warp = math.ceil(regs_per_thread * 32 / reg_alloc_unit) * reg_alloc_unit
    limits = {
        "warps": spec.max_warps_per_sm // warps_per_block,
        "blocks": spec.max_blocks_per_sm,
        "registers": spec.regs_per_sm // (regs_per_warp * warps_per_block),
    }
    if smem_per_block > 0:
        smem_sm = spec.smem_per_sm_kb * 1024
        limits["shared memory"] = smem_sm // (smem_per_block + smem_reserved_per_block)
    limiter = min(limits, key=limits.get)
    blocks = limits[limiter]
    warps = blocks * warps_per_block
    return Occupancy(blocks, warps, warps / spec.max_warps_per_sm, limiter)


# endregion occupancy


def littles_law_bytes_in_flight(bandwidth_tb_s: float, latency_ns: float) -> float:
    """利特尔定律：要跑满带宽，在途字节数 = 带宽 × 延迟。"""
    return bandwidth_tb_s * 1e12 * latency_ns * 1e-9
