"""QLoRA 的 4-bit NormalFloat（NF4）分块量化与双重量化。

NF4 的 16 个码值取标准正态分布的分位数并缩放到 [-1, 1]：预训练权重近似服从零均值正态，
按分位数放置码值能让每个码值“负责”大致相同数量的权重（信息论意义上的最优量化）。
为了让 0 能被精确表示，正半轴取 8 个分位数、负半轴取 7 个，再加上 0，共 16 个。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# QLoRA 论文附录与 bitsandbytes 中公布的 NF4 码表（用来核对 nf4_code 的推导）
NF4_REFERENCE = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0,
)


# region nf4_code
def nf4_code(offset: float = 0.9677083) -> torch.Tensor:
    """推导 NF4 码表。

    offset 是最外侧分位数的概率位置：取 (1 - 1/(2·15) + 1 - 1/(2·16)) / 2 ≈ 0.9677，
    使最外侧码值落在两端尾部的“中点”，避免 Φ^{-1}(1) = ∞。
    正半轴：在 [0.5, offset] 上等距取 9 个概率点，去掉 0.5 本身 → 8 个正分位数；
    负半轴：在 [0.5, offset] 上等距取 8 个概率点，去掉 0.5 → 7 个，取负号。
    最后除以最大值，使码表落在 [-1, 1]。
    """
    ndtri = torch.special.ndtri  # 标准正态分布的分位数函数 Φ^{-1}
    pos = ndtri(torch.linspace(offset, 0.5, 9, dtype=torch.float64)[:-1])
    neg = -ndtri(torch.linspace(offset, 0.5, 8, dtype=torch.float64)[:-1])
    values = torch.cat([pos, torch.zeros(1, dtype=torch.float64), neg]).sort().values
    return (values / values.max()).float()
# endregion nf4_code


# region quantize
@dataclass
class NF4Tensor:
    codes: torch.Tensor  # uint8，每个元素一个 0..15 的码（实际存储时两个码打包进一个字节）
    absmax: torch.Tensor  # 每个块一个缩放因子 [n_blocks]
    shape: torch.Size
    block_size: int


def quantize_nf4(w: torch.Tensor, block_size: int = 64) -> NF4Tensor:
    """分块 absmax 量化：每块除以自身绝对值最大值落到 [-1, 1]，再取最近的 NF4 码值。"""
    flat = w.detach().float().reshape(-1)
    if flat.numel() % block_size:
        raise ValueError("元素个数必须是 block_size 的整数倍")
    blocks = flat.view(-1, block_size)
    absmax = blocks.abs().amax(dim=1).clamp_min(1e-12)
    normed = blocks / absmax[:, None]
    code = nf4_code()
    codes = (normed[..., None] - code).abs().argmin(-1).to(torch.uint8)
    return NF4Tensor(codes.view(-1), absmax, w.shape, block_size)


def dequantize_nf4(q: NF4Tensor, absmax: torch.Tensor | None = None) -> torch.Tensor:
    """码 → 码值 × 块缩放因子。absmax 可以换成双重量化后重建的版本。"""
    scale = q.absmax if absmax is None else absmax
    values = nf4_code()[q.codes.long()].view(-1, q.block_size) * scale[:, None]
    return values.view(q.shape)
# endregion quantize


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """两个 4-bit 码装进一个 uint8：高 4 位放偶数下标，低 4 位放奇数下标。"""
    c = codes.view(-1, 2)
    return (c[:, 0] << 4) | c[:, 1]


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack([packed >> 4, packed & 0xF], dim=1).view(-1)


def quantize_absmax_int4(w: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """对照组：同样分块 absmax，但码值在 [-1, 1] 上均匀分布（-7..7 共 15 级）。返回反量化结果。"""
    blocks = w.detach().float().reshape(-1, block_size)
    absmax = blocks.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    return (torch.round(blocks / absmax * 7) / 7 * absmax).view(w.shape)


# region double_quant
@dataclass
class DoubleQuantAbsmax:
    codes: torch.Tensor  # int8，每个一级缩放因子一个
    scales: torch.Tensor  # 每 256 个一级缩放因子一个 FP32 二级缩放因子
    mean: float  # absmax 全为正，先减去均值使其对称，再做对称量化
    block_size: int


def double_quantize(absmax: torch.Tensor, block_size: int = 256) -> DoubleQuantAbsmax:
    """把一级缩放因子再做一次 8-bit 分块量化。

    QLoRA 用 8-bit 浮点（FP8）量化二级常数；这里用对称 int8，结构相同、更容易读懂。
    """
    centered = absmax - absmax.mean()
    pad = (-centered.numel()) % block_size
    blocks = torch.cat([centered, centered.new_zeros(pad)]).view(-1, block_size)
    scales = blocks.abs().amax(dim=1).clamp_min(1e-12) / 127
    codes = torch.round(blocks / scales[:, None]).to(torch.int8)
    return DoubleQuantAbsmax(codes.view(-1)[:absmax.numel()], scales, absmax.mean().item(),
                             block_size)


def double_dequantize(dq: DoubleQuantAbsmax) -> torch.Tensor:
    n = dq.codes.numel()
    pad = (-n) % dq.block_size
    codes = torch.cat([dq.codes.float(), torch.zeros(pad)]).view(-1, dq.block_size)
    return (codes * dq.scales[:, None]).view(-1)[:n] + dq.mean
# endregion double_quant


def bits_per_param(block_size: int = 64, double_quant: bool = False,
                   block_size2: int = 256) -> float:
    """每个权重平均占用的比特数：4 bit 码 + 摊销的缩放因子。"""
    if not double_quant:
        return 4 + 32 / block_size
    return 4 + 8 / block_size + 32 / (block_size * block_size2)
