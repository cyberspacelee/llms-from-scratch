"""混合精度：浮点格式模拟、FP16 动态损失缩放、FP8 逐张量 / 延迟 / 分块缩放与受限精度累加。

`round_to_format` 用纯 float32/64 运算模拟任意 (指数位, 尾数位) 格式的“就近舍入到偶数”，
可以与 PyTorch 自带的 float16、bfloat16、float8_e4m3fn、float8_e5m2 逐元素对照。
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import torch


# region formats
@dataclass(frozen=True)
class FloatFormat:
    name: str
    exp_bits: int
    man_bits: int
    max_value: float          # 最大有限值（E4M3 牺牲了 inf，最大值是 448 而非 240）

    @property
    def bias(self) -> int:
        return 2 ** (self.exp_bits - 1) - 1

    @property
    def min_normal(self) -> float:
        return 2.0 ** (1 - self.bias)

    @property
    def min_subnormal(self) -> float:
        return 2.0 ** (1 - self.bias - self.man_bits)

    @property
    def eps(self) -> float:
        """1 与下一个可表示数之间的距离 2^(-尾数位)。"""
        return 2.0 ** -self.man_bits


FP32 = FloatFormat("FP32", 8, 23, float(torch.finfo(torch.float32).max))
FP16 = FloatFormat("FP16", 5, 10, 65504.0)
BF16 = FloatFormat("BF16", 8, 7, float(torch.finfo(torch.bfloat16).max))
E4M3 = FloatFormat("FP8 E4M3", 4, 3, 448.0)
E5M2 = FloatFormat("FP8 E5M2", 5, 2, 57344.0)


def round_to_format(x: torch.Tensor, fmt: FloatFormat) -> torch.Tensor:
    """把 x 就近舍入（ties-to-even）到格式 fmt，超出范围的值饱和到 ±max_value。

    思路：对每个数求出它所在区间 [2^e, 2^(e+1)) 的量化步长 2^(e - man_bits)，
    除以步长后用 torch.round（本身就是 ties-to-even）取整再乘回。指数低于最小正规数时
    步长固定为最小次正规数，这就是渐进下溢。
    """
    x64 = x.double()
    mag = x64.abs()
    e = torch.floor(torch.log2(mag.clamp_min(fmt.min_normal)))   # 次正规区使用最小正规指数
    step = torch.pow(2.0, e - fmt.man_bits)
    y = torch.round(x64 / step) * step
    y = y.clamp(-fmt.max_value, fmt.max_value)
    return torch.where(mag == 0, x64, y).to(x.dtype)
# endregion formats


# region master_weights
def accumulate_updates(start: float, update: float, steps: int, fmt: FloatFormat) -> float:
    """把同一个小更新反复加到权重上，每步后把权重舍入到 fmt。

    BF16 中 1 + 0.001 会被舍入回 1（“淹没”，swamping），所以长期累积的小更新全部丢失；
    这就是混合精度训练要保留一份 FP32 主权重的原因。
    """
    w = torch.tensor(start, dtype=torch.float64)
    for _ in range(steps):
        w = round_to_format(w + update, fmt)
    return w.item()


def autocast_dtypes(device: str = "cpu", dtype: torch.dtype = torch.bfloat16) -> dict[str, torch.dtype]:
    """在 autocast 区域内运行几个代表性算子，返回各自的输出精度（算子级策略因设备而异）。"""
    x, w = torch.randn(4, 8), torch.randn(8, 8)
    target = torch.zeros(4, dtype=torch.long)
    with torch.autocast(device, dtype=dtype):
        y = x @ w
        return {
            "matmul": y.dtype,                                          # 低精度：走 Tensor Core
            "linear": torch.nn.functional.linear(x, w).dtype,
            "add_fp32_input": (y + x).dtype,                            # 混合输入：提升到较宽类型
            "cross_entropy": torch.nn.functional.cross_entropy(y, target).dtype,  # FP32
        }
# endregion master_weights


# region loss_scaler
class DynamicLossScaler:
    """FP16 训练的动态损失缩放（与 torch.amp.GradScaler 的默认策略相同）。

    反向前把损失乘以 S，使小梯度不下溢；更新前除回 S。
    出现 inf/NaN：跳过本步，S ← S·backoff；连续 growth_interval 步无溢出：S ← S·growth。
    """

    def __init__(self, init_scale: float = 2.0**16, growth_factor: float = 2.0,
                 backoff_factor: float = 0.5, growth_interval: int = 2000) -> None:
        self.scale = init_scale
        self.growth_factor, self.backoff_factor = growth_factor, backoff_factor
        self.growth_interval = growth_interval
        self._good_steps = 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale

    @torch.no_grad()
    def unscale_(self, params) -> bool:
        """把梯度除以 S（在 FP32 中进行），返回是否发现非有限值。"""
        found_inf = False
        for p in params:
            if p.grad is None:
                continue
            p.grad.div_(self.scale)
            if not torch.isfinite(p.grad).all():
                found_inf = True
        return found_inf

    def update(self, found_inf: bool) -> None:
        if found_inf:
            self.scale *= self.backoff_factor
            self._good_steps = 0
        else:
            self._good_steps += 1
            if self._good_steps % self.growth_interval == 0:
                self.scale *= self.growth_factor

    def step(self, optimizer: torch.optim.Optimizer, params) -> bool:
        """unscale → 检查 → 只有梯度有限时才更新参数 → 调整 S。返回本步是否真正更新。"""
        params = list(params)
        found_inf = self.unscale_(params)
        if not found_inf:
            optimizer.step()
        self.update(found_inf)
        return not found_inf
# endregion loss_scaler


# region fp8_quant
def quantize_fp8(x: torch.Tensor, fmt: FloatFormat = E4M3, amax: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """逐张量缩放：s = max_value / amax，把 x·s 舍入到 FP8。返回（FP8 值，缩放因子 s）。"""
    amax = x.abs().max() if amax is None else amax
    scale = fmt.max_value / amax.clamp_min(1e-12)
    return round_to_format(x * scale, fmt), scale


def quantize_blockwise(x: torch.Tensor, block: tuple[int, int], fmt: FloatFormat = E4M3
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """分块缩放：每个 block[0] × block[1] 子块有自己的 amax 与缩放因子。

    DeepSeek-V3：激活用 (1, 128)（每个 token 每 128 个通道一组），权重用 (128, 128)。
    返回 FP8 值（与 x 同形状）与缩放因子（形状 [M/bm, N/bn]）。
    """
    M, N = x.shape
    bm, bn = block
    if M % bm or N % bn:
        raise ValueError("矩阵尺寸必须是块大小的整数倍")
    blocks = x.reshape(M // bm, bm, N // bn, bn)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(1e-12)
    scale = fmt.max_value / amax
    q = round_to_format(blocks * scale, fmt)
    return q.reshape(M, N), scale.reshape(M // bm, N // bn)


def dequantize_blockwise(q: torch.Tensor, scale: torch.Tensor, block: tuple[int, int]) -> torch.Tensor:
    M, N = q.shape
    bm, bn = block
    blocks = q.reshape(M // bm, bm, N // bn, bn)
    return (blocks / scale.reshape(M // bm, 1, N // bn, 1)).reshape(M, N)


def relative_error(approx: torch.Tensor, exact: torch.Tensor) -> float:
    return ((approx - exact).norm() / exact.norm()).item()


class DelayedScaler:
    """延迟缩放（Transformer Engine 的 DelayedScaling）：用过去若干步 amax 的最大值算本步的 s。

    好处是量化前不必先扫描一遍当前张量；代价是本步出现比历史更大的离群值时会饱和。
    """

    def __init__(self, history: int = 16, fmt: FloatFormat = E4M3, margin: int = 0) -> None:
        self.amax_history: deque[float] = deque(maxlen=history)
        self.fmt, self.margin = fmt, margin

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        amax = max(self.amax_history) if self.amax_history else x.abs().max().item()
        scale = self.fmt.max_value / max(amax, 1e-12) / 2**self.margin
        q = round_to_format(x * scale, self.fmt)
        self.amax_history.append(x.abs().max().item())      # 用本步的 amax 更新历史，供下一步使用
        return q, scale
# endregion fp8_quant


# region blockwise_gemm
def fp8_gemm_blockwise(a: torch.Tensor, b: torch.Tensor, block: int = 128) -> torch.Tensor:
    """模拟 DeepSeek-V3 的 FP8 GEMM：C = A @ Bᵀ，A:[M, K] 激活，B:[N, K] 权重（nn.Linear 布局）。

    A 按 (1, block) 量化，B 按 (block, block) 量化。沿 K 每 block 个元素得到一段部分和，
    乘上这段对应的两个缩放因子后在 FP32 中累加——这正是“提升到 CUDA Core 做 FP32 累加”。
    """
    qa, sa = quantize_blockwise(a, (1, block))        # sa: [M, K/block]
    qb, sb = quantize_blockwise(b, (block, block))    # sb: [N/block, K/block]
    M, K = a.shape
    N = b.shape[0]
    out = torch.zeros(M, N, dtype=torch.float32)
    for kb in range(K // block):
        cols = slice(kb * block, (kb + 1) * block)
        partial = qa[:, cols].float() @ qb[:, cols].float().T            # Tensor Core 上的一段
        scale_b = sb[:, kb].repeat_interleave(block)[:N]                 # 每个输出列所属权重块的 s
        out += partial / (sa[:, kb:kb + 1] * scale_b[None, :])           # 反缩放后 FP32 累加
    return out


def limited_precision_sum(values: torch.Tensor, man_bits: int, promote_every: int | None = None
                          ) -> float:
    """顺序累加，每次加法后把累加器舍入到 man_bits 位尾数；可选每 promote_every 项把
    部分和转入 FP64 累加器并清零（模拟 DeepSeek-V3 每 N_C = 128 个元素提升一次）。"""
    fmt = FloatFormat(f"acc{man_bits}", 8, man_bits, 3.0e38)
    total, acc = 0.0, torch.zeros((), dtype=torch.float64)
    for i, v in enumerate(values.double()):
        acc = round_to_format(acc + v, fmt)
        if promote_every and (i + 1) % promote_every == 0:
            total += acc.item()
            acc = torch.zeros((), dtype=torch.float64)
    return total + acc.item()
# endregion blockwise_gemm


def fp8_dynamic_range(fmt: FloatFormat) -> float:
    """以 2 为底的动态范围（最大值 / 最小次正规数 的对数），单位“二进位”（binade）。"""
    return math.log2(fmt.max_value / fmt.min_subnormal)
