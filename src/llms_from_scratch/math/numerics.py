"""浮点数与数值稳定性：位布局、舍入与累加误差、稳定 softmax 与初始化。

对应《浮点数与数值稳定性》一章。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


# region format
@dataclass(frozen=True)
class FloatFormat:
    """一种二进制浮点格式：1 位符号 + ``exp_bits`` 位指数 + ``man_bits`` 位尾数。

    ``ieee=True`` 表示指数全 1 保留给 inf/NaN（FP32、FP16、BF16、FP8 E5M2）；
    FP8 E4M3（PyTorch 中的 ``float8_e4m3fn``）只把 S.1111.111 留作 NaN，没有 inf，
    以换取多一档指数。
    """

    name: str
    exp_bits: int
    man_bits: int
    ieee: bool = True

    @property
    def bias(self) -> int:
        return 2 ** (self.exp_bits - 1) - 1

    @property
    def max_normal(self) -> float:
        if self.ieee:
            top_exp = 2**self.exp_bits - 2 - self.bias  # 指数全 1 被保留
            return 2.0**top_exp * (2 - 2.0**-self.man_bits)
        top_exp = 2**self.exp_bits - 1 - self.bias  # 指数全 1 仍可用，但尾数全 1 是 NaN
        return 2.0**top_exp * (2 - 2.0 ** (1 - self.man_bits))

    @property
    def min_normal(self) -> float:
        return 2.0 ** (1 - self.bias)

    @property
    def min_subnormal(self) -> float:
        return 2.0 ** (1 - self.bias - self.man_bits)

    @property
    def eps(self) -> float:
        """机器精度：1 与下一个可表示数之间的距离，2^(−man_bits)。"""
        return 2.0**-self.man_bits

    def decode(self, bits: int) -> float:
        """把一个整数位模式解释为实数值。"""
        sign = -1.0 if bits >> (self.exp_bits + self.man_bits) & 1 else 1.0
        e = bits >> self.man_bits & (2**self.exp_bits - 1)
        m = bits & (2**self.man_bits - 1)
        if self.ieee and e == 2**self.exp_bits - 1:
            return sign * math.inf if m == 0 else math.nan
        if not self.ieee and e == 2**self.exp_bits - 1 and m == 2**self.man_bits - 1:
            return math.nan
        if e == 0:  # 次正规数：没有隐含的 1，指数固定为 1 − bias
            return sign * (m / 2**self.man_bits) * 2.0 ** (1 - self.bias)
        return sign * (1 + m / 2**self.man_bits) * 2.0 ** (e - self.bias)


FP32 = FloatFormat("FP32", 8, 23)
FP16 = FloatFormat("FP16", 5, 10)
BF16 = FloatFormat("BF16", 8, 7)
FP8_E4M3 = FloatFormat("FP8 E4M3", 4, 3, ieee=False)
FP8_E5M2 = FloatFormat("FP8 E5M2", 5, 2)

TORCH_DTYPES = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16,
    "FP8 E4M3": torch.float8_e4m3fn,
    "FP8 E5M2": torch.float8_e5m2,
}


def bit_fields(value: float, fmt: FloatFormat) -> tuple[str, str, str]:
    """把 value 转成 fmt（就近舍入），返回 (符号, 指数, 尾数) 三段位串。"""
    dtype = TORCH_DTYPES[fmt.name]
    width = 1 + fmt.exp_bits + fmt.man_bits
    int_dtype = {8: torch.uint8, 16: torch.int16, 32: torch.int32}[width]
    raw = torch.tensor([value], dtype=torch.float32).to(dtype).view(int_dtype).item()
    bits = format(raw & (2**width - 1), f"0{width}b")
    return bits[0], bits[1 : 1 + fmt.exp_bits], bits[1 + fmt.exp_bits :]


# endregion


# region accumulate
def sequential_sum(values: Tensor, accumulate_dtype: torch.dtype) -> float:
    """逐个相加，每一步都把部分和舍入到 accumulate_dtype——模拟低精度累加器。"""
    total = torch.zeros((), dtype=accumulate_dtype)
    for v in values.to(accumulate_dtype):
        total = total + v
    return total.item()


def kahan_sum(values: Tensor, dtype: torch.dtype) -> float:
    """Kahan 补偿求和：用第二个变量 c 记住每次加法丢掉的低位，下一次加回来。"""
    total = torch.zeros((), dtype=dtype)
    c = torch.zeros((), dtype=dtype)
    for v in values.to(dtype):
        y = v - c
        t = total + y
        c = (t - total) - y  # (t − total) 是实际加上的量，减去 y 得到丢失的部分
        total = t
    return total.item()


# endregion


# region stable
def naive_softmax(z: Tensor) -> Tensor:
    e = torch.exp(z)
    return e / e.sum(dim=-1, keepdim=True)


def logsumexp(z: Tensor) -> Tensor:
    """log Σ exp(z_i) = m + log Σ exp(z_i − m)，m = max z：指数的参数都 ≤ 0，不会上溢。"""
    m = z.amax(dim=-1, keepdim=True)
    return (m + torch.log(torch.exp(z - m).sum(dim=-1, keepdim=True))).squeeze(-1)


def stable_softmax(z: Tensor) -> Tensor:
    return torch.exp(z - logsumexp(z)[..., None])


def naive_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """先 softmax 再取 log：概率下溢为 0 时得到 inf。"""
    p = naive_softmax(logits)
    return -torch.log(p.gather(-1, targets[:, None])).mean()


def stable_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """直接在对数域计算：ℓ = logsumexp(z) − z_y，永远不显式构造概率。"""
    return (logsumexp(logits) - logits.gather(-1, targets[:, None]).squeeze(-1)).mean()


# endregion


# region init
def init_std(fan_in: int, fan_out: int, scheme: str) -> float:
    """方差保持初始化的标准差。

    * xavier：Var(W) = 2 / (fan_in + fan_out)，兼顾前向与反向（线性/tanh）；
    * kaiming：Var(W) = 2 / fan_in，补偿 ReLU 砍掉一半的方差；
    * lecun：Var(W) = 1 / fan_in，只保证线性层前向方差不变。
    """
    if scheme == "xavier":
        return math.sqrt(2.0 / (fan_in + fan_out))
    if scheme == "kaiming":
        return math.sqrt(2.0 / fan_in)
    if scheme == "lecun":
        return math.sqrt(1.0 / fan_in)
    raise ValueError(scheme)


def activation_rms(
    depth: int, width: int, std: float, nonlinearity: str = "relu", batch: int = 512, seed: int = 0
) -> list[float]:
    """把标准正态输入送过 depth 层 (Linear → 非线性)，记录每层输出的均方根 sqrt(E[h²])。

    方差保持推导里传递的量正是二阶矩 E[h²]（ReLU 的输出均值不为 0，所以不用标准差）。
    """
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(batch, width, generator=g)
    rms = [h.square().mean().sqrt().item()]
    act = {"relu": torch.relu, "tanh": torch.tanh, "linear": lambda t: t}[nonlinearity]
    for _ in range(depth):
        w = torch.randn(width, width, generator=g) * std
        h = act(h @ w.T)
        rms.append(h.square().mean().sqrt().item())
    return rms


# endregion
