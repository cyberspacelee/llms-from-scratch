"""推理量化：整数量化基础、GPTQ、AWQ、SmoothQuant、FP8 与 KV Cache 量化。

全部用浮点张量“模拟”量化（fake quantization）：先量化到整数网格再反量化回浮点，
这样可以在 CPU 上精确测量误差。真实部署时整数权重与 scale 分开存放，由 kernel 在计算时反量化。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# region basics
@dataclass
class QTensor:
    q: torch.Tensor  # 整数码，形状 [..., n_groups, group]
    scale: torch.Tensor  # [..., n_groups, 1]
    zero: torch.Tensor | None  # 非对称量化的零点，形状同 scale
    shape: torch.Size

    def dequantize(self) -> torch.Tensor:
        q = self.q.float() if self.zero is None else self.q.float() - self.zero
        return (q * self.scale).reshape(self.shape)


def _groups(x: torch.Tensor, granularity: str, group_size: int | None) -> torch.Tensor:
    if granularity == "tensor":
        return x.reshape(1, -1)
    if granularity == "channel":  # 每一行（线性层的每个输出通道）一个 scale
        return x.reshape(*x.shape[:-1], 1, x.shape[-1])
    if granularity == "group":  # 沿输入维每 group_size 个元素一个 scale
        return x.reshape(*x.shape[:-1], x.shape[-1] // group_size, group_size)
    raise ValueError(granularity)


def quantize(x: torch.Tensor, bits: int = 8, symmetric: bool = True,
             granularity: str = "channel", group_size: int | None = None) -> QTensor:
    g = _groups(x.float(), granularity, group_size)
    if symmetric:
        qmax = 2 ** (bits - 1) - 1  # 对称区间 [-qmax, qmax]，例如 INT8 是 [-127, 127]
        scale = g.abs().amax(-1, keepdim=True).clamp(min=1e-12) / qmax
        q = torch.clamp(torch.round(g / scale), -qmax, qmax)
        return QTensor(q, scale, None, x.shape)
    lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True)
    levels = 2 ** bits - 1  # 非对称区间 [0, 2^b − 1]
    scale = (hi - lo).clamp(min=1e-12) / levels
    zero = torch.round(-lo / scale)
    q = torch.clamp(torch.round(g / scale) + zero, 0, levels)
    return QTensor(q, scale, zero, x.shape)


def fake_quantize(x: torch.Tensor, **kwargs) -> torch.Tensor:
    return quantize(x, **kwargs).dequantize().to(x.dtype)
# endregion basics


def output_error(x: torch.Tensor, w: torch.Tensor, w_hat: torch.Tensor) -> float:
    """线性层输出的相对误差 ‖XWᵀ − XŴᵀ‖ / ‖XWᵀ‖。"""
    ref = x @ w.T
    return float((ref - x @ w_hat.T).norm() / ref.norm())


# region gptq
def gptq_quantize(w: torch.Tensor, x: torch.Tensor, bits: int = 4, group_size: int = 128,
                  damp: float = 0.01) -> tuple[torch.Tensor, torch.Tensor]:
    """GPTQ（不含懒批量更新的简化版）：逐列量化，并把每一列的误差按 H⁻¹ 补偿给尚未量化的列。

    w: [out, in]；x: 校准激活 [n, in]。返回反量化后的权重（与 w 同形状）与每组的 scale。
    """
    w = w.clone().float()
    n_in = w.shape[1]
    h = 2 * x.float().T @ x.float()  # 层输出平方误差关于 w 每一行的 Hessian
    dead = torch.diag(h) == 0  # 从未被激活的输入通道：权重无关紧要
    h[dead, dead] = 1
    w[:, dead] = 0
    h += damp * torch.diag(h).mean() * torch.eye(n_in)  # 阻尼，保证数值上正定
    # H⁻¹ 的上三角 Cholesky 因子：第 i 行恰好给出“去掉前 i 列后”的逆 Hessian 所需的量
    h_inv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    w_hat = torch.zeros_like(w)
    scales = []
    for i in range(n_in):
        if i % group_size == 0:  # 分组的 scale 用“已被前面补偿过”的当前权重来定
            group = w[:, i:i + group_size]
            scale = group.abs().amax(1).clamp(min=1e-12) / (2 ** (bits - 1) - 1)
            scales.append(scale)
        qmax = 2 ** (bits - 1) - 1
        q = torch.clamp(torch.round(w[:, i] / scale), -qmax, qmax) * scale
        w_hat[:, i] = q
        err = (w[:, i] - q) / h_inv[i, i]
        w[:, i + 1:] -= err[:, None] * h_inv[i, i + 1:][None, :]  # OBS 式的误差补偿
    return w_hat, torch.stack(scales, dim=1)
# endregion gptq


def rtn_quantize(w: torch.Tensor, bits: int = 4, group_size: int = 128) -> torch.Tensor:
    """最近取整（round-to-nearest）的分组对称量化，作为 GPTQ 与 AWQ 的基线。"""
    return fake_quantize(w, bits=bits, granularity="group", group_size=group_size)


# region awq
def awq_quantize(w: torch.Tensor, x: torch.Tensor, bits: int = 4, group_size: int = 128,
                 n_grid: int = 20) -> tuple[torch.Tensor, torch.Tensor, float]:
    """AWQ：按激活幅度给输入通道乘缩放 s（s = mean|x|^α），在网格上搜 α 使输出误差最小。

    量化的是 W·diag(s)，推理时 x 先除以 s（可融进前一层），所以返回 Q(W·diag(s))·diag(s)⁻¹。
    """
    x_mag = x.abs().mean(0)
    best = (float("inf"), None, None)
    for k in range(n_grid + 1):
        alpha = k / n_grid
        s = x_mag.pow(alpha).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()  # 归一化，避免整体缩放影响分组的取值范围
        w_hat = rtn_quantize(w * s, bits, group_size) / s
        err = output_error(x, w, w_hat)
        if err < best[0]:
            best = (err, w_hat, s)
    err, w_hat, s = best
    return w_hat, s, err
# endregion awq


# region smooth
def smooth_scales(x_absmax: torch.Tensor, w: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """SmoothQuant 的逐输入通道迁移因子 s_j = max|X_j|^α / max|W_j|^(1−α)。"""
    w_absmax = w.abs().amax(0)  # w: [out, in]，按输入通道取最大
    return (x_absmax.pow(alpha) / w_absmax.pow(1 - alpha)).clamp(min=1e-5)


def w8a8_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """激活逐 token、权重逐输出通道的对称 INT8 量化；整数矩阵乘后用两组 scale 还原。"""
    qx = quantize(x, bits=8, granularity="channel")  # x 的每一行（token）一个 scale
    qw = quantize(w, bits=8, granularity="channel")
    acc = qx.q.squeeze(-2) @ qw.q.squeeze(-2).T  # 真实硬件上是 INT8×INT8→INT32
    return acc * qx.scale.squeeze(-1) * qw.scale.squeeze(-1).T
# endregion smooth


# region fp8
FP8_E4M3_MAX = 448.0


def fp8_quantize(x: torch.Tensor, per_token: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """缩放到 E4M3 的表示范围后转换为 float8_e4m3fn；返回 FP8 张量与 scale。"""
    amax = x.abs().amax(-1, keepdim=True) if per_token else x.abs().amax()
    scale = amax.float().clamp(min=1e-12) / FP8_E4M3_MAX
    return (x.float() / scale).to(torch.float8_e4m3fn), scale


def fp8_dequantize(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale
# endregion fp8


# region kv
def quantize_kv(k: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """KV Cache 量化（模拟）：每个 token 的每个头一个 scale，即沿 head_dim 分组。"""
    return fake_quantize(k, bits=bits, granularity="channel")
# endregion kv
