"""22 · 2025–2026：DeepSeek mHC、Qwen4-Exp GR 与 Kimi AttnRes。

同样保持 streams[B,T,N_streams,D]，mHC 的 doubly-stochastic mixing 与 GR gate 不同。
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..layers import RMSNorm


def sinkhorn(logits: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    """输入有限 float logits[...,N,N]、迭代次数；返回近似双随机矩阵同 shape。

    fp16/bf16 统计升 fp32，fp64 保留；每行/列和近似 1，元素非负。

    Args:
        logits: finite float [...,N_streams,N_streams]，未归一化的 mixing scores。
        iterations: Sinkhorn 迭代次数，正整数。

    Returns:
        float [...,N,N] 近似双随机矩阵，统计 dtype为fp32/fp64。
    """
    if (
        logits.ndim < 2
        or logits.shape[-1] != logits.shape[-2]
        or logits.shape[-1] < 1
        or not logits.is_floating_point()
        or not torch.isfinite(logits).all()
        or type(iterations) is not int
        or iterations < 1
    ):
        raise ValueError("expected finite square matrices and positive iterations")
    work = logits if logits.dtype == torch.float64 else logits.float()
    matrix = (
        (work - work.amax((-2, -1), keepdim=True)).exp().clamp_min(torch.finfo(work.dtype).tiny)
    )
    for _ in range(iterations):
        matrix = matrix / matrix.sum(-1, keepdim=True)
        matrix = matrix / matrix.sum(-2, keepdim=True)
    return matrix


def mhc_update(
    streams: torch.Tensor, branch: torch.Tensor, post: torch.Tensor, mixing: torch.Tensor
) -> torch.Tensor:
    """输入 streams[B,T,N,D]、branch[B,T,D]、post[B,T,N]、mixing[B,T,N,N]。

    返回 mixing@streams + post*branch 的 [B,T,N,D] 新残差流。

    Args:
        streams: float [B,T,N_streams,D] 多残差流。
        branch: float [B,T,D] 分支输出。
        post: float [B,T,N_streams] 分支注入权重。
        mixing: float [B,T,N_streams,N_streams] 残差流混合矩阵。

    Returns:
        float [B,T,N_streams,D] 新残差状态。
    """
    if (
        streams.ndim != 4
        or branch.shape != streams.shape[:2] + streams.shape[-1:]
        or post.shape != streams.shape[:-1]
        or mixing.shape != streams.shape[:-1] + (streams.shape[-2],)
    ):
        raise ValueError("incompatible residual stream shapes")
    return mixing @ streams + post.unsqueeze(-1) * branch.unsqueeze(-2)


def attention_residual(
    sources: torch.Tensor, pseudo_query: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """沿深度选择已有 residual sources；sequence 轴保持不变。

    sources 可为已完成 block 与当前 partial sum 的 stack；这里不调度 block 边界。

    Args:
        sources: float [B,T,N_sources,D]，各层/块原始输出。
        pseudo_query: float [D]，当前子层的可学习深度查询向量。

    Returns:
        tuple: hidden[B,T,D]、depth_weights[B,T,N_sources]，后者沿深度求和为1。
    """
    if sources.ndim != 4 or min(sources.shape) < 1 or pseudo_query.shape != sources.shape[-1:]:
        raise ValueError("expected nonempty sources[B,T,N_sources,D] and pseudo_query[D]")
    if (
        not sources.is_floating_point()
        or pseudo_query.dtype != sources.dtype
        or pseudo_query.device != sources.device
    ):
        raise ValueError("sources and query must share a floating dtype and device")
    work = sources if sources.dtype == torch.float64 else sources.float()
    keys = work * torch.rsqrt(work.square().mean(-1, keepdim=True) + 1e-6)
    scores = (keys * pseudo_query.to(work.dtype)).sum(-1)  # [B,T,N_sources]
    weights = scores.softmax(-1).to(sources.dtype)
    return (weights.unsqueeze(-1) * sources).sum(-2), weights  # [B,T,D]


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 mHC/GR shape、双随机性、梯度和残差保存检查。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    x = torch.randn(2, 5, 3, 8, device=device, dtype=torch.float64, requires_grad=True)
    flat = x.flatten(-2)  # [B,T,N*D=24]
    norm = RMSNorm(24).to(device).double()
    projection = nn.Linear(24, 15, device=device, dtype=x.dtype)  # N*N+2N
    raw = projection(norm(flat))
    pre, post, mixes = raw.split((3, 3, 9), -1)
    pre, post = pre.sigmoid(), 2 * post.sigmoid()  # [B,T,N]
    mixing = sinkhorn(mixes.reshape(2, 5, 3, 3))  # [B,T,N,N]
    h = (pre.unsqueeze(-1) * x).sum(-2)  # [B,T,D]
    branch = F.silu(h)
    result = mhc_update(x, branch, post, mixing)  # [B,T,N,D]
    torch.testing.assert_close(
        mixing.sum(-1), torch.ones_like(mixing.sum(-1)), atol=1e-8, rtol=1e-8
    )
    torch.testing.assert_close(
        mixing.sum(-2), torch.ones_like(mixing.sum(-2)), atol=1e-8, rtol=1e-8
    )
    result.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    # GR 用每个 stream 的每个坐标 gate；不对 stream 混合矩阵做 Sinkhorn。
    down, up = (
        nn.Linear(24, 4, device=device, dtype=x.dtype),
        nn.Linear(4, 24, device=device, dtype=x.dtype),
    )
    gates = up(F.silu(down(norm(flat)) / 3)).sigmoid().reshape_as(x)  # [B,T,N,D]
    gr_input = (gates * norm(flat).reshape_as(x)).mean(-2)  # [B,T,D]
    gr_result = x + post.unsqueeze(-1) * F.silu(gr_input).unsqueeze(-2)
    sources = x.detach().clone().requires_grad_()  # 此处第3轴解释为 depth source，不是并行 stream。
    pseudo_query = torch.zeros(8, device=device, dtype=x.dtype, requires_grad=True)
    selected, depth_weights = attention_residual(sources, pseudo_query)
    torch.testing.assert_close(selected, sources.mean(-2))  # 零初始化得到等权平均。
    torch.testing.assert_close(depth_weights.sum(-1), torch.ones_like(depth_weights.sum(-1)))
    selected.square().mean().backward()
    assert pseudo_query.grad is not None and torch.isfinite(pseudo_query.grad).all()
    assert sources.grad is not None and torch.isfinite(sources.grad).all()
    return {
        "streams_shape": list(x.shape),
        "collapsed_shape": list(h.shape),
        "mixing_shape": list(mixing.shape),
        "mhc_output_shape": list(result.shape),
        "gr_gate_shape": list(gates.shape),
        "gr_output_shape": list(gr_result.shape),
        "attnres_depth_weights_shape": list(depth_weights.shape),
        "attnres_output_shape": list(selected.shape),
        "checks": ["nonnegative row/column sums", "finite gradients"],
        "note": "mHC/GR/AttnRes mechanism references; no Single-Pass fusion, block scheduling or checkpoint mapping",
    }
