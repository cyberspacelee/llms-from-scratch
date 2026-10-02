"""20 · 2026：Qwen3.5/3.6/3.8 的 Gated DeltaNet + Gated Attention。

结构参照 Qwen 官方 config 和 Transformers 实现；小张量参考不加载官方权重。
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from ..config import AttentionConfig, ModelConfig, PositionConfig
from ..experiments.runners import consistency
from .ch11_gqa import config as gqa_config


def config() -> ModelConfig:
    """无输入；返回 3 个 Gated Delta + 1 个 GQA 的四层机制配方。

    核心配方用于缓存对照；下方函数单独补充短卷积、异构头、输出门机制。

    Args:
        无显式输入。

    Returns:
        ModelConfig 本章的独立配方。
    """
    c = gqa_config()
    attention = replace(c.block.attention, qk_norm=True)
    full = replace(c.block, attention=attention)
    linear = replace(
        full, attention=AttentionConfig(kind="gated_delta", position=PositionConfig(kind="none"))
    )
    return replace(c, layers=4, block=full, layer_blocks=(linear, linear, linear, full))


def causal_depthwise_conv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """输入 x[B,T,C]、weight[C,1,W]；返回 SiLU(因果 depthwise conv)[B,T,C]。

    左侧补 W-1 个零，每个通道独立，不读取未来；不持有卷积缓存。

    Args:
        x: float [B,T,C]，短卷积的序列输入。
        weight: 卷积 weight[C,1,W]。

    Returns:
        float [B,T,C] 因果短卷积与 SiLU输出。
    """
    if (
        x.ndim != 3
        or weight.ndim != 3
        or weight.shape[:2] != (x.shape[-1], 1)
        or weight.shape[-1] < 1
        or x.shape[1] < 1
    ):
        raise ValueError("expected x[B,T,C] and weight[C,1,W]")
    transposed = x.transpose(1, 2)  # [B,C,T]
    padded = F.pad(transposed, (weight.shape[-1] - 1, 0))  # [B,C,T+W-1]
    return F.silu(F.conv1d(padded, weight, groups=x.shape[-1])).transpose(1, 2)


def gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    gate: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """输入 q/k[B,H_k,T,D_k]、v/gate[B,H_v,T,D_v]、beta/log_decay[B,H_v,T]。

    H_v 必须是 H_k 的倍数；beta∈[0,1]、log_decay≤0。state 可为 [B,H_v,D_k,D_v]。
    返回 gated RMS-normalized context[B,H_v,T,D_v] 和新 state[B,H_v,D_k,D_v]。

    Args:
        q: float [B,H_k,T,D_k]，在递归前归一化并扩展到 H_v。
        k: float [B,H_k,T,D_k]，在递归前归一化并扩展到 H_v。
        v: float [B,H_v,T,D_v]，H_v 是 H_k 的倍数。
        beta: float [B,H_v,T] 更新门，范围 [0,1]。
        log_decay: float [B,H_v,T]，有限且≤0；指数后为衰减因子。
        gate: float [B,H_v,T,D_v] 输出门的预激活值。
        state: 可选 float [B,H_v,D_k,D_v] 递归状态。

    Returns:
        tuple: context[B,H_v,T,D_v]，新state[B,H_v,D_k,D_v]。
    """
    if (
        q.ndim != 4
        or k.shape != q.shape
        or v.ndim != 4
        or gate.shape != v.shape
        or v.shape[0] != q.shape[0]
        or v.shape[2] != q.shape[2]
        or q.shape[1] < 1
        or v.shape[1] % q.shape[1]
        or q.shape[2] < 1
        or beta.shape != v.shape[:3]
        or log_decay.shape != beta.shape
    ):
        raise ValueError("incompatible Gated Delta tensors")
    if (
        not torch.isfinite(beta).all()
        or not torch.isfinite(log_decay).all()
        or (beta < 0).any()
        or (beta > 1).any()
        or (log_decay > 0).any()
    ):
        raise ValueError("beta must be in [0,1], log_decay finite and nonpositive")
    q, k = (
        F.normalize(t, dim=-1).repeat_interleave(v.shape[1] // q.shape[1], 1) for t in (q, k)
    )  # [B,H_v,T,D_k]
    q = q * q.shape[-1] ** -0.5
    shape = (q.shape[0], v.shape[1], q.shape[-1], v.shape[-1])
    if state is None:
        state = q.new_zeros(shape)
    elif state.shape != shape or state.device != q.device or state.dtype != q.dtype:
        raise ValueError("invalid recurrent state")
    outputs = []
    # ponytail: tokenwise recurrence; chunkwise kernels are needed for large-scale training.
    for index in range(q.shape[2]):
        qi, ki, vi = q[:, :, index], k[:, :, index], v[:, :, index]  # [B,H_v,D_k/D_v]
        decayed = state * log_decay[:, :, index, None, None].exp()  # [B,H_v,D_k,D_v]
        residual = vi - (ki.unsqueeze(-2) @ decayed).squeeze(-2)  # [B,H_v,D_v]
        state = decayed + beta[:, :, index, None, None] * ki.unsqueeze(-1) * residual.unsqueeze(-2)
        outputs.append((qi.unsqueeze(-2) @ state).squeeze(-2))  # [B,H_v,D_v]
    context = torch.stack(outputs, 2)  # [B,H_v,T,D_v]
    work = context if context.dtype == torch.float64 else context.float()
    normalized = (work * torch.rsqrt(work.square().mean(-1, keepdim=True) + 1e-6)).to(context.dtype)
    return normalized * F.silu(gate), state


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回短卷积因果性、异构头递归 chunk一致性和 3:1 配方检查。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    x = torch.randn(2, 6, 32, device=device, dtype=torch.float64, requires_grad=True)
    conv_weight = torch.randn(32, 1, 3, device=device, dtype=x.dtype, requires_grad=True)
    convolved = causal_depthwise_conv(x, conv_weight)  # [B,T,C]
    changed = x.detach().clone()
    changed[:, 4:] += 100
    torch.testing.assert_close(convolved[:, :4], causal_depthwise_conv(changed, conv_weight)[:, :4])
    q, k = (torch.randn(2, 2, 6, 4, device=device, dtype=x.dtype) for _ in range(2))
    v, gate = (
        torch.randn(2, 4, 6, 3, device=device, dtype=x.dtype, requires_grad=True) for _ in range(2)
    )
    beta = torch.randn(2, 4, 6, device=device, dtype=x.dtype).sigmoid()
    log_decay = -F.softplus(
        torch.randn_like(beta)
    )  # -exp(A_log)*softplus(a+dt_bias), 此处固定 A_log/dt_bias=0
    output, state = gated_delta(q, k, v, beta, log_decay, gate)
    first, prefix = gated_delta(
        q[:, :, :3], k[:, :, :3], v[:, :, :3], beta[:, :, :3], log_decay[:, :, :3], gate[:, :, :3]
    )
    rest, last = gated_delta(
        q[:, :, 3:],
        k[:, :, 3:],
        v[:, :, 3:],
        beta[:, :, 3:],
        log_decay[:, :, 3:],
        gate[:, :, 3:],
        prefix,
    )
    torch.testing.assert_close(output, torch.cat((first, rest), 2))
    torch.testing.assert_close(state, last)
    (output.square().mean() + convolved.square().mean()).backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()
    assert conv_weight.grad is not None and torch.isfinite(conv_weight.grad).all()
    # Full Attention 的 gate 在合头前逐元素 sigmoid；D 可以不等于 H_q*D_h。
    gate_projection = nn.Linear(32, 4 * 8, device=device, dtype=x.dtype)
    full_gate = gate_projection(x).reshape(2, 6, 4, 8).transpose(1, 2).sigmoid()  # [B,H_q,T,D_h]
    return {
        "conv_shape": list(convolved.shape),
        "qk_shape": list(q.shape),
        "value_shape": list(v.shape),
        "recurrent_output_shape": list(output.shape),
        "state_shape": list(state.shape),
        "full_attention_gate_shape": list(full_gate.shape),
        "hybrid_check": consistency(config(), device),
        "note": "mechanism references; the shared core recipe uses simpler gates and has no convolution cache",
    }
