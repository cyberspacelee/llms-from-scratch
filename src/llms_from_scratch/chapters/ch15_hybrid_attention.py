"""15 · 2024–2025：hybrid_attention，机制与数值核对独立成章。"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.nn import functional as F

from ..analysis import attention_cost
from ..config import ModelConfig, PositionConfig
from ..experiments.runners import consistency
from .ch11_gqa import config as gqa_config


def config() -> ModelConfig:
    """无输入；返回本章机制的显式配置，其他支线不自动启用。

    Args:
        无显式输入。

    Returns:
        ModelConfig 本章的独立配方。
    """
    return replace(gqa_config(), layers=2)


def channel_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    log_decay: torch.Tensor,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA 的逐 key 通道衰减参考；状态按 [D_k,D_v] 存放。

    Args:
        q: float [B,H,T,D_k]，读取前做 L2 normalization 与 D_k**-0.5 缩放。
        k: float [B,H,T,D_k]，写入前做 L2 normalization。
        v: float [B,H,T,D_v]。
        beta: float [B,H,T]，更新门，范围 [0,1]。
        log_decay: float [B,H,T,D_k]，有限且不大于零。
        state: 可选 float [B,H,D_k,D_v]，None 从零状态开始。

    Returns:
        tuple: context[B,H,T,D_v]、末尾 state[B,H,D_k,D_v]。
    """
    if (
        q.ndim != 4
        or k.shape != q.shape
        or v.ndim != 4
        or v.shape[:3] != q.shape[:3]
        or beta.shape != q.shape[:3]
        or log_decay.shape != q.shape
        or min(q.shape) < 1
        or v.shape[-1] < 1
    ):
        raise ValueError("incompatible channel-wise delta shapes")
    tensors = (q, k, v, beta, log_decay)
    if any(
        t.dtype != q.dtype or t.device != q.device or not t.is_floating_point() for t in tensors
    ):
        raise ValueError("all inputs must share a floating dtype and device")
    if (
        not torch.isfinite(log_decay).all()
        or (log_decay > 0).any()
        or not torch.isfinite(beta).all()
        or ((beta < 0) | (beta > 1)).any()
    ):
        raise ValueError("expected nonpositive log_decay and beta in [0,1]")
    q, k = F.normalize(q, dim=-1) * q.shape[-1] ** -0.5, F.normalize(k, dim=-1)
    shape = (*q.shape[:2], q.shape[-1], v.shape[-1])
    if state is None:
        state = q.new_zeros(shape)
    elif state.shape != shape or state.dtype != q.dtype or state.device != q.device:
        raise ValueError("invalid channel-wise recurrent state")
    outputs = []
    # ponytail: sequential KDA recurrence; production training needs chunkwise parallel kernels.
    for index in range(q.shape[2]):
        qi, ki, vi = q[:, :, index], k[:, :, index], v[:, :, index]
        decayed = log_decay[:, :, index].exp().unsqueeze(-1) * state  # [B,H,D_k,D_v]
        error = vi - (ki.unsqueeze(-2) @ decayed).squeeze(-2)  # [B,H,D_v]
        state = decayed + beta[:, :, index, None, None] * ki.unsqueeze(-1) * error.unsqueeze(-2)
        outputs.append((qi.unsqueeze(-2) @ state).squeeze(-2))  # [B,H,D_v]
    return torch.stack(outputs, 2), state


def run(device: torch.device) -> dict[str, object]:
    """递归层与 softmax 层共享 Block 接口；递归状态大小不随 S 增长。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    # region state_size
    c = config()
    report = {}
    for kind in ("linear", "delta", "gated_delta"):
        a = replace(
            c.block.attention, kind=kind, qk_norm=False, position=PositionConfig(kind="none")
        )
        variant = replace(c, block=replace(c.block, attention=a))
        short = attention_cost(a, c.dim, key_tokens=16, bytes_per_element=4).cache_bytes
        long = attention_cost(a, c.dim, key_tokens=4096, bytes_per_element=4).cache_bytes
        assert short == long
        report[kind] = {"state_bytes_fp32": short, "check": consistency(variant, device)}
    hybrid = replace(c, layer_blocks=(variant.block, c.block))
    report["gated_delta_then_gqa"] = consistency(hybrid, device)
    # endregion state_size
    # iRoPE 入门分支：在不同层之间交替使用 RoPE/NoPE，不是对单个 Q/K 做半旋转。
    nope = replace(
        c.block, attention=replace(c.block.attention, position=PositionConfig(kind="none"))
    )
    report["rope_then_nope"] = consistency(replace(c, layer_blocks=(c.block, nope)), device)
    q, k, v = (
        torch.randn(2, 3, 6, 4, device=device, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    )
    beta = torch.randn(2, 3, 6, device=device, dtype=q.dtype).sigmoid()
    decay = -F.softplus(torch.randn_like(q))  # [B,H,T,D_k]，区别于 scalar alpha[B,H,T]
    output, state = channel_delta(q, k, v, beta, decay)
    first, prefix = channel_delta(
        q[:, :, :2], k[:, :, :2], v[:, :, :2], beta[:, :, :2], decay[:, :, :2]
    )
    rest, last = channel_delta(
        q[:, :, 2:], k[:, :, 2:], v[:, :, 2:], beta[:, :, 2:], decay[:, :, 2:], prefix
    )
    torch.testing.assert_close(output, torch.cat((first, rest), 2))
    torch.testing.assert_close(state, last)
    output.square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    report["kda_channel_decay"] = {
        "decay_shape": list(decay.shape),
        "state_shape": list(state.shape),
        "chunk_check": "passed",
    }
    return report
