"""FlashAttention-2 前向的 Triton 实现（参照 CS336 作业 2 与 Triton 官方 fused attention 教程）。

* 网格为 ``(cdiv(T_q, Q_TILE), N)``：每个 program 负责一个 Q 块，循环所有（因果时为左下方的）K/V 块；
* 前向输出 O 与 logsumexp L；反向复用 CPU/GPU 通用的分块重计算实现
  :func:`llms_from_scratch.gpu.flash_attention.flash_attention_backward`（纯 PyTorch 运算）。
* ``D`` 需要是 ≥ 16 的 2 的幂（``tl.dot`` 的要求）。

无 Triton 时模块仍可导入，``HAS_TRITON`` 为假。
"""

from __future__ import annotations

import math

import torch

from llms_from_scratch.gpu.flash_attention import flash_attention_backward

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - 取决于环境
    HAS_TRITON = False


if HAS_TRITON:  # pragma: no cover - 需要 GPU

    # region flash_fwd_kernel
    @triton.jit
    def flash_fwd_kernel(
        Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS, scale,
        D: tl.constexpr, Q_TILE: tl.constexpr, K_TILE: tl.constexpr, IS_CAUSAL: tl.constexpr,
    ):
        q_tile = tl.program_id(0)  # 第几个 Q 块
        b = tl.program_id(1)  # 第几个 (batch, head)
        q_block = tl.make_block_ptr(
            Q_ptr + b * stride_qb, shape=(N_QUERIES, D), strides=(stride_qq, stride_qd),
            offsets=(q_tile * Q_TILE, 0), block_shape=(Q_TILE, D), order=(1, 0),
        )
        k_block = tl.make_block_ptr(
            K_ptr + b * stride_kb, shape=(N_KEYS, D), strides=(stride_kk, stride_kd),
            offsets=(0, 0), block_shape=(K_TILE, D), order=(1, 0),
        )
        v_block = tl.make_block_ptr(
            V_ptr + b * stride_vb, shape=(N_KEYS, D), strides=(stride_vk, stride_vd),
            offsets=(0, 0), block_shape=(K_TILE, D), order=(1, 0),
        )
        q = tl.load(q_block, boundary_check=(0, 1), padding_option="zero")  # 读一次，常驻寄存器
        q_idx = q_tile * Q_TILE + tl.arange(0, Q_TILE)
        m = tl.full((Q_TILE,), float("-inf"), dtype=tl.float32)
        ell = tl.zeros((Q_TILE,), dtype=tl.float32)
        acc = tl.zeros((Q_TILE, D), dtype=tl.float32)
        hi = N_KEYS
        if IS_CAUSAL:  # 只遍历对角线及其左侧的 K 块
            hi = tl.minimum(N_KEYS, (q_tile + 1) * Q_TILE)
        for start in range(0, hi, K_TILE):
            k = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
            v = tl.load(v_block, boundary_check=(0, 1), padding_option="zero")
            s = tl.dot(q, tl.trans(k)) * scale  # [Q_TILE, K_TILE]，FP32
            k_idx = start + tl.arange(0, K_TILE)
            valid = k_idx[None, :] < N_KEYS
            if IS_CAUSAL:
                valid = valid & (k_idx[None, :] <= q_idx[:, None])
            s = tl.where(valid, s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, axis=1))
            p = tl.exp(s - m_new[:, None])
            alpha = tl.exp(m - m_new)
            ell = alpha * ell + tl.sum(p, axis=1)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)  # P 转回 BF16/FP16 走 Tensor Core
            m = m_new
            k_block = tl.advance(k_block, (K_TILE, 0))
            v_block = tl.advance(v_block, (K_TILE, 0))
        acc = acc / ell[:, None]
        o_block = tl.make_block_ptr(
            O_ptr + b * stride_ob, shape=(N_QUERIES, D), strides=(stride_oq, stride_od),
            offsets=(q_tile * Q_TILE, 0), block_shape=(Q_TILE, D), order=(1, 0),
        )
        tl.store(o_block, acc.to(o_block.dtype.element_ty), boundary_check=(0, 1))
        tl.store(L_ptr + b * stride_lb + q_idx * stride_lq, m + tl.log(ell), mask=q_idx < N_QUERIES)

    # endregion flash_fwd_kernel

    # region flash_launch
    def flash_forward_triton(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False,
        q_tile: int = 64, k_tile: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n, t_q, d = q.shape
        t_k = k.shape[1]
        o = torch.empty_like(q)
        lse = torch.empty(n, t_q, device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(t_q, q_tile), n)
        flash_fwd_kernel[grid](
            q, k, v, o, lse,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            lse.stride(0), lse.stride(1),
            t_q, t_k, 1.0 / math.sqrt(d),
            D=d, Q_TILE=q_tile, K_TILE=k_tile, IS_CAUSAL=causal,
        )
        return o, lse

    # endregion flash_launch


# region autograd_function
class FlashAttentionTriton(torch.autograd.Function):
    """前向用 Triton kernel；反向用分块重计算（只保存 Q、K、V、O 与 L）。"""

    @staticmethod
    def forward(ctx, q, k, v, causal=False):
        o, lse = flash_forward_triton(q, k, v, causal)
        ctx.save_for_backward(q, k, v, o, lse)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse = ctx.saved_tensors
        f32 = [t.float() for t in (q, k, v, o, do)]
        dq, dk, dv = flash_attention_backward(*f32, lse, causal=ctx.causal, block_q=64, block_k=64)
        return dq.to(q.dtype), dk.to(k.dtype), dv.to(v.dtype), None


# endregion autograd_function
