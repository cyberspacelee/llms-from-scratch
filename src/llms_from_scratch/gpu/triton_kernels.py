"""Triton kernel：向量加法、融合 softmax、分组排序的矩阵乘法（带 autotune）。

没有安装 Triton（或没有 GPU）时模块仍可导入：kernel 只在 ``HAS_TRITON`` 为真时定义，
同时提供按 program 粒度执行的 CPU 参考实现（``*_ref``），用来测试算法本身。
"""

from __future__ import annotations

import torch

try:  # Triton 只在 Linux + GPU 环境中可用
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - 取决于环境
    HAS_TRITON = False


if HAS_TRITON:  # pragma: no cover - 需要 GPU

    # region add_kernel
    @triton.jit
    def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(axis=0)  # 第几个 program（≈ CUDA 的 block）
        offsets = pid * BLOCK + tl.arange(0, BLOCK)  # 本 program 负责的 BLOCK 个下标
        mask = offsets < n  # 最后一块可能越界
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x + y, mask=mask)

    def add(x: torch.Tensor, y: torch.Tensor, block: int = 1024) -> torch.Tensor:
        out = torch.empty_like(x)
        n = out.numel()
        grid = (triton.cdiv(n, block),)
        add_kernel[grid](x, y, out, n, BLOCK=block)
        return out

    # endregion add_kernel

    # region softmax_kernel
    @triton.jit
    def softmax_kernel(
        out_ptr, in_ptr, in_row_stride, out_row_stride, n_cols, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0)  # 一个 program 处理一整行
        cols = tl.arange(0, BLOCK)  # BLOCK 是 ≥ n_cols 的 2 的幂
        mask = cols < n_cols
        x = tl.load(in_ptr + row * in_row_stride + cols, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)  # 整行都在寄存器里：一次读，一次写
        num = tl.exp(x)
        y = num / tl.sum(num, axis=0)
        tl.store(out_ptr + row * out_row_stride + cols, y, mask=mask)

    def softmax(x: torch.Tensor) -> torch.Tensor:
        rows, cols = x.shape
        out = torch.empty_like(x)
        block = triton.next_power_of_2(cols)
        num_warps = 4 if block <= 2048 else 8 if block <= 4096 else 16
        softmax_kernel[(rows,)](
            out, x, x.stride(0), out.stride(0), cols, BLOCK=block, num_warps=num_warps
        )
        return out

    # endregion softmax_kernel

    # region matmul_kernel
    def _configs():
        return [
            triton.Config({"BM": 128, "BN": 256, "BK": 64, "GROUP_M": 8}, num_stages=3, num_warps=8),
            triton.Config({"BM": 128, "BN": 128, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BM": 64, "BN": 128, "BK": 32, "GROUP_M": 8}, num_stages=4, num_warps=4),
            triton.Config({"BM": 64, "BN": 64, "BK": 32, "GROUP_M": 8}, num_stages=5, num_warps=2),
        ]

    @triton.autotune(configs=_configs(), key=["M", "N", "K"])
    @triton.jit
    def matmul_kernel(
        a_ptr, b_ptr, c_ptr, M, N, K,
        stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
    ):
        # —— 分组排序：把一维 pid 映射到 (pid_m, pid_n)，让相邻 program 共享 A 的行块 ——
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BM)
        num_pid_n = tl.cdiv(N, BN)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        # —— 指针块：A 的 BM×BK 子块与 B 的 BK×BN 子块 ——
        offs_m = (pid_m * BM + tl.arange(0, BM)) % M
        offs_n = (pid_n * BN + tl.arange(0, BN)) % N
        offs_k = tl.arange(0, BK)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BM, BN), dtype=tl.float32)  # FP32 累加器
        for k in range(0, tl.cdiv(K, BK)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BK, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BK, other=0.0)
            acc = tl.dot(a, b, acc)  # 编译到 Tensor Core（mma / wgmma）
            a_ptrs += BK * stride_ak
            b_ptrs += BK * stride_bk
        c = acc.to(c_ptr.dtype.element_ty)
        offs_cm = pid_m * BM + tl.arange(0, BM)
        offs_cn = pid_n * BN + tl.arange(0, BN)
        c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
        tl.store(c_ptrs, c, mask=(offs_cm[:, None] < M) & (offs_cn[None, :] < N))

    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        M, K = a.shape
        _, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)

        def grid(meta):
            return (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),)

        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        )
        return c

    # endregion matmul_kernel


# ---------------------------------------------------------------------------
# CPU 参考：按 program 粒度执行同样的分块逻辑
# ---------------------------------------------------------------------------


# region add_ref
def add_ref(x: torch.Tensor, y: torch.Tensor, block: int = 1024) -> torch.Tensor:
    """逐个 program 执行 add_kernel：每个 program 处理 BLOCK 个元素，越界部分被掩码。"""
    n = x.numel()
    out = torch.empty_like(x)
    for pid in range((n + block - 1) // block):
        offsets = pid * block + torch.arange(block)
        mask = offsets < n
        idx = offsets[mask]
        out[idx] = x[idx] + y[idx]
    return out


# endregion add_ref


def softmax_ref(x: torch.Tensor) -> torch.Tensor:
    """逐行执行 softmax_kernel：补 -inf 到 2 的幂长度后求 max、exp、sum。"""
    rows, cols = x.shape
    block = 1 << (cols - 1).bit_length()
    out = torch.empty_like(x)
    for row in range(rows):
        padded = torch.full((block,), float("-inf"), dtype=x.dtype)
        padded[:cols] = x[row]
        z = padded - padded.max()
        num = torch.exp(z)
        out[row] = (num / num.sum())[:cols]
    return out


# region grouped_order
def program_to_tile(pid: int, num_pid_m: int, num_pid_n: int, group_m: int) -> tuple[int, int]:
    """与 matmul_kernel 开头完全相同的映射：一维 pid → 输出块坐标 (pid_m, pid_n)。"""
    num_pid_in_group = group_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * group_m
    group_size_m = min(num_pid_m - first_pid_m, group_m)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


def panels_touched(num_pid_m: int, num_pid_n: int, group_m: int, wave: int) -> int:
    """前 ``wave`` 个同时运行的 program 一共要读多少个 A 行块与 B 列块。

    数值越小，同一批 program 在 L2 中的复用越多。``group_m=1`` 即行优先顺序。
    """
    tiles = [program_to_tile(p, num_pid_m, num_pid_n, group_m) for p in range(wave)]
    return len({m for m, _ in tiles}) + len({n for _, n in tiles})


# endregion grouped_order


def matmul_ref(
    a: torch.Tensor, b: torch.Tensor, bm: int = 16, bn: int = 16, bk: int = 16, group_m: int = 2
) -> torch.Tensor:
    """逐个 program 执行 matmul_kernel 的分块循环（FP32 累加），用来验证映射与掩码。"""
    m, k = a.shape
    n = b.shape[1]
    num_pid_m, num_pid_n = -(-m // bm), -(-n // bn)
    c = torch.zeros(m, n, dtype=a.dtype)
    seen = set()
    for pid in range(num_pid_m * num_pid_n):
        pid_m, pid_n = program_to_tile(pid, num_pid_m, num_pid_n, group_m)
        seen.add((pid_m, pid_n))
        rm = slice(pid_m * bm, min((pid_m + 1) * bm, m))
        rn = slice(pid_n * bn, min((pid_n + 1) * bn, n))
        acc = torch.zeros(rm.stop - rm.start, rn.stop - rn.start, dtype=torch.float32)
        for k0 in range(0, k, bk):
            acc += a[rm, k0 : k0 + bk].float() @ b[k0 : k0 + bk, rn].float()
        c[rm, rn] = acc.to(a.dtype)
    assert len(seen) == num_pid_m * num_pid_n, "映射必须覆盖每个输出块恰好一次"
    return c
