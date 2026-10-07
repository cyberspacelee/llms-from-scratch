"""集合通信：进程组启动工具、用点对点通信手写的环形 all-reduce，以及 α-β 成本模型。

本部分所有并行实现都在 CPU 上用 gloo 后端的多进程验证。`launch` 负责启动 world_size 个进程、
建立进程组、收集每个 rank 的返回值并在结束时销毁进程组；测试在父进程中把这些返回值与
单进程参考实现逐项比较。
"""

from __future__ import annotations

import math
import socket
import tempfile
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# region launch


def free_port() -> int:
    """向操作系统要一个当前空闲的 TCP 端口，作为本次进程组的 rendezvous 地址。"""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _entry(rank: int, fn: Callable[..., Any], world_size: int, port: int,
           out_dir: str, args: tuple) -> None:
    torch.set_num_threads(1)  # 每个进程一个线程：多个 rank 共享少量 CPU 核时更快也更稳定
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        result = fn(rank, world_size, *args)
        torch.save(result, Path(out_dir) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def launch(fn: Callable[..., Any], world_size: int, *args: Any) -> list[Any]:
    """在 world_size 个进程中运行 fn(rank, world_size, *args)，按 rank 顺序返回各进程的结果。

    fn 必须是模块顶层函数（spawn 按名字在子进程中重新导入它），返回值要能被 torch.save 序列化。
    """
    with tempfile.TemporaryDirectory() as out_dir:
        mp.spawn(_entry, args=(fn, world_size, free_port(), out_dir, args),
                 nprocs=world_size, join=True)
        return [torch.load(Path(out_dir) / f"rank{r}.pt", weights_only=False)
                for r in range(world_size)]

# endregion


def chunk_bounds(n: int, world_size: int) -> list[tuple[int, int]]:
    """把长度 n 的一维缓冲区切成 world_size 段（前几段多 1 个元素），返回每段的 [start, end)。"""
    base, extra = divmod(n, world_size)
    bounds, start = [], 0
    for r in range(world_size):
        end = start + base + (1 if r < extra else 0)
        bounds.append((start, end))
        start = end
    return bounds


# region ring


def _ring_step(send: torch.Tensor, recv: torch.Tensor, rank: int, world_size: int) -> None:
    """向右邻居发 send，同时从左邻居收 recv。先发后收：isend 不阻塞，所以环上不会死锁。"""
    request = dist.isend(send.contiguous(), dst=(rank + 1) % world_size)
    dist.recv(recv, src=(rank - 1) % world_size)
    request.wait()


def ring_reduce_scatter(flat: torch.Tensor) -> int:
    """环形 reduce-scatter（原地）：结束时本 rank 的第 rank 段等于所有 rank 该段之和。

    第 k 步（k = 0..p-2）发送第 (rank-k-1) mod p 段、接收第 (rank-k-2) mod p 段并累加。
    返回本 rank 发出的元素个数。
    """
    rank, p = dist.get_rank(), dist.get_world_size()
    bounds = chunk_bounds(flat.numel(), p)
    sent = 0
    for k in range(p - 1):
        s0, s1 = bounds[(rank - k - 1) % p]
        r0, r1 = bounds[(rank - k - 2) % p]
        buffer = torch.empty(r1 - r0, dtype=flat.dtype)
        _ring_step(flat[s0:s1], buffer, rank, p)
        flat[r0:r1] += buffer
        sent += s1 - s0
    return sent


def ring_all_gather(flat: torch.Tensor) -> int:
    """环形 all-gather（原地）：开始时本 rank 只有第 rank 段是有效的，结束时每段都有效。

    第 k 步把第 (rank-k) mod p 段发给右邻居，从左邻居收第 (rank-k-1) mod p 段。
    """
    rank, p = dist.get_rank(), dist.get_world_size()
    bounds = chunk_bounds(flat.numel(), p)
    sent = 0
    for k in range(p - 1):
        s0, s1 = bounds[(rank - k) % p]
        r0, r1 = bounds[(rank - k - 1) % p]
        buffer = torch.empty(r1 - r0, dtype=flat.dtype)
        _ring_step(flat[s0:s1], buffer, rank, p)
        flat[r0:r1] = buffer
        sent += s1 - s0
    return sent


def ring_all_reduce(tensor: torch.Tensor) -> int:
    """环形 all-reduce = 环形 reduce-scatter + 环形 all-gather，结果（求和）原地写回 tensor。

    返回本 rank 发出的元素个数；当 n 能被 p 整除时恰为 2(p-1)/p · n。
    """
    flat = tensor.view(-1)  # 要求 tensor 连续，flat 与 tensor 共享存储
    return ring_reduce_scatter(flat) + ring_all_gather(flat)

# endregion


# region cost


def ring_all_reduce_time(nbytes: float, p: int, alpha: float, bandwidth: float) -> float:
    """α-β 模型下环形 all-reduce 的时间：2(p-1) 步，每步发 n/p 字节。

    alpha 是每条消息的固定延迟（秒），bandwidth 是每个 rank 单向发送带宽（字节/秒）。
    """
    if p == 1:
        return 0.0
    return 2 * (p - 1) * alpha + 2 * (p - 1) / p * nbytes / bandwidth


def tree_all_reduce_time(nbytes: float, p: int, alpha: float, bandwidth: float) -> float:
    """流水化（双）二叉树 all-reduce 的近似时间：延迟项 2·log2(p)·α，带宽项约 2n/B。

    先沿树归约到根、再从根广播回叶子；把数据切成小块流水传输时，带宽项与 p 几乎无关。
    """
    if p == 1:
        return 0.0
    return 2 * math.ceil(math.log2(p)) * alpha + 2 * nbytes / bandwidth


def bus_bandwidth(nbytes: float, seconds: float, p: int, op: str = "all_reduce") -> float:
    """nccl-tests 的 busbw：把“数据量/时间”换算成每条链路实际承担的带宽，便于与硬件峰值比较。"""
    algbw = nbytes / seconds
    factor = {"all_reduce": 2 * (p - 1) / p, "reduce_scatter": (p - 1) / p,
              "all_gather": (p - 1) / p, "all_to_all": (p - 1) / p,
              "broadcast": 1.0, "reduce": 1.0}[op]
    return algbw * factor

# endregion

