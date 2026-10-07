"""CUDA Graph：按批大小分桶捕获 decode 前向，重放时把输入拷进固定地址的静态缓冲区。

- CUDAGraphRunner 用 torch.cuda.CUDAGraph 真正捕获与重放（需要 GPU）。
- FakeGraphRunner 在 CPU 上模拟同样的契约：捕获时记下静态缓冲区的地址，重放时只允许
  通过这些缓冲区传入数据，并检查地址没有变化。它让分桶、padding 与静态地址的逻辑
  在没有 GPU 的机器上也能被测试。
"""

from __future__ import annotations

import bisect
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.utils._python_dispatch import TorchDispatchMode

# forward(input_ids, positions, slots, seq_lens, block_tables) -> [n, ...]
ForwardFn = Callable[..., torch.Tensor]


# region count
class _OpCounter(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func.overloadpacket))
        return func(*args, **(kwargs or {}))


def count_aten_ops(fn: Callable[[], object]) -> list[str]:
    """记录 fn 执行时派发到 ATen 的算子序列。在 GPU 上，每个计算算子大致对应一次 kernel 启动。"""
    with _OpCounter() as counter:
        fn()
    return counter.ops


# 只改变元数据（形状、步长）的算子不启动 kernel
VIEW_OPS = {"aten.view", "aten._unsafe_view", "aten.t", "aten.transpose", "aten.slice",
            "aten.unsqueeze", "aten.expand", "aten.view_as_complex", "aten.view_as_real"}


def kernel_ops(ops: list[str]) -> list[str]:
    return [op for op in ops if op not in VIEW_OPS]
# endregion count


def launch_bound_time(n_kernels: int, launch_us: float, gpu_us_per_kernel: float) -> float:
    """CPU 逐个发射 kernel 时一步的耗时（微秒）：每个 kernel 取 CPU 发射与 GPU 执行的较大者。"""
    return n_kernels * max(launch_us, gpu_us_per_kernel)


# region buckets
def capture_sizes(max_size: int) -> list[int]:
    """vLLM 默认的捕获尺寸：[1, 2, 4] + 8 的倍数到 255 + 16 的倍数到 max_size。"""
    sizes = [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_size + 1, 16))
    return [s for s in sizes if s <= max_size]


def padded_size(n: int, sizes: list[int]) -> int | None:
    """不小于 n 的最小捕获尺寸；n 超过最大尺寸时返回 None（回退到 eager 执行）。"""
    i = bisect.bisect_left(sizes, n)
    return sizes[i] if i < len(sizes) else None
# endregion buckets


# region static
@dataclass
class StaticInputs:
    """所有捕获共用的输入缓冲区。尺寸为 s 的图读的是每个缓冲区的前 s 行。"""

    input_ids: torch.Tensor
    positions: torch.Tensor
    slots: torch.Tensor
    seq_lens: torch.Tensor
    block_tables: torch.Tensor

    @classmethod
    def allocate(cls, max_size: int, max_blocks: int, device=None) -> StaticInputs:
        z = lambda *shape: torch.zeros(shape, dtype=torch.long, device=device)  # noqa: E731
        return cls(z(max_size), z(max_size), z(max_size), z(max_size), z(max_size, max_blocks))

    def views(self, size: int) -> tuple[torch.Tensor, ...]:
        return (self.input_ids[:size], self.positions[:size], self.slots[:size],
                self.seq_lens[:size], self.block_tables[:size])

    def stage(self, input_ids, positions, slots, seq_lens, block_tables, size: int) -> None:
        """把 n 个真实请求的数据拷进缓冲区前 n 行，第 n..size-1 行填成无害的 padding。

        padding 行：token 0、位置 0、写入槽 0（物理块 0 是永不分配的空块）、
        可见长度 1、块表全指向空块。它们会被计算，但结果被丢弃，也不会污染任何真实请求的 KV。
        """
        n = input_ids.shape[0]
        for buf, src in zip(self.views(size)[:4], (input_ids, positions, slots, seq_lens),
                            strict=True):
            buf[:n].copy_(src)
            buf[n:].zero_()
        self.seq_lens[n:size].fill_(1)
        self.block_tables[:size].zero_()
        self.block_tables[:n, :block_tables.shape[1]].copy_(block_tables)
# endregion static


class _GraphRunnerBase:
    def __init__(self, fn: ForwardFn, static: StaticInputs, sizes: list[int]) -> None:
        self.fn, self.static, self.sizes = fn, static, sorted(sizes)
        self.outputs: dict[int, torch.Tensor] = {}

    def run(self, input_ids, positions, slots, seq_lens, block_tables) -> torch.Tensor:
        n = input_ids.shape[0]
        size = padded_size(n, self.sizes)
        if size is None:  # 超过最大捕获尺寸：直接 eager
            return self.fn(input_ids, positions, slots, seq_lens, block_tables)
        self.static.stage(input_ids, positions, slots, seq_lens, block_tables, size)
        return self.replay(size)[:n]

    def replay(self, size: int) -> torch.Tensor:
        raise NotImplementedError


# region cuda
class CUDAGraphRunner(_GraphRunnerBase):
    """真正的 CUDA Graph。从大到小捕获，所有图共享一个显存池。"""

    def __init__(self, fn: ForwardFn, static: StaticInputs, sizes: list[int]) -> None:
        super().__init__(fn, static, sizes)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        pool = None
        for size in reversed(self.sizes):
            args = static.views(size)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):  # 预热：让 cuBLAS 等库完成懒初始化
                for _ in range(2):
                    fn(*args)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                self.outputs[size] = fn(*args)  # 输出张量的地址也被固定下来
            pool = graph.pool() if pool is None else pool
            self.graphs[size] = graph
        torch.cuda.synchronize()

    def replay(self, size: int) -> torch.Tensor:
        self.graphs[size].replay()  # 一次 cudaGraphLaunch 代替上百次 kernel 启动
        return self.outputs[size]
# endregion cuda


# region fake
class FakeGraphRunner(_GraphRunnerBase):
    """CPU 上的“伪图”：遵守与 CUDA Graph 相同的静态地址契约，用来测试逻辑。"""

    def __init__(self, fn: ForwardFn, static: StaticInputs, sizes: list[int]) -> None:
        super().__init__(fn, static, sizes)
        self.addresses = {s: [t.data_ptr() for t in static.views(s)] for s in self.sizes}
        self.replays: list[int] = []

    def replay(self, size: int) -> torch.Tensor:
        args = self.static.views(size)
        if [t.data_ptr() for t in args] != self.addresses[size]:
            raise RuntimeError("静态缓冲区的地址变了：真实的 CUDA Graph 会读到旧地址")
        self.replays.append(size)
        return self.fn(*args)
# endregion fake
