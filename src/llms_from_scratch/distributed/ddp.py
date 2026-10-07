"""数据并行：朴素 DDP 与基于反向 hook 的分桶 DDP。

两者的数学含义相同——每个 rank 用自己那份数据算出局部梯度，all-reduce 求平均后，
每个 rank 都持有全局平均梯度，再各自执行完全相同的优化器更新。区别只在通信的时机与粒度：
朴素版本在 backward 结束后逐个参数同步；分桶版本在 backward 进行中、一个桶的梯度
全部就绪就立即发起异步 all-reduce，让通信与剩余的反向计算重叠。
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


def broadcast_parameters(module: nn.Module, src: int = 0) -> None:
    """把 src 上的参数与 buffer 广播到所有 rank，保证训练从同一个起点出发。"""
    with torch.no_grad():
        for tensor in [*module.parameters(), *module.buffers()]:
            dist.broadcast(tensor, src=src)


# region naive


class NaiveDDP(nn.Module):
    """每个参数一次 all-reduce，在 backward 全部结束之后同步。"""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        broadcast_parameters(module)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def sync_gradients(self) -> None:
        world_size = dist.get_world_size()
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world_size

# endregion


# region bucketed


class BucketedDDP(nn.Module):
    """分桶 DDP：参数按“反向中梯度就绪的大致顺序”（即注册顺序的逆序）装桶。

    每个参数注册 post-accumulate-grad hook；当一个桶里所有参数的梯度都已累加完成，
    就把它们拼成一块连续缓冲区并发起异步 all-reduce。优化器 step 之前必须调用
    finish_gradient_synchronization()，等待所有通信完成并把平均梯度写回 .grad。
    """

    def __init__(self, module: nn.Module, bucket_size_mb: float = 25.0) -> None:
        super().__init__()
        self.module = module
        broadcast_parameters(module)
        limit = bucket_size_mb * 2**20
        params = [p for p in module.parameters() if p.requires_grad]
        self.buckets: list[list[nn.Parameter]] = [[]]
        size = 0
        for p in reversed(params):
            nbytes = p.numel() * p.element_size()
            if self.buckets[-1] and size + nbytes > limit:
                self.buckets.append([])
                size = 0
            self.buckets[-1].append(p)
            size += nbytes
        self._bucket_of = {p: i for i, bucket in enumerate(self.buckets) for p in bucket}
        self._ready = [0] * len(self.buckets)
        self._inflight: dict[int, tuple[dist.Work, torch.Tensor]] = {}
        self.launched_in_backward = 0  # 由 hook 在反向过程中发起的桶数，用于验证重叠
        for p in params:
            p.register_post_accumulate_grad_hook(self._on_grad_ready)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _launch(self, index: int) -> None:
        bucket = self.buckets[index]
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in bucket]
        flat = torch.cat([g.reshape(-1) for g in grads])
        flat /= dist.get_world_size()  # 先除后加，与“加完再除”数学上相同，低精度下更不易溢出
        self._inflight[index] = (dist.all_reduce(flat, async_op=True), flat)

    def _on_grad_ready(self, p: nn.Parameter) -> None:
        index = self._bucket_of[p]
        self._ready[index] += 1
        if self._ready[index] == len(self.buckets[index]):
            self.launched_in_backward += 1
            self._launch(index)  # 桶满即发：此时更靠前的层仍在做反向计算

    def finish_gradient_synchronization(self) -> None:
        for index in range(len(self.buckets)):
            if index not in self._inflight:  # 某些参数本步没有梯度，桶未满：现在补发
                self._launch(index)
        for index, (work, flat) in self._inflight.items():
            work.wait()
            offset = 0
            for p in self.buckets[index]:
                n = p.numel()
                synced = flat[offset:offset + n].view_as(p)
                if p.grad is None:
                    p.grad = synced.clone()
                else:
                    p.grad.copy_(synced)
                offset += n
        self._inflight.clear()
        self._ready = [0] * len(self.buckets)

# endregion
