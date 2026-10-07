"""训练循环的部件：数据窗口、可恢复的采样器、手写 SGD 与梯度裁剪、学习率调度、检查点。"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset, Sampler


# region dataset
class TokenWindows(Dataset):
    """把一维 token 序列切成 (x, y) 窗口：y 是 x 向右错一位，长度都为 context_length。"""

    def __init__(self, tokens: torch.Tensor, context_length: int) -> None:
        if tokens.dim() != 1 or len(tokens) <= context_length:
            raise ValueError("需要长度大于 context_length 的一维 token 序列")
        self.tokens, self.context_length = tokens, context_length

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.tokens[i: i + self.context_length + 1]
        return chunk[:-1], chunk[1:]
# endregion dataset


# region sampler
class ResumableBatchSampler(Sampler[list[int]]):
    """每个 epoch 用 (seed, epoch) 生成一个确定的随机排列，按批产出下标。

    它自己记录“下一批从哪里开始”，state_dict 只有三个整数，因此可以精确恢复数据位置。
    注意：配合 DataLoader 使用时要求 num_workers=0；多进程预取会让采样器跑在训练前面。
    """

    def __init__(self, dataset_size: int, batch_size: int, seed: int = 0) -> None:
        self.n, self.batch_size, self.seed = dataset_size, batch_size, seed
        self.epoch, self.cursor = 0, 0  # cursor：本 epoch 已产出的样本数

    def _order(self) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed * 100_003 + self.epoch)
        return torch.randperm(self.n, generator=g)

    def __iter__(self) -> Iterator[list[int]]:
        while True:  # 无限流：跨 epoch 连续产出，训练循环按步数停止
            order = self._order()
            while self.cursor + self.batch_size <= self.n:  # 丢掉每个 epoch 末尾不满的一批
                batch = order[self.cursor: self.cursor + self.batch_size].tolist()
                self.cursor += self.batch_size
                yield batch
            self.epoch, self.cursor = self.epoch + 1, 0

    def state_dict(self) -> dict[str, int]:
        return {"seed": self.seed, "epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state: dict[str, int]) -> None:
        self.seed, self.epoch, self.cursor = state["seed"], state["epoch"], state["cursor"]
# endregion sampler


# region sgd
class SGDMomentum(torch.optim.Optimizer):
    """带动量与权重衰减的 SGD，更新式与 torch.optim.SGD 相同：

        g ← ∇θ + λθ
        v ← μ v + g          （第一步 v 初始化为 g）
        θ ← θ − η v
    """

    def __init__(self, params: Iterable[nn.Parameter], lr: float, momentum: float = 0.0,
                 weight_decay: float = 0.0) -> None:
        super().__init__(params, {"lr": lr, "momentum": momentum, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D401
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad if wd == 0 else p.grad + wd * p
                if mu != 0:
                    state = self.state[p]  # 优化器状态按参数存放，随 state_dict 保存
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = g.clone()
                    else:
                        state["momentum_buffer"].mul_(mu).add_(g)
                    g = state["momentum_buffer"]
                p.add_(g, alpha=-lr)
# endregion sgd


# region clip
def clip_grad_norm(params: Iterable[nn.Parameter], max_norm: float, eps: float = 1e-6) -> torch.Tensor:
    """把所有参数梯度拼成一个大向量，若其 L2 范数超过 max_norm，就整体按比例缩小。"""
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
    scale = (max_norm / (total + eps)).clamp(max=1.0)
    for g in grads:
        g.mul_(scale)
    return total
# endregion clip


# region schedule
def warmup_cosine(step: int, warmup: int, total: int, min_ratio: float = 0.1) -> float:
    """学习率倍率：前 warmup 步线性升到 1，之后余弦衰减到 min_ratio，total 步以后保持不变。"""
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
# endregion schedule


# region accumulate
def accumulated_step(model: nn.Module, loss_fn, micro_batches: list[tuple[torch.Tensor, torch.Tensor]],
                     optimizer: torch.optim.Optimizer, max_norm: float | None = None) -> float:
    """一次优化步 = 若干个微批的前向/反向 + 一次 step。

    每个微批的 loss 都是该微批内的平均；除以微批数 k 后再 backward，
    梯度在 .grad 中累加得到 (1/k)Σ∇L_i —— 恰好是整批平均损失的梯度（各微批等大时）。
    """
    optimizer.zero_grad(set_to_none=True)
    k, total = len(micro_batches), 0.0
    for x, y in micro_batches:
        loss = loss_fn(model(x), y) / k
        loss.backward()
        total += loss.item()
    if max_norm is not None:
        clip_grad_norm(model.parameters(), max_norm)
    optimizer.step()
    return total
# endregion accumulate


# region checkpoint
def rng_state() -> dict:
    state = {"torch": torch.get_rng_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, step: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    sampler: ResumableBatchSampler) -> None:
    """精确恢复训练需要的全部状态：少了任何一项，恢复后的轨迹都会偏离。"""
    torch.save({
        "step": step,
        "model": model.state_dict(),          # 参数与 buffer
        "optimizer": optimizer.state_dict(),  # 动量/Adam 矩估计 + 超参数
        "scheduler": scheduler.state_dict(),  # 调度器走到了第几步
        "sampler": sampler.state_dict(),      # 数据读到了哪里
        "rng": rng_state(),                   # dropout、采样等用到的随机数流
    }, path)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler,
                    sampler: ResumableBatchSampler) -> int:
    ckpt = torch.load(path, weights_only=False)  # RNG 状态含 Python 对象，只加载自己写的文件
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    sampler.load_state_dict(ckpt["sampler"])
    set_rng_state(ckpt["rng"])
    return ckpt["step"]
# endregion checkpoint


# region loop
def train(model: nn.Module, dataset: Dataset, optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.LRScheduler, sampler: ResumableBatchSampler,
          start_step: int, end_step: int, accum_steps: int = 1, max_norm: float | None = 1.0,
          checkpoint: tuple[int, str | Path] | None = None) -> list[float]:
    """从 start_step 训练到 end_step；checkpoint=(step, path) 时在该步结束后保存。"""
    # 给 DataLoader 一个独立的生成器：否则每次创建迭代器都会从全局 RNG 抽一个种子，
    # 恢复训练时多抽的这一次会让 dropout 的随机数流错位
    loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=0,
                                         generator=torch.Generator())
    batches = iter(loader)
    loss_fn = nn.functional.cross_entropy
    losses = []
    model.train()
    for step in range(start_step, end_step):
        micro = [next(batches) for _ in range(accum_steps)]
        loss = accumulated_step(
            model, lambda logits, y: loss_fn(logits.flatten(0, -2), y.flatten()),
            micro, optimizer, max_norm,
        )
        scheduler.step()
        losses.append(loss)
        if checkpoint is not None and step + 1 == checkpoint[0]:
            save_checkpoint(checkpoint[1], step + 1, model, optimizer, scheduler, sampler)
    return losses
# endregion loop
