"""流水线并行：调度表生成、时间线模拟，以及按调度表执行的流水线（单进程模拟与多进程两种）。

记号：p 个 stage、m 个微批。一个动作是 (kind, mb)：
- "F"：前向；
- "B"：反向。在不拆分的调度（GPipe、1F1B）中 B 是完整反向；在零气泡调度中 B 只算输入梯度，
  权重梯度留给单独的 "W" 动作。
调度表 schedule[s] 是 stage s 依次执行的动作列表。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn

from llms_from_scratch.transformer.model import GPT

Action = tuple[str, int]
Schedule = list[list[Action]]

# region schedules


def gpipe_schedule(p: int, m: int) -> Schedule:
    """GPipe：先做完所有微批的前向，再按逆序做完所有反向。"""
    return [[("F", i) for i in range(m)] + [("B", i) for i in reversed(range(m))]
            for _ in range(p)]


def one_f_one_b_schedule(p: int, m: int) -> Schedule:
    """1F1B（PipeDream-Flush）：stage s 先做 p-s-1 个预热前向，然后一前一后交替，最后排空反向。"""
    schedule = []
    for s in range(p):
        warmup = min(p - s - 1, m)
        actions = [("F", i) for i in range(warmup)]
        for i in range(m - warmup):
            actions += [("F", warmup + i), ("B", i)]
        actions += [("B", i) for i in range(m - warmup, m)]
        schedule.append(actions)
    return schedule


def zero_bubble_h1_schedule(p: int, m: int) -> Schedule:
    """ZB-H1 式调度：反向拆成 B（输入梯度）与 W（权重梯度）。

    骨架与 1F1B 相同（p-s 个前向后开始一前一后），B 一就绪就做，好让上游尽早拿到梯度；
    stage s 把自己的 W 推迟 s 个位置，留到排空阶段去填补本来空闲的时间。
    推迟的 W 让 stage s 多保存 s 份激活，峰值仍不超过 stage 0 的 p 份，与 1F1B 相同。
    """
    schedule = []
    for s in range(p):
        warmup = min(p - s, m)
        actions = [("F", i) for i in range(warmup)]
        next_f, next_w = warmup, 0
        for i in range(m):
            actions.append(("B", i))
            if i >= s:
                actions.append(("W", next_w))
                next_w += 1
            if next_f < m:
                actions.append(("F", next_f))
                next_f += 1
        actions += [("W", j) for j in range(next_w, m)]
        schedule.append(actions)
    return schedule

# endregion


# region simulate


@dataclass(frozen=True)
class Event:
    stage: int
    kind: str
    mb: int
    start: float
    end: float


def _deps(kind: str, s: int, mb: int, p: int) -> list[tuple[str, int, int]]:
    if kind == "F":
        return [("F", s - 1, mb)] if s > 0 else []
    if kind == "B":
        return [("F", s, mb)] + ([("B", s + 1, mb)] if s < p - 1 else [])
    return [("B", s, mb)]  # W 只依赖本 stage 的 B


def simulate(schedule: Schedule, cost: dict[str, float]) -> list[Event]:
    """按调度表推演时间线：每个动作在“本 stage 空闲”且“依赖都已完成”时开始（忽略通信时间）。"""
    p = len(schedule)
    end: dict[tuple[str, int, int], float] = {}
    free, pos = [0.0] * p, [0] * p
    events: list[Event] = []
    total = sum(len(actions) for actions in schedule)
    while len(events) < total:
        progressed = False
        for s in range(p):
            while pos[s] < len(schedule[s]):
                kind, mb = schedule[s][pos[s]]
                deps = _deps(kind, s, mb, p)
                if any(d not in end for d in deps):
                    break  # 依赖还没被排上：先去推进别的 stage
                start = max([free[s], *(end[d] for d in deps)])
                end[(kind, s, mb)] = free[s] = start + cost[kind]
                events.append(Event(s, kind, mb, start, start + cost[kind]))
                pos[s] += 1
                progressed = True
        if not progressed:
            raise RuntimeError("调度表存在循环等待（死锁）")
    return events


def bubble_fraction(events: list[Event]) -> float:
    """所有 stage 的空闲时间占 p × 总时长 的比例。"""
    makespan = max(e.end for e in events)
    p = 1 + max(e.stage for e in events)
    busy = sum(e.end - e.start for e in events)
    return 1 - busy / (p * makespan)


def peak_activations(actions: list[Action]) -> int:
    """一个 stage 同时保存的微批激活数的峰值：前向后保存，完整反向（或 W）之后释放。"""
    release = "W" if any(kind == "W" for kind, _ in actions) else "B"
    live = peak = 0
    for kind, _ in actions:
        live += (kind == "F") - (kind == release)
        peak = max(peak, live)
    return peak

# endregion


# region stages


class GPTStage(nn.Module):
    """GPT 的一段连续层。第一段含词嵌入，最后一段含最终归一化与输出层。"""

    def __init__(self, model: GPT, start: int, end: int, first: bool, last: bool) -> None:
        super().__init__()
        self.embed = model.embed if first else None
        self.blocks = nn.ModuleList(model.blocks[start:end])
        self.norm = model.norm if last else None
        self.lm_head = model.lm_head if last else None
        self.register_buffer("freqs", model.freqs, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embed is not None:
            x = self.embed(x)
        freqs = self.freqs[: x.shape[1]]
        for block in self.blocks:
            x = block(x, freqs)
        if self.lm_head is not None:
            x = self.lm_head(self.norm(x))
        return x


def split_gpt(model: GPT, p: int) -> list[GPTStage]:
    """把 GPT 的层均匀切成 p 段。要求不共享词嵌入与输出层（共享时首末 stage 还需同步这份梯度）。"""
    if model.config.tie_embeddings and p > 1:
        raise ValueError("流水线切分要求 tie_embeddings=False")
    n = model.config.n_layers
    if n % p:
        raise ValueError("层数必须能被 stage 数整除")
    per = n // p
    return [GPTStage(model, s * per, (s + 1) * per, s == 0, s == p - 1) for s in range(p)]

# endregion


# region local_executor

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def run_pipeline_local(stages: list[nn.Module], schedule: Schedule, inputs: list[torch.Tensor],
                       targets: list[torch.Tensor], loss_fn: LossFn) -> torch.Tensor:
    """在单进程里按调度表执行流水线，返回平均损失；梯度累积在各 stage 的参数上。

    动作按模拟时间线的开始时间排序执行，与真实多卡执行的依赖顺序一致。
    B 若伴随 W（零气泡调度），B 只用 autograd.grad 求输入梯度，W 再求权重梯度。
    """
    p, m = len(stages), len(inputs)
    split = any(kind == "W" for actions in schedule for kind, _ in actions)
    order = sorted(simulate(schedule, {"F": 1.0, "B": 1.0, "W": 1.0}),
                   key=lambda e: (e.start, e.stage))
    saved_in: dict[tuple[int, int], torch.Tensor] = {}
    saved_out: dict[tuple[int, int], torch.Tensor] = {}
    grad_in: dict[tuple[int, int], torch.Tensor | None] = {}  # stage s 收到的“输出梯度”
    total = torch.zeros(())
    for e in order:
        s, mb = e.stage, e.mb
        params = [q for q in stages[s].parameters() if q.requires_grad]
        if e.kind == "F":
            if s == 0:
                x = inputs[mb]
            else:  # 跨 stage 边界：上游输出被“发送”过来，作为本 stage 的叶子张量
                x = saved_out[(s - 1, mb)].detach().requires_grad_()
            out = stages[s](x)
            if s == p - 1:
                out = loss_fn(out, targets[mb]) / m  # 每个微批的损失按 1/m 缩放
                total = total + out.detach()
            saved_in[(s, mb)], saved_out[(s, mb)] = x, out
        elif e.kind == "B":
            out, x = saved_out[(s, mb)], saved_in[(s, mb)]
            grad_out = None if s == p - 1 else grad_in[(s, mb)]
            if not split:
                torch.autograd.backward(out, grad_out)
                if s > 0:
                    grad_in[(s - 1, mb)] = x.grad
            elif s > 0:
                (gx,) = torch.autograd.grad(out, x, grad_out, retain_graph=True)
                grad_in[(s - 1, mb)] = gx
            if not split:
                del saved_out[(s, mb)], saved_in[(s, mb)]
        else:  # W：权重梯度，可以推迟到任何时候
            out = saved_out.pop((s, mb))
            saved_in.pop((s, mb))
            grad_out = None if s == p - 1 else grad_in[(s, mb)]
            grads = torch.autograd.grad(out, params, grad_out)
            for q, g in zip(params, grads, strict=True):
                q.grad = g if q.grad is None else q.grad + g
    return total

# endregion


# region dist_executor


def run_pipeline_stage(stage: nn.Module, actions: list[Action], inputs: list[torch.Tensor],
                       targets: list[torch.Tensor], loss_fn: LossFn,
                       activation_shape: tuple[int, ...],
                       dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """多进程流水线中本 rank（= stage）的执行循环：前向收上游激活、发给下游；反向反之。

    发送一律用 isend（不阻塞），接收用阻塞的 recv。只要调度表的依赖无环，就不会死锁。
    tag 区分微批与方向：激活用 2·mb，梯度用 2·mb+1。
    """
    s, p = dist.get_rank(), dist.get_world_size()
    m = len(inputs)
    saved: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    pending: list[dist.Work] = []
    total = torch.zeros(())
    for kind, mb in actions:
        if kind == "F":
            if s == 0:
                x = inputs[mb]
            else:
                x = torch.empty(activation_shape, dtype=dtype)
                dist.recv(x, src=s - 1, tag=2 * mb)
                x.requires_grad_()
            out = stage(x)
            if s == p - 1:
                out = loss_fn(out, targets[mb]) / m
                total = total + out.detach()
            else:
                pending.append(dist.isend(out.detach().contiguous(), dst=s + 1, tag=2 * mb))
            saved[mb] = (x, out)
        else:
            x, out = saved.pop(mb)
            if s == p - 1:
                out.backward()
            else:
                grad = torch.empty(activation_shape, dtype=dtype)
                dist.recv(grad, src=s + 1, tag=2 * mb + 1)
                out.backward(grad)
            if s > 0:
                pending.append(dist.isend(x.grad.contiguous(), dst=s - 1, tag=2 * mb + 1))
    for work in pending:
        work.wait()
    return total

# endregion
