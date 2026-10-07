"""迭代级调度器：连续批处理 + chunked prefill + token 预算 + 抢占（重计算）。

调度思路与 vLLM V1 相同：没有“prefill 阶段”和“decode 阶段”之分。每个请求只记录
num_computed_tokens（KV 已写入缓存的 token 数）和 num_tokens（提示 + 已生成），
每一步给请求分配若干 token，让前者追上后者。decode 是“差 1 个”，prefill 是“差一整段”，
chunked prefill 是“这一步只追一部分”，前缀缓存命中是“一开始就追上了一截”。
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, field

from llms_from_scratch.inference.cost_model import HardwareSpec, ModelSpec, step_cost
from llms_from_scratch.inference.prefix_cache import KVCacheManager


class Status(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


# region request
@dataclass(eq=False)
class Request:
    request_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    arrival_time: float = 0.0
    priority: int = 0  # 数值越小越优先（与 vLLM 一致）
    eos_token_id: int | None = None
    output_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    num_cached_prompt_tokens: int = 0  # 首次调度时前缀缓存命中的 token 数
    num_preemptions: int = 0
    status: Status = Status.WAITING
    first_token_time: float | None = None
    finish_time: float | None = None

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)
# endregion request


@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 2048  # 每步的 token 预算
    max_num_seqs: int = 256  # 同时运行的请求数上限
    long_prefill_token_threshold: int = 0  # >0 时单个请求每步最多拿这么多 token
    policy: str = "fcfs"  # "fcfs" 或 "priority"


@dataclass
class ScheduledBatch:
    requests: list[Request]
    num_tokens: dict[str, int]  # 本步给每个请求分配的 token 数
    preempted: list[Request]

    @property
    def total_tokens(self) -> int:
        return sum(self.num_tokens.values())


class Scheduler:
    def __init__(self, config: SchedulerConfig, kv: KVCacheManager) -> None:
        self.config = config
        self.kv = kv
        self.waiting: list[Request] = []
        self.running: list[Request] = []

    def add_request(self, request: Request) -> None:
        self.waiting.append(request)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    def _rank(self, r: Request) -> tuple:
        """排序键：越小越先服务。FCFS 只看到达时间，priority 先看优先级。"""
        if self.config.policy == "priority":
            return (r.priority, r.arrival_time)
        return (r.arrival_time,)

    def _cap(self, n: int, budget: int) -> int:
        limit = self.config.long_prefill_token_threshold
        return min(n, budget, limit) if limit > 0 else min(n, budget)

    # region schedule
    def schedule(self) -> ScheduledBatch:
        budget = self.config.max_num_batched_tokens
        scheduled: list[Request] = []
        num_tokens: dict[str, int] = {}
        preempted: list[Request] = []

        # 1. 先服务 RUNNING 请求（正在 decode 的，或 chunked prefill 做到一半的）
        i = 0
        while i < len(self.running) and budget > 0:
            req = self.running[i]
            n = self._cap(req.num_tokens - req.num_computed_tokens, budget)
            if n == 0:
                i += 1
                continue
            while not self.kv.allocate_slots(req.request_id, req.num_computed_tokens + n):
                # KV 块不够：抢占排序最靠后的请求（可能就是自己），释放它的块
                victim = max(reversed(self.running), key=self._rank)  # 并列时取最后加入的
                self.running.remove(victim)
                if victim.request_id in num_tokens:  # 它本步已被调度：退回预算
                    budget += num_tokens.pop(victim.request_id)
                    scheduled.remove(victim)
                self._preempt(victim)
                preempted.append(victim)
                if victim is req:
                    break
            else:
                scheduled.append(req)
                num_tokens[req.request_id] = n
                budget -= n
            i = self.running.index(req) + 1 if req in self.running else i

        # 2. 有余量且本步没有发生抢占时，按策略从 WAITING 中接纳新请求
        self.waiting.sort(key=self._rank)
        while (not preempted and self.waiting and budget > 0
               and len(self.running) < self.config.max_num_seqs):
            req = self.waiting[0]
            hits, n_hit = self.kv.get_computed_blocks(req.all_token_ids)
            n = self._cap(req.num_tokens - n_hit, budget)
            if not self.kv.allocate_slots(req.request_id, n_hit + n, hits):
                break  # 显存不够：队头请求接着等（不越过它调度后面的，避免饿死）
            self.waiting.pop(0)
            if req.status == Status.WAITING:
                req.num_cached_prompt_tokens = n_hit
            req.num_computed_tokens = n_hit
            req.status = Status.RUNNING
            self.running.append(req)
            scheduled.append(req)
            num_tokens[req.request_id] = n
            budget -= n
        return ScheduledBatch(scheduled, num_tokens, preempted)
    # endregion schedule

    def _preempt(self, req: Request) -> None:
        """重计算式抢占：丢掉全部 KV，回到等待队列队头；已生成的 token 保留，重新调度时一起 prefill。"""
        self.kv.free(req.request_id)
        req.num_computed_tokens = 0
        req.num_preemptions += 1
        req.status = Status.PREEMPTED
        self.waiting.insert(0, req)

    # region update
    def update_from_output(self, batch: ScheduledBatch, sampled: dict[str, int],
                           now: float = 0.0) -> list[Request]:
        """模型执行完一步后：推进 num_computed_tokens，登记满块，追加采样结果并判断结束。"""
        finished = []
        for req in batch.requests:
            req.num_computed_tokens += batch.num_tokens[req.request_id]
            self.kv.cache_blocks(req.request_id, req.all_token_ids, req.num_computed_tokens)
            if req.num_computed_tokens < req.num_tokens:
                continue  # chunked prefill 的中间块：采样结果没有意义，丢弃
            token = sampled[req.request_id]
            req.output_token_ids.append(token)
            if req.first_token_time is None:
                req.first_token_time = now
            if len(req.output_token_ids) >= req.max_tokens or token == req.eos_token_id:
                req.status = Status.FINISHED
                req.finish_time = now
                self.running.remove(req)
                self.kv.free(req.request_id)
                finished.append(req)
        return finished
    # endregion update


# ---------------------------------------------------------------- 离散事件模拟

def affine_step_time(fixed: float, per_token: float, per_seq: float = 0.0
                     ) -> Callable[[ScheduledBatch], float]:
    """一步的耗时 = 固定开销（读权重、启动 kernel）+ 与 token 数、序列数成正比的部分。"""
    return lambda b: fixed + per_token * b.total_tokens + per_seq * len(b.requests)


def roofline_step_time(model: ModelSpec, hw: HardwareSpec, mfu: float = 0.5, mbu: float = 0.7,
                       overhead: float = 0.0) -> Callable[[ScheduledBatch], float]:
    """用上一章的成本模型估计一步的耗时：每个请求贡献 (本步 token 数, 已缓存长度)。"""
    def time(b: ScheduledBatch) -> float:
        items = [(b.num_tokens[r.request_id], r.num_computed_tokens) for r in b.requests]
        return overhead + step_cost(model, items).time(hw, mfu=mfu, mbu=mbu)
    return time


@dataclass
class SimResult:
    requests: list[Request]
    steps: list[tuple[float, float, dict[str, int]]]  # (开始, 结束, 本步分配)
    makespan: float

    def ttft(self) -> list[float]:
        return [r.first_token_time - r.arrival_time for r in self.requests]

    def tpot(self) -> list[float]:
        return [(r.finish_time - r.first_token_time) / (len(r.output_token_ids) - 1)
                for r in self.requests if len(r.output_token_ids) > 1]

    def max_stall(self) -> float:
        """含 decode 的步里最长的一步：正在生成的用户会感到的最长停顿（最差 ITL）。"""
        return max((end - start for start, end, alloc in self.steps
                    if any(n == 1 for n in alloc.values())), default=0.0)

    def throughput(self) -> float:
        return sum(len(r.output_token_ids) for r in self.requests) / self.makespan

    def goodput(self, ttft_slo: float, tpot_slo: float) -> float:
        """每秒完成的、同时满足 TTFT 与 TPOT 两个 SLO 的请求数。"""
        ok = 0
        for r in self.requests:
            n = len(r.output_token_ids)
            tpot = (r.finish_time - r.first_token_time) / (n - 1) if n > 1 else 0.0
            ok += (r.first_token_time - r.arrival_time <= ttft_slo) and tpot <= tpot_slo
        return ok / self.makespan


# region simulate
def simulate(requests: list[Request], config: SchedulerConfig, num_blocks: int, block_size: int,
             step_time: Callable[[ScheduledBatch], float], next_token: int = 1,
             enable_prefix_caching: bool = False) -> SimResult:
    """用假模型（总是输出 next_token）驱动调度器，时间由 step_time 给出。"""
    sched = Scheduler(config, KVCacheManager(num_blocks, block_size, enable_prefix_caching))
    pending = sorted(requests, key=lambda r: r.arrival_time)
    now, steps = 0.0, []
    while pending or sched.has_unfinished():
        while pending and pending[0].arrival_time <= now:
            sched.add_request(pending.pop(0))
        if not sched.has_unfinished():
            now = pending[0].arrival_time  # 空闲：快进到下一个请求到达
            continue
        batch = sched.schedule()
        if not batch.requests:
            raise RuntimeError("调度器无法推进：KV 块不足以容纳任何请求")
        start, now = now, now + step_time(batch)
        steps.append((start, now, dict(batch.num_tokens)))
        sched.update_from_output(batch, {r.request_id: next_token for r in batch.requests}, now)
    return SimResult(requests, steps, now)
# endregion simulate


# region static
def simulate_static(requests: list[Request], batch_size: int,
                    step_time: Callable[[ScheduledBatch], float]) -> SimResult:
    """静态批处理：凑满一批（或队列已空）才开始，整批跑到最长的请求结束才接纳下一批。"""
    pending = sorted(requests, key=lambda r: r.arrival_time)
    now, steps = 0.0, []
    while pending:
        now = max(now, pending[min(batch_size, len(pending)) - 1].arrival_time)
        group, pending = pending[:batch_size], pending[batch_size:]
        # prefill：所有提示补齐到最长；decode：所有槽位一直占到最长的输出结束
        longest_prompt = max(len(r.prompt_token_ids) for r in group)
        prefill = ScheduledBatch(group, {r.request_id: longest_prompt for r in group}, [])
        start, now = now, now + step_time(prefill)
        steps.append((start, now, dict(prefill.num_tokens)))
        for r in group:
            r.output_token_ids, r.first_token_time = [1], now
        for _ in range(max(r.max_tokens for r in group) - 1):
            decode = ScheduledBatch(group, {r.request_id: 1 for r in group}, [])
            start, now = now, now + step_time(decode)
            steps.append((start, now, dict(decode.num_tokens)))
            for r in group:
                if len(r.output_token_ids) < r.max_tokens:
                    r.output_token_ids.append(1)
                    r.finish_time = now  # 生成完以后只是空转（被 padding 占着槽位）
        for r in group:
            r.finish_time = r.finish_time or now
            r.status = Status.FINISHED
    return SimResult(requests, steps, now)
# endregion static
