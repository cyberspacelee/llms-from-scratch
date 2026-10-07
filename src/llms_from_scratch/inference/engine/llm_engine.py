"""迷你推理引擎：调度器 + 分页 KV Cache（带前缀缓存）+ 模型运行器 + 采样器。

一次 step() 就是 vLLM EngineCore 忙循环的一次迭代：schedule → execute → sample → update。
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

from llms_from_scratch.inference.engine.model_runner import ModelRunner
from llms_from_scratch.inference.engine.sampler import SamplingParams, sample
from llms_from_scratch.inference.prefix_cache import KVCacheManager
from llms_from_scratch.inference.scheduler import Request, Scheduler, SchedulerConfig
from llms_from_scratch.transformer.model import GPT


@dataclass(eq=False)
class EngineRequest(Request):
    params: SamplingParams | None = None
    generator: torch.Generator | None = None


@dataclass
class EngineConfig:
    num_blocks: int = 256
    block_size: int = 16
    max_num_batched_tokens: int = 512
    max_num_seqs: int = 32
    long_prefill_token_threshold: int = 0
    enable_prefix_caching: bool = True
    cuda_graph: str | None = None  # None / "cuda" / "fake"
    max_graph_size: int = 16


class LLMEngine:
    def __init__(self, model: GPT, config: EngineConfig | None = None) -> None:
        self.config = config = config or EngineConfig()
        self.model = model.eval()
        self.kv = KVCacheManager(config.num_blocks, config.block_size, config.enable_prefix_caching)
        self.scheduler = Scheduler(SchedulerConfig(
            config.max_num_batched_tokens, config.max_num_seqs,
            config.long_prefill_token_threshold), self.kv)
        self.runner = ModelRunner(model, config.num_blocks, config.block_size,
                                  config.cuda_graph, config.max_graph_size)
        self._ids = itertools.count()
        self._clock = itertools.count()  # 用步数当作到达时间，保证 FCFS 顺序

    def add_request(self, prompt: list[int], params: SamplingParams | None = None) -> str:
        params = params or SamplingParams()
        room = self.model.config.context_length - len(prompt)
        if room <= 0:
            raise ValueError("提示已经占满 context_length")
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self.runner.device).manual_seed(params.seed)
        req = EngineRequest(f"req-{next(self._ids)}", list(prompt), min(params.max_tokens, room),
                            arrival_time=next(self._clock), params=params, generator=generator)
        self.scheduler.add_request(req)
        return req.request_id

    # region step
    def step(self) -> list[EngineRequest]:
        batch = self.scheduler.schedule()  # 1. 本步每个请求算多少 token、需要哪些块
        if not batch.requests:
            raise RuntimeError("KV 块不足以容纳任何请求")
        logits = self.runner.execute(batch, self.kv)  # 2. 一次前向，取每个请求最后一行
        tokens = sample(logits, [r.params for r in batch.requests],  # 3. 采样
                        [r.generator for r in batch.requests])
        sampled = {r.request_id: t for r, t in zip(batch.requests, tokens, strict=True)}
        return self.scheduler.update_from_output(batch, sampled)  # 4. 推进状态、释放结束的请求
    # endregion step

    def generate(self, prompts: list[list[int]],
                 params: SamplingParams | list[SamplingParams] | None = None) -> list[list[int]]:
        if not isinstance(params, list):
            params = [params] * len(prompts)
        ids = [self.add_request(p, sp) for p, sp in zip(prompts, params, strict=True)]
        done: dict[str, list[int]] = {}
        while self.scheduler.has_unfinished():
            for req in self.step():
                done[req.request_id] = req.output_token_ids
        return [done[i] for i in ids]
