"""模型运行器：把调度结果变成打平的输入张量，执行前向（eager 或 CUDA Graph），取出需要的 logits。"""

from __future__ import annotations

import math

import torch

from llms_from_scratch.inference.cuda_graph import (
    CUDAGraphRunner,
    FakeGraphRunner,
    StaticInputs,
    capture_sizes,
)
from llms_from_scratch.inference.engine.paged_model import PagedGPT
from llms_from_scratch.inference.prefix_cache import KVCacheManager
from llms_from_scratch.inference.scheduler import ScheduledBatch
from llms_from_scratch.transformer.model import GPT


class ModelRunner:
    def __init__(self, model: GPT, num_blocks: int, block_size: int,
                 cuda_graph: str | None = None, max_graph_size: int = 16) -> None:
        """cuda_graph: None（eager）、"cuda"（真实 CUDA Graph，需要 GPU）或 "fake"（CPU 模拟）。"""
        self.paged = PagedGPT(model, num_blocks, block_size)
        self.block_size = block_size
        self.device = model.embed.weight.device
        self.max_blocks = math.ceil(model.config.context_length / block_size)
        self.graph = None
        if cuda_graph is not None:
            static = StaticInputs.allocate(max_graph_size, self.max_blocks, self.device)
            cls = CUDAGraphRunner if cuda_graph == "cuda" else FakeGraphRunner
            self.graph = cls(self.paged.forward, static, capture_sizes(max_graph_size))

    # region prepare
    def prepare_inputs(self, batch: ScheduledBatch, kv: KVCacheManager):
        """每个请求本步的 token 依次排开；同一请求的各行共享块表，可见长度 = 位置 + 1。"""
        ids, pos, slots, seq_lens, tables, logits_indices = [], [], [], [], [], []
        for req in batch.requests:
            n, start = batch.num_tokens[req.request_id], req.num_computed_tokens
            table = kv.block_ids(req.request_id)
            padded = table + [0] * (self.max_blocks - len(table))
            for p in range(start, start + n):
                ids.append(req.all_token_ids[p])
                pos.append(p)
                slots.append(table[p // self.block_size] * self.block_size + p % self.block_size)
                seq_lens.append(p + 1)
                tables.append(padded)
            logits_indices.append(len(ids) - 1)  # 只有每个请求最后一个 token 的 logits 有用
        t = lambda x: torch.tensor(x, dtype=torch.long, device=self.device)  # noqa: E731
        return (t(ids), t(pos), t(slots), t(seq_lens), t(tables)), t(logits_indices)
    # endregion prepare

    def execute(self, batch: ScheduledBatch, kv: KVCacheManager) -> torch.Tensor:
        inputs, logits_indices = self.prepare_inputs(batch, kv)
        is_decode = all(n == 1 for n in batch.num_tokens.values())
        if self.graph is not None and is_decode:  # 只有纯 decode 批走图（形状只由批大小决定）
            hidden = self.graph.run(*inputs)
        else:
            hidden = self.paged.forward(*inputs)
        return self.paged.compute_logits(hidden[logits_indices])
