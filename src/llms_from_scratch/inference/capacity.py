"""PD 分离部署的容量估算：给定模型、硬件、流量与 SLO，估算 prefill 与 decode 各需要多少 GPU。

模型很粗：roofline 时间乘以利用率，张量并行的通信用环形 all-reduce 公式，排队用利特尔法则。
它的用处是把各个量之间的关系说清楚，并给出数量级，而不是代替压测。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from llms_from_scratch.inference.cost_model import (
    HardwareSpec,
    ModelSpec,
    kv_bytes_per_token,
    step_cost,
)


@dataclass(frozen=True)
class Cluster:
    gpu: HardwareSpec
    link_bandwidth: float = 450e9  # 节点内每卡单向带宽，H100 NVLink4 合计 900 GB/s 双向
    link_latency: float = 5e-6  # 每次点对点传输的固定延迟
    network_bandwidth: float = 50e9  # 节点间每卡带宽（400 Gb/s InfiniBand ≈ 50 GB/s）
    mfu: float = 0.5  # prefill 能达到的算力利用率
    mbu: float = 0.7  # decode 能达到的带宽利用率
    memory_utilization: float = 0.9


@dataclass(frozen=True)
class Workload:
    request_rate: float  # 每秒到达的请求数
    prompt_len: int
    output_len: int
    prefix_hit_rate: float = 0.0  # 提示中命中前缀缓存、无需重算的比例


@dataclass(frozen=True)
class SLO:
    ttft: float
    tpot: float


# region allreduce
def allreduce_time(nbytes: float, n: int, bandwidth: float, latency: float) -> float:
    """环形 all-reduce：reduce-scatter + all-gather 各 n−1 步，每卡收发 2(n−1)/n 份数据。"""
    if n == 1:
        return 0.0
    return 2 * (n - 1) / n * nbytes / bandwidth + 2 * (n - 1) * latency


def tp_comm_time(model: ModelSpec, tokens: int, tp: int, cluster: Cluster,
                 act_bytes: int = 2) -> float:
    """Megatron 式张量并行：每层注意力与 FFN 之后各一次 all-reduce，数据量 tokens·d_model。"""
    per = allreduce_time(tokens * model.d_model * act_bytes, tp, cluster.link_bandwidth,
                         cluster.link_latency)
    return 2 * model.n_layers * per
# endregion allreduce


def _kv_shards(model: ModelSpec, tp: int) -> int:
    """KV 头按 TP 切分；tp 超过 KV 头数时多出来的卡只能复制 KV。"""
    return min(tp, model.n_kv_heads)


@dataclass(frozen=True)
class PhasePlan:
    tp: int
    per_instance: float  # prefill：token/s；decode：并发请求数（批大小）
    step_time: float  # prefill：单个提示的 TTFT 下界；decode：每步耗时
    instances: int

    @property
    def gpus(self) -> int:
        return self.tp * self.instances


# region prefill
def plan_prefill(model: ModelSpec, cluster: Cluster, work: Workload, slo: SLO, tp: int,
                 headroom: float = 0.7) -> PhasePlan | None:
    """prefill 受算力限制：实例吞吐 = tp 张卡的有效算力 / 每 token FLOPs。"""
    mem = cluster.gpu.mem_bytes * cluster.memory_utilization
    kv = work.prompt_len * kv_bytes_per_token(model) / _kv_shards(model, tp)
    if model.num_params() * 2 / tp + kv > mem:  # 权重加一个提示的 KV 放不下
        return None
    new_tokens = math.ceil(work.prompt_len * (1 - work.prefix_hit_rate))
    cached = work.prompt_len - new_tokens
    cost = step_cost(model, [(new_tokens, cached)])
    t = cost.flops / (tp * cluster.gpu.peak_flops * cluster.mfu) + tp_comm_time(
        model, new_tokens, tp, cluster)
    if t > slo.ttft:  # 连单个请求都满足不了 TTFT
        return None
    throughput = new_tokens / t
    # 留出余量（headroom < 1）以控制排队延迟
    instances = math.ceil(work.request_rate * new_tokens / (throughput * headroom))
    return PhasePlan(tp, throughput, t, max(1, instances))
# endregion prefill


# region decode
def decode_step(model: ModelSpec, cluster: Cluster, tp: int, batch: int, context: int) -> float:
    cost = step_cost(model, [(1, context)] * batch)
    kv = batch * (context + 2) * kv_bytes_per_token(model)  # 读 context+1 个位置，写 1 个
    weights = cost.bytes - kv
    per_gpu = weights / tp + kv / _kv_shards(model, tp)
    return per_gpu / (cluster.gpu.mem_bandwidth * cluster.mbu) + tp_comm_time(
        model, batch, tp, cluster)


def max_decode_batch(model: ModelSpec, cluster: Cluster, tp: int, context: int,
                     tpot: float) -> int:
    """同时满足显存与 TPOT 的最大批：显存给出上界，再二分查找满足步长约束的最大值。"""
    mem = cluster.gpu.mem_bytes * cluster.memory_utilization
    free = mem - model.num_params() * 2 / tp
    per_seq = context * kv_bytes_per_token(model) / _kv_shards(model, tp)
    hi = max(0, math.floor(free / per_seq))
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if decode_step(model, cluster, tp, mid, context) <= tpot:
            lo = mid
        else:
            hi = mid - 1
    return lo


def plan_decode(model: ModelSpec, cluster: Cluster, work: Workload, slo: SLO,
                tp: int) -> PhasePlan | None:
    context = work.prompt_len + work.output_len // 2  # 生成过程中的平均上下文长度
    batch = max_decode_batch(model, cluster, tp, context, slo.tpot)
    if batch == 0:
        return None
    step = decode_step(model, cluster, tp, batch, context)
    # 利特尔法则：并发数 = 到达率 × 每个请求在 decode 中停留的时间
    concurrency = work.request_rate * work.output_len * step
    return PhasePlan(tp, batch, step, max(1, math.ceil(concurrency / batch)))
# endregion decode


@dataclass(frozen=True)
class Plan:
    prefill: PhasePlan
    decode: PhasePlan
    kv_transfer_bytes_per_s: float  # prefill 集群向 decode 集群发送 KV 的总带宽
    kv_transfer_time: float  # 单个请求的 KV 经节点间网络传输的时间

    @property
    def gpus(self) -> int:
        return self.prefill.gpus + self.decode.gpus


# region plan
def plan_disaggregated(model: ModelSpec, cluster: Cluster, work: Workload, slo: SLO,
                       tp_options: tuple[int, ...] = (1, 2, 4, 8)) -> Plan:
    """为两个阶段分别挑选总 GPU 数最少的 TP 度——PD 分离的意义正在于两边可以不同。"""
    def best(fn):
        plans = [p for tp in tp_options if (p := fn(model, cluster, work, slo, tp)) is not None]
        if not plans:
            raise ValueError("没有任何 TP 配置能满足 SLO")
        return min(plans, key=lambda p: (p.gpus, p.tp))

    kv_per_req = work.prompt_len * kv_bytes_per_token(model)
    return Plan(best(plan_prefill), best(plan_decode), work.request_rate * kv_per_req,
                kv_per_req / cluster.network_bandwidth)
# endregion plan


# region ep
def experts_per_gpu(n_routed: int, n_redundant: int, ep: int) -> int:
    """大规模专家并行：路由专家加冗余副本均匀放到 ep 张卡上（DeepSeek-V3：256 + 32）。"""
    total = n_routed + n_redundant
    if total % ep:
        raise ValueError("专家数必须能被 EP 度整除")
    return total // ep
# endregion ep
