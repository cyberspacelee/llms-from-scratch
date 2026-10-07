import pytest

from llms_from_scratch.inference.capacity import (
    SLO,
    Cluster,
    Workload,
    allreduce_time,
    decode_step,
    experts_per_gpu,
    max_decode_batch,
    plan_decode,
    plan_disaggregated,
    plan_prefill,
    tp_comm_time,
)
from llms_from_scratch.inference.cost_model import H100_SXM, LLAMA3_8B, LLAMA3_70B

CLUSTER = Cluster(H100_SXM)
WORK = Workload(request_rate=20, prompt_len=2048, output_len=256)
SLO_ = SLO(ttft=1.0, tpot=0.05)


def test_ring_allreduce_formula():
    # 4 卡、400 MB、带宽 100 GB/s、无延迟：2·3/4·4e8/1e11 = 6 ms
    assert allreduce_time(4e8, 4, 1e11, 0) == pytest.approx(6e-3)
    assert allreduce_time(4e8, 1, 1e11, 1e-6) == 0
    # 小消息由延迟主导：decode 一步的 TP 通信几乎与批大小无关
    small = tp_comm_time(LLAMA3_70B, 1, 8, CLUSTER)
    assert small == pytest.approx(2 * 80 * 2 * 7 * 5e-6, rel=0.01)


def test_deepseek_v3_expert_layout():
    # 官方概览：prefill EP32 每卡 9 个路由专家，decode EP144 每卡 2 个；均含 32 个冗余专家
    assert experts_per_gpu(256, 32, 32) == 9
    assert experts_per_gpu(256, 32, 144) == 2
    with pytest.raises(ValueError):
        experts_per_gpu(256, 0, 144)


def test_tensor_parallel_has_diminishing_returns_for_decode():
    steps = [decode_step(LLAMA3_70B, CLUSTER, tp, 32, 2048) for tp in (2, 4, 8)]
    assert steps[0] > steps[1] > steps[2]
    assert steps[0] / steps[1] < 2 and steps[1] / steps[2] < 2


def test_decode_batch_respects_tpot_and_memory():
    b_loose = max_decode_batch(LLAMA3_8B, CLUSTER, 1, 2048, tpot=1.0)
    b_tight = max_decode_batch(LLAMA3_8B, CLUSTER, 1, 2048, tpot=0.02)
    assert 0 < b_tight < b_loose
    assert decode_step(LLAMA3_8B, CLUSTER, 1, b_tight, 2048) <= 0.02
    assert decode_step(LLAMA3_8B, CLUSTER, 1, b_tight + 1, 2048) > 0.02
    # 宽松 SLO 下只受显存限制：(72 GB − 16 GB) / (2048 · 128 KiB) ≈ 208
    assert b_loose == pytest.approx(208, abs=2)
    assert max_decode_batch(LLAMA3_70B, CLUSTER, 1, 2048, 1.0) == 0  # 权重都放不下


def test_prefill_scales_with_rate_and_benefits_from_prefix_cache():
    a = plan_prefill(LLAMA3_70B, CLUSTER, WORK, SLO_, tp=4)
    b = plan_prefill(LLAMA3_70B, CLUSTER, Workload(40, 2048, 256), SLO_, tp=4)
    c = plan_prefill(LLAMA3_70B, CLUSTER, Workload(20, 2048, 256, prefix_hit_rate=0.5), SLO_, 4)
    assert b.instances >= 2 * a.instances - 1
    assert c.instances < a.instances
    assert plan_prefill(LLAMA3_70B, CLUSTER, WORK, SLO(ttft=0.05, tpot=1), tp=1) is None


def test_littles_law_in_decode_plan():
    p = plan_decode(LLAMA3_8B, CLUSTER, WORK, SLO_, tp=1)
    concurrency = WORK.request_rate * WORK.output_len * p.step_time
    assert p.instances * p.per_instance >= concurrency > (p.instances - 1) * p.per_instance


def test_disaggregated_plan_for_llama3_70b():
    plan = plan_disaggregated(LLAMA3_70B, CLUSTER, WORK, SLO_)
    assert plan.prefill.step_time <= SLO_.ttft and plan.decode.step_time <= SLO_.tpot
    assert plan.prefill.tp >= 2 and plan.decode.tp >= 2  # 70B 的 BF16 权重至少要两张卡
    assert plan_prefill(LLAMA3_70B, CLUSTER, WORK, SLO_, tp=1) is None
    # 每个请求 2048 · 320 KiB ≈ 671 MB 的 KV 要从 prefill 节点发往 decode 节点
    assert plan.kv_transfer_time == pytest.approx(2048 * 327680 / 50e9)
    assert plan.kv_transfer_bytes_per_s == pytest.approx(20 * 2048 * 327680)
