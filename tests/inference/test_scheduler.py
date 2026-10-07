import random

from llms_from_scratch.inference.prefix_cache import KVCacheManager
from llms_from_scratch.inference.scheduler import (
    Request,
    Scheduler,
    SchedulerConfig,
    Status,
    affine_step_time,
    simulate,
    simulate_static,
)


def _workload(n=40, seed=0, rate=50.0):
    rng = random.Random(seed)
    t, out = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate)
        prompt = [rng.randrange(1000) for _ in range(rng.randint(8, 120))]
        out.append(Request(f"r{i}", prompt, max_tokens=rng.randint(1, 60), arrival_time=t))
    return out


def test_budget_and_seq_limits_hold_every_step():
    config = SchedulerConfig(max_num_batched_tokens=64, max_num_seqs=8)
    result = simulate(_workload(), config, num_blocks=400, block_size=16,
                      step_time=affine_step_time(0.01, 1e-4))
    for _, _, alloc in result.steps:
        assert sum(alloc.values()) <= 64
        assert len(alloc) <= 8
    assert all(len(r.output_token_ids) == r.max_tokens for r in result.requests)


def test_chunked_prefill_runs_alongside_decode():
    kv = KVCacheManager(num_blocks=64, block_size=4)
    sched = Scheduler(SchedulerConfig(max_num_batched_tokens=32), kv)
    short = Request("short", [1, 2, 3], max_tokens=10)
    sched.add_request(short)
    batch = sched.schedule()
    sched.update_from_output(batch, {"short": 7})
    long = Request("long", list(range(100)), max_tokens=2)
    sched.add_request(long)
    chunks = []
    for _ in range(4):
        batch = sched.schedule()
        assert batch.num_tokens["short"] == 1  # 正在 decode 的请求每步都拿到 1 个 token
        chunks.append(batch.num_tokens["long"])
        sched.update_from_output(batch, {"short": 7, "long": 9})
    assert chunks == [31, 31, 31, 7]
    assert long.output_token_ids == [9]  # 只有最后一块的采样结果被采纳


def test_preemption_by_recompute_still_finishes_everything():
    reqs = [Request(f"r{i}", list(range(i, i + 20)), max_tokens=40) for i in range(6)]
    # 24 个可用块 × 4 槽：装不下 6 个 60-token 的序列（每个要 15 块）
    simulate(reqs, SchedulerConfig(max_num_batched_tokens=256), num_blocks=25,
             block_size=4, step_time=affine_step_time(1.0, 0.0))
    assert sum(r.num_preemptions for r in reqs) > 0
    assert all(r.status == Status.FINISHED and len(r.output_token_ids) == 40 for r in reqs)


def test_preemption_returns_blocks_to_pool():
    kv = KVCacheManager(num_blocks=9, block_size=4)  # 8 个可用块
    sched = Scheduler(SchedulerConfig(), kv)
    a, b = Request("a", list(range(16)), 50), Request("b", list(range(100, 116)), 50)
    sched.add_request(a)
    sched.add_request(b)
    batch = sched.schedule()
    sched.update_from_output(batch, {"a": 1, "b": 1})
    assert kv.num_free_blocks == 0  # 两个请求的 16 个提示 token 各占 4 块
    batch = sched.schedule()  # 都需要第 5 块：后到的 b 被抢占
    assert [r.request_id for r in batch.preempted] == ["b"]
    assert b.num_computed_tokens == 0 and sched.waiting == [b]
    assert batch.num_tokens == {"a": 1}


def test_priority_policy_admits_and_preempts_by_priority():
    kv = KVCacheManager(num_blocks=9, block_size=4)
    sched = Scheduler(SchedulerConfig(policy="priority"), kv)
    low = Request("low", list(range(16)), 50, arrival_time=0.0, priority=5)
    high = Request("high", list(range(100, 116)), 50, arrival_time=1.0, priority=0)
    sched.add_request(low)
    sched.add_request(high)
    batch = sched.schedule()
    assert [r.request_id for r in batch.requests] == ["high", "low"]
    sched.update_from_output(batch, {"low": 1, "high": 1})
    batch = sched.schedule()
    assert [r.request_id for r in batch.preempted] == ["low"]


def test_prefix_cache_reduces_prefill_tokens():
    system = list(range(64))
    reqs = [Request(f"r{i}", system + [1000 + i] * 5, max_tokens=3, arrival_time=i)
            for i in range(4)]
    result = simulate(reqs, SchedulerConfig(), num_blocks=100, block_size=16,
                      step_time=affine_step_time(0.1, 0.0), enable_prefix_caching=True)
    assert [r.num_cached_prompt_tokens for r in reqs] == [0, 64, 64, 64]
    prefill_tokens = [steps[2][r.request_id] for steps, r in zip(
        [result.steps[i * 3] for i in range(4)], reqs)]
    assert prefill_tokens == [69, 5, 5, 5]


def test_continuous_batching_beats_static_batching():
    step = affine_step_time(0.02, 2e-5)
    continuous = simulate(_workload(seed=1), SchedulerConfig(max_num_seqs=8), 2000, 16, step)
    static = simulate_static(_workload(seed=1), 8, step)
    assert continuous.throughput() > 1.3 * static.throughput()
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    assert mean(continuous.ttft()) < mean(static.ttft())
