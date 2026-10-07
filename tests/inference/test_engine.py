import pytest
import torch

from llms_from_scratch.inference.engine import EngineConfig, LLMEngine, SamplingParams
from llms_from_scratch.transformer.model import GPT, GPTConfig


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=97, context_length=64, d_model=64, n_layers=2, n_heads=4,
                       n_kv_heads=2)
    return GPT(config).eval()


def _prompts(seed=0, n=6, shared=None):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        length = int(torch.randint(3, 25, (1,), generator=g))
        tail = torch.randint(0, 97, (length,), generator=g).tolist()
        out.append((shared or []) + tail)
    return out


def _reference(model, prompts, max_tokens):
    outs = []
    for p in prompts:
        full = model.generate(torch.tensor([p]), max_tokens, temperature=0)
        outs.append(full[0, len(p):].tolist())
    return outs


@pytest.mark.parametrize("config", [
    EngineConfig(block_size=4, num_blocks=128),  # 默认：一次 prefill
    EngineConfig(block_size=4, num_blocks=128, max_num_batched_tokens=16),  # chunked prefill
    EngineConfig(block_size=8, num_blocks=128, long_prefill_token_threshold=5),
    EngineConfig(block_size=4, num_blocks=25, max_num_batched_tokens=32),  # 显存紧张：会抢占
    EngineConfig(block_size=4, num_blocks=128, cuda_graph="fake", max_graph_size=8),
])
def test_concurrent_greedy_matches_sequential_generate(model, config):
    prompts = _prompts()
    engine = LLMEngine(model, config)
    outs = engine.generate(prompts, SamplingParams(max_tokens=12))
    assert outs == _reference(model, prompts, 12)


def test_preemption_actually_happens(model):
    engine = LLMEngine(model, EngineConfig(block_size=4, num_blocks=25,
                                           max_num_batched_tokens=32))
    prompts = _prompts(seed=3)
    ids = [engine.add_request(p, SamplingParams(max_tokens=12)) for p in prompts]
    finished = []
    while engine.scheduler.has_unfinished():
        finished += engine.step()
    assert sum(r.num_preemptions for r in finished) > 0
    by_id = {r.request_id: r.output_token_ids for r in finished}
    assert [by_id[i] for i in ids] == _reference(model, prompts, 12)


def test_prefix_cache_hits_do_not_change_outputs(model):
    shared = list(range(10, 30))  # 20 个 token 的“系统提示”，块大小 4 → 5 个满块
    prompts = _prompts(seed=1, n=4, shared=shared)
    engine = LLMEngine(model, EngineConfig(block_size=4, num_blocks=128, max_num_seqs=1))
    ids = [engine.add_request(p, SamplingParams(max_tokens=10)) for p in prompts]
    finished = []
    while engine.scheduler.has_unfinished():
        finished += engine.step()
    by_id = {r.request_id: r for r in finished}
    hits = [by_id[i].num_cached_prompt_tokens for i in ids]
    assert hits[0] == 0 and all(h == 20 for h in hits[1:])
    assert [by_id[i].output_token_ids for i in ids] == _reference(model, prompts, 10)


def test_fake_graph_replays_padded_decode_batches(model):
    engine = LLMEngine(model, EngineConfig(block_size=4, cuda_graph="fake", max_graph_size=8))
    engine.generate(_prompts(n=3), SamplingParams(max_tokens=5))
    assert engine.runner.graph.replays and set(engine.runner.graph.replays) == {4}  # 3 → 4
    # 空块（块 0）只接收 padding 行的写入，不属于任何请求
    assert all(0 not in engine.kv.block_ids(r) for r in engine.kv.req_blocks)


def test_context_limit_matches_generate(model):
    prompt = list(range(60))
    engine = LLMEngine(model, EngineConfig(block_size=4))
    (out,) = engine.generate([prompt], SamplingParams(max_tokens=10))
    assert out == _reference(model, [prompt], 10)[0] and len(out) == 4


def test_seeded_sampling_matches_generate(model):
    prompt = [5, 6, 7, 8]
    engine = LLMEngine(model, EngineConfig(block_size=4))
    (out,) = engine.generate([prompt], SamplingParams(max_tokens=8, temperature=0.8, top_k=20,
                                                       seed=123))
    g = torch.Generator().manual_seed(123)
    ref = model.generate(torch.tensor([prompt]), 8, temperature=0.8, top_k=20, generator=g)
    assert out == ref[0, 4:].tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
def test_cuda_graph_engine_matches_eager_on_gpu(model):
    gpu_model = GPT(model.config).cuda().eval()
    gpu_model.load_state_dict(model.state_dict())
    prompts = _prompts()
    eager = LLMEngine(gpu_model, EngineConfig(block_size=4)).generate(prompts)
    graph = LLMEngine(gpu_model, EngineConfig(block_size=4, cuda_graph="cuda")).generate(prompts)
    assert eager == graph
