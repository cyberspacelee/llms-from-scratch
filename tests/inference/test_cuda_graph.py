import pytest
import torch

from llms_from_scratch.inference.cuda_graph import (
    CUDAGraphRunner,
    FakeGraphRunner,
    StaticInputs,
    capture_sizes,
    count_aten_ops,
    kernel_ops,
    padded_size,
)
from llms_from_scratch.inference.engine.paged_model import PagedGPT
from llms_from_scratch.inference.paged_kv import slot_mapping
from llms_from_scratch.transformer.model import GPT, GPTConfig, KVCache


def test_capture_sizes_and_padding():
    sizes = capture_sizes(512)
    assert sizes[:6] == [1, 2, 4, 8, 16, 24] and sizes[-1] == 512
    assert len(sizes) == 3 + 31 + 17
    assert [padded_size(n, sizes) for n in (1, 3, 5, 8, 9, 250, 257)] == [1, 4, 8, 8, 16, 256, 272]
    assert padded_size(513, sizes) is None
    # 分桶带来的 padding 浪费不超过 1/2（小批）或 16 行（大批）
    for n in range(1, 513):
        p = padded_size(n, sizes)
        assert p - n <= max(n, 16)


def test_count_ops_of_one_decode_step():
    config = GPTConfig(vocab_size=50, context_length=32, d_model=32, n_layers=4, n_heads=4,
                       n_kv_heads=2)
    model = GPT(config).eval()
    cache = KVCache(config, 1)
    with torch.no_grad():
        model(torch.zeros(1, 8, dtype=torch.long), cache, 0)
        one = count_aten_ops(lambda: model(torch.zeros(1, 1, dtype=torch.long), cache, 8))
    two_layer = GPT(GPTConfig(vocab_size=50, context_length=32, d_model=32, n_layers=2,
                              n_heads=4, n_kv_heads=2)).eval()
    cache2 = KVCache(two_layer.config, 1)
    with torch.no_grad():
        two_layer(torch.zeros(1, 8, dtype=torch.long), cache2, 0)
        fewer = count_aten_ops(lambda: two_layer(torch.zeros(1, 1, dtype=torch.long), cache2, 8))
    per_layer = (len(one) - len(fewer)) / 2
    kernels = (len(kernel_ops(one)) - len(kernel_ops(fewer))) / 2
    assert per_layer > 60  # 一层 decode 就有几十次算子派发
    assert 25 <= kernels <= 45  # 去掉视图类算子后，约 35 个真正的计算 kernel


def _tiny():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=40, context_length=32, d_model=32, n_layers=2, n_heads=4,
                       n_kv_heads=2)
    return GPT(config).eval()


def _decode_inputs(n, block_size=4, max_blocks=8):
    """n 个请求，各自已有 5 个 token 的 KV，本步 decode 位置 5。"""
    tables = [[1 + 2 * i, 2 + 2 * i] for i in range(n)]
    ids = torch.arange(n) + 3
    pos = torch.full((n,), 5)
    slots = torch.tensor([slot_mapping(t, [5], block_size)[0] for t in tables])
    seq_lens = torch.full((n,), 6)
    bt = torch.tensor([t + [0] * (max_blocks - 2) for t in tables])
    return ids, pos, slots, seq_lens, bt


def test_fake_graph_matches_eager_and_padding_is_harmless():
    model = _tiny()
    eager = PagedGPT(model, num_blocks=16, block_size=4)
    graphed = PagedGPT(model, num_blocks=16, block_size=4)
    for c in (*eager.k_caches, *eager.v_caches):
        c.normal_()
    for dst, src in zip(graphed.k_caches + graphed.v_caches, eager.k_caches + eager.v_caches):
        dst.copy_(src)
    runner = FakeGraphRunner(graphed.forward, StaticInputs.allocate(8, 8), capture_sizes(8))
    inputs = _decode_inputs(3)
    out_graph = runner.run(*inputs)
    out_eager = eager.forward(*inputs)
    assert runner.replays == [4]  # 3 个请求被 pad 到捕获尺寸 4
    torch.testing.assert_close(out_graph, out_eager)
    # padding 行只写了空块 0；其余所有块与 eager 完全一致
    for a, b in zip(graphed.k_caches, eager.k_caches):
        torch.testing.assert_close(a[1:], b[1:])


def test_fake_graph_rejects_reallocated_buffers():
    model = _tiny()
    paged = PagedGPT(model, num_blocks=16, block_size=4)
    static = StaticInputs.allocate(4, 8)
    runner = FakeGraphRunner(paged.forward, static, [4])
    static.input_ids = torch.zeros(4, dtype=torch.long)  # 换了一块新内存
    with pytest.raises(RuntimeError):
        runner.run(*_decode_inputs(2))


def test_oversized_batch_falls_back_to_eager():
    model = _tiny()
    paged = PagedGPT(model, num_blocks=32, block_size=4)
    runner = FakeGraphRunner(paged.forward, StaticInputs.allocate(4, 8), [1, 2, 4])
    out = runner.run(*_decode_inputs(6))
    assert out.shape[0] == 6 and runner.replays == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
def test_real_cuda_graph_matches_eager():
    model = _tiny().cuda()
    paged = PagedGPT(model, num_blocks=16, block_size=4)
    static = StaticInputs.allocate(8, 8, device="cuda")
    runner = CUDAGraphRunner(paged.forward, static, capture_sizes(8))
    inputs = [t.cuda() for t in _decode_inputs(3)]
    out = runner.run(*inputs).clone()
    torch.testing.assert_close(out, paged.forward(*inputs))
