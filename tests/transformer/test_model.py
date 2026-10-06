import torch

from llms_from_scratch.transformer.model import GPT, GPTConfig, KVCache


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=50, context_length=32, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    model = GPT(config).eval()
    idx = torch.randint(0, 50, (2, 12))
    full = model(idx)
    cache = KVCache(config, 2)
    first = model(idx[:, :7], cache, 0)
    rest = [model(idx[:, t:t + 1], cache, t) for t in range(7, 12)]
    stepped = torch.cat([first, *rest], dim=1)
    torch.testing.assert_close(stepped, full, atol=1e-5, rtol=1e-4)


def test_generate_shape():
    model = GPT(GPTConfig(vocab_size=30, context_length=16, d_model=32, n_layers=1, n_heads=2))
    out = model.generate(torch.zeros(1, 3, dtype=torch.long), 5, temperature=0)
    assert out.shape == (1, 8)
