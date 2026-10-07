import torch

from llms_from_scratch.architecture.window_cache import SinkRingCache, masked_attention, window_mask


def test_window_mask():
    m = window_mask(6, 3, n_sink=1)
    assert m[5].tolist() == [True, False, False, True, True, True]
    assert m[1].tolist() == [True, True, False, False, False, False]


def test_ring_cache_decode_matches_masked_attention():
    torch.manual_seed(0)
    T, d = 20, 8
    q, k, v = torch.randn(T, d), torch.randn(T, d), torch.randn(T, d)
    for n_sink in (0, 2):
        ref = masked_attention(q, k, v, window_mask(T, 5, n_sink))
        cache = SinkRingCache(window=5, dim=d, n_sink=n_sink)
        outs = []
        for t in range(T):
            cache.append(t, k[t], v[t])
            outs.append(cache.attend(q[t]))
        torch.testing.assert_close(torch.stack(outs), ref, atol=1e-5, rtol=1e-5)
        assert cache.k.shape[0] == 5 + n_sink  # 生成 20 个 token，缓存仍只有 W + sink 个槽位
