"""Check one five-token RoPE example and full/cached attention equivalence."""

import math

import torch

from examples.principles.position_encoding import apply_rope, attention
from llms_from_scratch import Transformer, basic_decoder_config


@torch.no_grad()
def cached_attention(q, k, v, chunks):
    """A single attention layer; unpadded batches, fixed RoPE frequencies."""
    if any(size <= 0 for size in chunks) or sum(chunks) != q.shape[-2]:
        raise ValueError("positive chunks must cover the sequence exactly")
    k_cache, v_cache, outputs = [], [], []
    offset = 0
    for size in chunks:
        stop = offset + size
        pos = torch.arange(offset, stop, device=q.device)
        qr = apply_rope(q[..., offset:stop, :], pos)
        # A key is rotated once at its absolute position, then retained.
        k_cache.append(apply_rope(k[..., offset:stop, :], pos))
        v_cache.append(v[..., offset:stop, :])
        keys, values = torch.cat(k_cache, dim=-2), torch.cat(v_cache, dim=-2)
        key_pos = torch.arange(stop, device=q.device)
        mask = key_pos[None, :] <= pos[:, None]  # [new queries, all keys]
        outputs.append(attention(qr, keys, values, mask))
        offset = stop
    return torch.cat(outputs, dim=-2)


def verify():
    # One head, one sequence: BOS/A/B are prefilled, C/D are appended.
    q = torch.zeros(1, 1, 5, 2, dtype=torch.float64)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    q[..., 0] = k[..., 0] = 1
    v[0, 0, :, 0] = torch.arange(5, dtype=torch.float64)
    pos = torch.arange(5)
    causal = pos[None, :] <= pos[:, None]
    full = attention(apply_rope(q, pos), apply_rope(k, pos), v, causal)
    scores_3 = torch.tensor([math.cos(3 - j) / math.sqrt(2) for j in range(4)], dtype=torch.float64)
    weights_3 = scores_3.softmax(0)
    expected_3 = (weights_3 * torch.arange(4)).sum()
    torch.testing.assert_close(full[0, 0, 3, 0], expected_3)
    torch.testing.assert_close(full, cached_attention(q, k, v, [3, 1, 1]))
    torch.testing.assert_close(full, cached_attention(q, k, v, [3, 2]))
    wrong_mask = torch.ones(2, 5, dtype=torch.bool).tril()
    wrong = attention(apply_rope(q[..., 3:, :], pos[3:]), apply_rope(k, pos), v, wrong_mask)
    assert not torch.allclose(wrong, full[..., 3:, :])
    once = apply_rope(k[..., 1:2, :], pos[1:2])
    assert not torch.allclose(apply_rope(once, pos[1:2]), once)
    print(
        f"PASS: five-token example, row 3 output={expected_3:.6f}; rectangular mask error detected"
    )

    torch.manual_seed(11)
    q, k, v = [torch.randn(2, 3, 9, 8, dtype=torch.float64) for _ in range(3)]
    pos = torch.arange(9)
    mask = pos[None, :] <= pos[:, None]
    full = attention(apply_rope(q, pos), apply_rope(k, pos), v, mask)
    for chunks in ([9], [1] * 9, [4, 1, 1, 1, 1, 1], [3, 2, 4]):
        cached = cached_attention(q, k, v, chunks)
        torch.testing.assert_close(cached, full, atol=1e-10, rtol=1e-10)
        print(f"PASS: chunks={chunks}, max_error={(cached - full).abs().max():.3e}")
    # Negative control: resetting a new query to zero changes its relative phases.
    good = attention(apply_rope(q[..., -1:, :], pos[-1:]), apply_rope(k, pos), v)
    bad = attention(apply_rope(q[..., -1:, :], pos[:1]), apply_rope(k, pos), v)
    assert not torch.allclose(good, bad)
    print("PASS: resetting the decode position is detected as incorrect")
    model = (
        Transformer(basic_decoder_config(13, dim=16, heads=4, ff_dim=32, layers=3, max_length=5))
        .double()
        .eval()
    )
    ids = torch.tensor([[0, 1, 2, 3, 4], [0, 2, 1, 4, 3]])
    with torch.no_grad():
        expected = model(ids).logits
        for chunks in ([5], [1] * 5, [3, 1, 1], [3, 2]):
            caches, outputs, offset = None, [], 0
            for size in chunks:
                result = model(ids[:, offset : offset + size], cache=caches, use_cache=True)
                logits, caches = result.logits, result.cache
                outputs.append(logits)
                offset += size
                assert all(layer.self_attention.length == offset for layer in caches.layers)
            torch.testing.assert_close(torch.cat(outputs, 1), expected, atol=1e-10, rtol=1e-10)
    print("PASS: complete three-layer Decoder cached logits equal full logits")

    # Two windowed layers: a retained upper-layer value can carry evicted input information.
    tokens = torch.tensor([0.0, 9.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)

    def window_mean(values):
        return torch.stack([values[max(0, i - 2) : i + 1].mean() for i in range(len(values))])

    full = window_mean(window_mean(tokens))
    input_cache, first_layer_cache, streamed = [], [], []
    for token in tokens:
        input_cache.append(token)
        first_layer = torch.stack(input_cache[-3:]).mean()
        first_layer_cache.append(first_layer)
        streamed.append(torch.stack(first_layer_cache[-3:]).mean())
    torch.testing.assert_close(torch.stack(streamed), full)
    torch.testing.assert_close(full[-1], torch.tensor(1.0, dtype=full.dtype))
    torch.testing.assert_close(
        window_mean(window_mean(tokens[-3:]))[-1], torch.tensor(0.0, dtype=full.dtype)
    )
    print(
        "PASS: two-layer window cache retains indirect history; truncated-prefix recompute differs"
    )


if __name__ == "__main__":
    verify()
