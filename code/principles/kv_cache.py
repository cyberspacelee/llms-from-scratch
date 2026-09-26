"""Compare full causal attention with prefill and cached decoding."""

import torch

# Python puts this file's directory on sys.path, so the sibling module imports directly.
from position_encoding import apply_rope, attention


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


if __name__ == "__main__":
    verify()
