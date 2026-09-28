"""Check S2's H100 SXM roofline arithmetic; this is not a GPU benchmark."""


def roofline_time(flops, hbm_bytes, peak_flops_s, peak_bytes_s):
    if min(flops, hbm_bytes, peak_flops_s, peak_bytes_s) <= 0:
        raise ValueError("FLOPs, bytes, and rates must be positive")
    return max(flops / peak_flops_s, hbm_bytes / peak_bytes_s)


def verify():
    layers, width, q_heads, kv_heads, head_dim = 32, 4096, 32, 8, 128
    ffn, vocab, context, element_bytes = 14336, 128256, 8192, 2
    blocks = layers * (
        2 * width * q_heads * head_dim
        + 2 * width * kv_heads * head_dim
        + 3 * width * ffn
    )
    head = width * vocab
    linear = blocks + head  # One output head projection; embedding is a lookup.
    kv_bytes = 2 * layers * kv_heads * head_dim * context * element_bytes

    bandwidth, dense_fp16_peak = 3.35e12, 1979e12 / 2
    ridge = dense_fp16_peak / bandwidth
    decode_linear_flops = 2 * linear
    decode_attention_flops = 4 * layers * width * context
    decode_flops = decode_linear_flops + decode_attention_flops
    decode_bytes = element_bytes * linear + kv_bytes
    prefill_linear_flops = 2 * context * blocks + 2 * head
    prefill_causal_attention_flops = 2 * layers * width * context * (context + 1)

    assert linear == 7504658432 and kv_bytes == 2**30
    assert 295 < ridge < 296
    assert 1.20 < decode_flops / decode_bytes < 1.21
    assert abs(roofline_time(decode_flops, decode_bytes, dense_fp16_peak, bandwidth) * 1000 - 4.800913) < 1e-5
    assert abs((prefill_linear_flops + prefill_causal_attention_flops) / dense_fp16_peak - 0.133345) < 1e-6
    print("S2 shape, units, decode bound, and prefill compute bound verified (no GPU measurement)")


if __name__ == "__main__":
    verify()
