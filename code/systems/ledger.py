"""Count persistent parameters separately from executed dense matrices."""
def ledger(layers, width, query_heads, kv_heads, head_dim, ffn, vocab, element_bytes):
    if any(value <= 0 for value in (layers, width, query_heads, kv_heads, head_dim, ffn, vocab, element_bytes)):
        raise ValueError("dimensions must be positive")
    if query_heads * head_dim != width or query_heads % kv_heads:
        raise ValueError("invalid attention dimensions")
    blocks = layers * (2 * width * query_heads * head_dim + 2 * width * kv_heads * head_dim + 3 * width * ffn)
    head = width * vocab
    total = blocks + 2 * head
    kv = 2 * layers * kv_heads * head_dim * element_bytes
    return {"blocks": blocks, "head": head, "total": total,
            "linear": blocks + head, "weight_bytes": total * element_bytes, "kv_token": kv}


def verify():
    toy = ledger(2, 256, 8, 2, 32, 512, 128, 2)
    assert toy["total"] == 1179648 and toy["kv_token"] == 512
    model = ledger(32, 4096, 32, 8, 128, 14336, 128256, 2)
    assert model["total"] == 8029995008 and model["linear"] == 7504658432
    assert model["kv_token"] * 8192 == 2 ** 30
    decode = 2 * model["linear"] * 16
    traffic = 2 * model["linear"] + 16 * 2 ** 30
    assert 7.45 < decode / traffic < 7.47
    prefill_last = 2 * 8192 * model["blocks"] + 2 * model["head"]
    prefill_all = 2 * 8192 * model["linear"]
    assert prefill_last < prefill_all
    bandwidth, peak = 3.35e12, 989.5e12
    ridge = peak / bandwidth
    assert 295 < ridge < 296
    assert min(peak, bandwidth * 4) == 13.4e12
    print("ledger: total vs linear, last-only head, KV, intensity and roofline verified")


if __name__ == "__main__":
    verify()
