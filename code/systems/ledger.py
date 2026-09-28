"""Check a decoder's storage and main multiply-add counts without a GPU."""


def ledger(layers, width, query_heads, kv_heads, head_dim, ffn, vocab, element_bytes, tied=False):
    dims = (layers, width, query_heads, kv_heads, head_dim, ffn, vocab, element_bytes)
    if any(value <= 0 for value in dims):
        raise ValueError("dimensions must be positive")
    if query_heads * head_dim != width or query_heads % kv_heads:
        raise ValueError("invalid attention dimensions")

    blocks = layers * (2 * width * query_heads * head_dim
                       + 2 * width * kv_heads * head_dim + 3 * width * ffn)
    head = width * vocab
    norms = (2 * layers + 1) * width
    total = blocks + norms + (head if tied else 2 * head)
    return {
        "blocks": blocks,
        "head": head,
        "norms": norms,
        "total": total,
        "linear": blocks + head,
        "weight_bytes": total * element_bytes,
        "kv_token": 2 * layers * kv_heads * head_dim * element_bytes,
    }


def workload(shape, layers, query_heads, head_dim, prompt):
    if prompt <= 0:
        raise ValueError("prompt must be positive")
    pair_flops = 4 * layers * query_heads * head_dim
    return {
        "prefill_linear": 2 * (prompt * shape["blocks"] + shape["head"]),
        "prefill_attention": pair_flops * prompt * (prompt + 1) // 2,
        "prefill_kv": prompt * shape["kv_token"],
        "decode_linear": 2 * shape["linear"],
        "decode_attention": pair_flops * (prompt + 1),
        "decode_kv_read": (prompt + 1) * shape["kv_token"],
        "decode_kv_write": shape["kv_token"],
    }


def verify():
    toy = ledger(2, 4, 2, 1, 2, 8, 16, 2, tied=True)
    assert (toy["blocks"], toy["head"], toy["norms"]) == (288, 64, 20)
    assert (toy["total"], toy["weight_bytes"], toy["kv_token"]) == (372, 744, 16)
    assert ledger(2, 4, 2, 1, 2, 8, 16, 2)["total"] == 436
    step = workload(toy, 2, 2, 2, prompt=4)
    assert step == {
        "prefill_linear": 2432, "prefill_attention": 320, "prefill_kv": 64,
        "decode_linear": 704, "decode_attention": 160,
        "decode_kv_read": 80, "decode_kv_write": 16,
    }
    assert 2 * 4 * toy["linear"] == 2816  # All-position logits during training.
    assert (step["decode_linear"] + step["decode_attention"]) / (
        2 * toy["linear"] + step["decode_kv_read"] + step["decode_kv_write"]
    ) == 1.08

    llama3 = ledger(32, 4096, 32, 8, 128, 14336, 128256, 2)
    assert llama3["blocks"] == 6979321856
    assert llama3["total"] == 8030261248
    assert llama3["linear"] == 7504658432
    assert llama3["kv_token"] * 8192 == 2 ** 30
    long_step = workload(llama3, 32, 32, 128, prompt=8191)
    assert long_step["decode_attention"] == 2 ** 32
    assert long_step["decode_kv_read"] == 2 ** 30
    assert long_step["decode_attention"] / long_step["decode_kv_read"] == 4

    # S2 reuses these ideal H100 roofline inputs, not a device measurement.
    bandwidth, peak = 3.35e12, 989.5e12
    assert 295 < peak / bandwidth < 296
    assert min(peak, bandwidth * 4) == 13.4e12
    print("decoder storage, prefill/decode work and idealized intensity verified")


if __name__ == "__main__":
    verify()
