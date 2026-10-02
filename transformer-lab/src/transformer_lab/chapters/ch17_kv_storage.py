"""17 · 2023–2024 推理支线：Paged KV、Prefix Sharing 与量化存储。"""

from __future__ import annotations

import torch

from ..cache import PagedKVCache, QuantizedTensor


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 page/COW shape、int8误差界和存储字节检查。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    key, value = (
        torch.randn(2, 2, 7, 8, device=device, dtype=torch.float64) for _ in range(2)
    )  # [B,H_kv,S_kv,D_h]
    pages = PagedKVCache(page_size=3)
    pages.append(key[:, :, :4], value[:, :, :4])
    fork = pages.fork()
    pages.append(key[:, :, 4:], value[:, :, 4:])
    assert fork.materialize().length == 4 and pages.materialize().length == 7
    assert pages.block_table[0] == fork.block_table[0]
    torch.testing.assert_close(pages.materialize().key, key)
    quantized = QuantizedTensor.encode(key)  # codes [B,H_kv,S_kv,D_h], scales [B,H_kv,S_kv,1]
    assert ((quantized.decode() - key).abs() <= quantized.scales / 2 + 1e-12).all()
    return {
        "input_shape": list(key.shape),
        "page_shapes": [list(k.shape) for k, _ in pages.pages],
        "fork_prefix_length": fork.materialize().length,
        "codes_shape": list(quantized.codes.shape),
        "scales_shape": list(quantized.scales.shape),
        "max_error": (quantized.decode() - key).abs().max().item(),
        "floating_bytes": key.numel() * key.element_size(),
        "quantized_bytes": quantized.nbytes,
        "note": "materialize copies the prefix; int8 storage is not a paged or quantized attention kernel",
    }
