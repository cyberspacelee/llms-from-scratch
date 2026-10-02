"""序列轴压缩的因果参考：完整旧块摘要 + 精确近期 token。"""

from __future__ import annotations

import torch

from ..attention import scaled_dot_product_attention


def compress_sequence(
    key: torch.Tensor, value: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean pooling completed blocks; never include an unfinished/future block."""
    if key.ndim != 4 or value.ndim != 4 or key.shape[:3] != value.shape[:3] or block_size < 1:
        raise ValueError("expected aligned [B,H,S,D] and positive block size")
    count = key.shape[-2] // block_size

    def pool(x: torch.Tensor) -> torch.Tensor:
        return (
            x[..., : count * block_size, :]
            .reshape(*x.shape[:2], count, block_size, x.shape[-1])
            .mean(-2)
        )

    ends = torch.arange(count, device=key.device) * block_size + block_size - 1
    return pool(key), pool(value), ends


def compressed_causal_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 4,
    window: int = 8,
    query_offset: int = 0,
) -> torch.Tensor:
    """Exact recent K/V plus mean-pooled completed old blocks (an approximation).

    Partial blocks between old summaries and the local window remain exact.
    Absolute block ends prevent future information entering an old summary.
    """
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or q.shape[:2] != k.shape[:2]
        or k.shape[:3] != v.shape[:3]
        or q.shape[-1] != k.shape[-1]
    ):
        raise ValueError("incompatible attention tensors")
    if window < 1 or query_offset < 0 or query_offset + q.shape[-2] > k.shape[-2]:
        raise ValueError("invalid local window or query positions")
    pooled_k, pooled_v, ends = compress_sequence(k, v, block_size)
    output = []
    for row in range(q.shape[-2]):
        pos = query_offset + row
        local_start = max(0, pos - window + 1)
        selected = ends < local_start
        exact_start = int(selected.sum()) * block_size
        keys = torch.cat((pooled_k[:, :, selected], k[:, :, exact_start : pos + 1]), -2)
        values = torch.cat((pooled_v[:, :, selected], v[:, :, exact_start : pos + 1]), -2)
        output.append(scaled_dot_product_attention(q[:, :, row : row + 1], keys, values))
    return torch.cat(output, -2)
