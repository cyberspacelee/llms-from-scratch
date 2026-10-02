"""21 · 2025–2026：DeepSeek DSA/CSA2 与 Qwen QSA 的 learned selection。

Indexer 在小维度评分，再 gather 正式 Attention；token/block selection 分开。
"""

from __future__ import annotations

import torch
from torch import nn

from ..attention import gathered_attention, scaled_dot_product_attention


def token_indices(scores: torch.Tensor, budget: int) -> tuple[torch.Tensor, torch.Tensor]:
    """输入 float scores[S,S] 与正整数预算；返回 indices/valid[S,M]，M=min(budget,S)。

    每行只选择因果可见 key；不足 M 个的位置用 False filler，indices 不越界。

    Args:
        scores: finite float [S,S]，query 行对 key 列的未遮挡评分。
        budget: 每个 query 的最大 key token 预算，正整数。

    Returns:
        tuple: long indices[S,M]、bool valid[S,M]。
    """
    if (
        scores.ndim != 2
        or scores.shape[0] != scores.shape[1]
        or scores.shape[0] < 1
        or not scores.is_floating_point()
        or not torch.isfinite(scores).all()
        or type(budget) is not int
        or budget < 1
    ):
        raise ValueError("expected finite square scores and positive budget")
    positions = torch.arange(scores.shape[0], device=scores.device)
    visible = positions[None, :] <= positions[:, None]  # [S_q,S_kv]
    indices = (
        scores.masked_fill(~visible, -torch.inf).topk(min(budget, scores.shape[0]), -1).indices
    )
    valid = indices <= positions[:, None]
    return indices, valid


def block_indices(
    scores: torch.Tensor, block_size: int, max_blocks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """输入 scores[S,S]、块宽 R、块预算；返回 indices/valid[S,max_blocks*R+R-1]。

    给完整因果块的平均分做 top-k，展开为连续 token；未完成的尾块保持精确。

    Args:
        scores: finite float [S,S]，query 行对 key 列的未遮挡评分。
        block_size: 块宽 R，正整数。
        max_blocks: 每个 query 最多选中的完整块数。

    Returns:
        tuple: long indices[S,max_blocks*R+R-1]、bool valid同shape。
    """
    token_indices(scores, 1)  # 复用输入边界检查
    if (
        type(block_size) is not int
        or block_size < 1
        or type(max_blocks) is not int
        or max_blocks < 1
    ):
        raise ValueError("block dimensions must be positive")
    width = max_blocks * block_size + block_size - 1
    indices = torch.zeros(scores.shape[0], width, device=scores.device, dtype=torch.long)
    valid = torch.zeros_like(indices, dtype=torch.bool)
    for row in range(scores.shape[0]):
        count = (row + 1) // block_size
        if count:
            block_scores = (
                scores[row, : count * block_size].reshape(count, block_size).mean(-1)
            )  # [N_blocks]
            selected = block_scores.topk(min(count, max_blocks)).indices
            tokens = (
                selected[:, None] * block_size + torch.arange(block_size, device=scores.device)
            ).flatten()
        else:
            tokens = indices.new_empty(0)
        tail = torch.arange(count * block_size, row + 1, device=scores.device)  # [N_tail<R]
        tokens = torch.cat((tokens, tail))
        indices[row, : tokens.numel()] = tokens
        valid[row, : tokens.numel()] = True
    return indices, valid


def run(device: torch.device) -> dict[str, object]:
    """输入设备；返回 learned token/block 选择、gather/dense 等价及因果性核对。

    Args:
        device: 执行 torch.device，默认由 CLI 选择 CPU。

    Returns:
        dict 本章 shape/成本/误差/不变量检查报告。
    """
    # ponytail: one sequence, shared indices across heads; batch-dependent gather needs a batched kernel.
    x = torch.randn(1, 7, 32, device=device, dtype=torch.float64)
    q_proj = nn.Linear(32, 8, bias=False, device=device, dtype=x.dtype)
    k_proj = nn.Linear(32, 4, bias=False, device=device, dtype=x.dtype)
    index_q = q_proj(x)[0].reshape(7, 2, 4).transpose(0, 1)  # [H_i,S,D_i]
    index_k = k_proj(x)[0]  # [S,D_i]，各 indexer query heads共享同一key
    scores = (index_q @ index_k.T).relu().sum(0)  # [S,S]，简化的 learned multihead score
    q, k, v = (torch.randn(1, 2, 7, 8, device=device, dtype=x.dtype) for _ in range(3))
    report = {}
    for name, (indices, valid) in {
        "token": token_indices(scores, 3),
        "block": block_indices(scores, 2, 1),
    }.items():
        positions = torch.arange(7, device=device)[:, None].expand_as(indices)
        assert (indices[valid] <= positions[valid]).all()
        dense_mask = torch.zeros(7, 7, device=device, dtype=torch.bool)
        dense_mask[positions[valid], indices[valid]] = True
        sparse = gathered_attention(q, k, v, indices, valid)
        dense = scaled_dot_product_attention(q, k, v, dense_mask)
        torch.testing.assert_close(sparse, dense)
        changed = v.clone()
        changed[:, :, 4:] += 100
        torch.testing.assert_close(
            sparse[:, :, :4], gathered_attention(q, k, changed, indices, valid)[:, :, :4]
        )
        report[name] = {
            "indices": indices.tolist(),
            "valid": valid.tolist(),
            "selection_shape": list(indices.shape),
            "output_shape": list(sparse.shape),
            "gather_dense_error": (sparse - dense).abs().max().item(),
        }
    return {
        "index_query_shape": list(index_q.shape),
        "score_shape": list(scores.shape),
        "selection": report,
        "note": "scoring is dense here; production indexer compression, distillation and candidate reuse are separate optimizations",
    }
