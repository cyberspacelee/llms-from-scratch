"""Independent principle experiments, not production serving kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Transformer
import torch

from .attention import scaled_dot_product_attention
from .cache import compress_sequence


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


@torch.no_grad()
def greedy_speculative_generate(
    target: Transformer,
    draft: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    draft_length: int = 4,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Greedy draft/target verification. Stochastic p/q acceptance is a different algorithm.

    Target verifies a proposal block in one forward; at the first mismatch,
    commit the target correction and discard all later draft tokens.
    """
    if target.config.architecture != "decoder" or draft.config.architecture != "decoder":
        raise ValueError("reference speculation supports decoder-only models")
    target.validate_ids(prompt, None)
    draft.validate_ids(prompt, None)
    if prompt.shape[0] != 1 or target.config.vocab_size != draft.config.vocab_size:
        raise ValueError("speculation requires one prompt and a shared vocabulary")
    if (
        type(max_new_tokens) is not int
        or max_new_tokens < 0
        or type(draft_length) is not int
        or draft_length < 1
    ):
        raise ValueError("invalid generation or draft length")
    if prompt.shape[1] + max_new_tokens > min(target.config.max_length, draft.config.max_length):
        raise ValueError("generation exceeds context")
    modes = target.training, draft.training
    target.eval()
    draft.eval()
    result = prompt.clone()
    accepted_total, proposed_total, target_calls = 0, 0, 0
    try:
        while result.shape[1] - prompt.shape[1] < max_new_tokens:
            remaining = max_new_tokens - (result.shape[1] - prompt.shape[1])
            count = min(draft_length, remaining)
            proposal = draft.generate(result, count)[:, result.shape[1] :]
            proposed_total += count
            # ponytail: full prefix recomputation exposes verification; KV rollback is the next speed upgrade.
            logits = target(torch.cat((result, proposal), 1)).logits
            expected = logits[:, result.shape[1] - 1 : result.shape[1] + count].argmax(-1)
            target_calls += 1
            accepted = 0
            while accepted < count and proposal[0, accepted] == expected[0, accepted]:
                accepted += 1
            accepted_total += accepted
            commit = proposal[:, :accepted]
            if accepted < remaining:
                commit = torch.cat((commit, expected[:, accepted : accepted + 1]), 1)
            result = torch.cat((result, commit), 1)
    finally:
        target.train(modes[0])
        draft.train(modes[1])
    return result, {
        "accepted": accepted_total,
        "proposed": proposed_total,
        "target_calls": target_calls,
    }
