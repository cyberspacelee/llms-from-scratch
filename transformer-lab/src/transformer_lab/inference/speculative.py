"""Greedy draft/target 推测解码，重点是接受、修正与位置对齐。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..models import Transformer


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

    Args:
        target: 用来验证并提交 token 的 Decoder-only 目标模型。
        draft: 与 target 共享词表的 Decoder-only 草稿模型。
        prompt: 无 padding long [1,S_prompt]，两个模型共享词表。
        max_new_tokens: 要生成的新 token 数，非负整数。
        draft_length: 每轮提议的最大 token 数，正整数。

    Returns:
        tuple: long tokens[1,S_prompt+max_new_tokens]、accepted/proposed/target_calls 计数字典。
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
            # prefix长度P：logits[:,P-1:P+K] -> expected[1,K+1]，最后一个为bonus。
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
