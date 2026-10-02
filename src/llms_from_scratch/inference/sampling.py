"""Finite filtering and reproducible sampling using the canonical model/cache API."""

import math

import torch


def distribution(
    logits, temperature=1.0, top_k=None, top_p=1.0, min_p=0.0, repetition_penalty=1.0, history=()
):
    if logits.ndim != 1 or logits.numel() == 0 or not torch.isfinite(logits).all():
        raise ValueError("expected nonempty finite one-dimensional logits")
    if (
        any(not math.isfinite(value) for value in (temperature, top_p, min_p, repetition_penalty))
        or temperature <= 0
        or not 0 < top_p <= 1
        or not 0 <= min_p <= 1
        or repetition_penalty < 1
    ):
        raise ValueError("temperature > 0, 0 < top_p <= 1, 0 <= min_p <= 1, penalty >= 1 required")
    history = tuple(history)
    if any(type(token) is not int or not 0 <= token < logits.numel() for token in history):
        raise ValueError("history IDs must be integers within the vocabulary")
    scores = logits.clone()
    # CTRL-style penalty on raw logits: divide positive, multiply negative, once per seen ID.
    seen = torch.tensor(sorted(set(history)), dtype=torch.long, device=logits.device)
    if seen.numel():
        picked = scores[seen]
        scores[seen] = torch.where(
            picked > 0, picked / repetition_penalty, picked * repetition_penalty
        )
    scores = scores / temperature
    if top_k is not None:
        if type(top_k) is not int or not 1 <= top_k <= scores.numel():
            raise ValueError("top_k must be an integer within the vocabulary")
        order = scores.argsort(descending=True, stable=True)
        allowed = torch.zeros_like(scores, dtype=torch.bool)
        allowed[order[:top_k]] = True
        scores = scores.masked_fill(~allowed, -torch.inf)
    order = scores.argsort(descending=True, stable=True)
    ordered_probs = scores[order].softmax(0)
    # Include the token crossing p; every nonempty distribution keeps its maximum.
    excluded = ordered_probs.cumsum(0) - ordered_probs >= top_p
    scores[order[excluded]] = -torch.inf
    # min-p: keep tokens whose probability is at least min_p times the current maximum.
    probs = scores.softmax(0)
    kept = probs.masked_fill(probs < min_p * probs.max(), 0.0)
    return kept / kept.sum()


@torch.no_grad()
def sample_generate(
    model,
    prompt,
    max_new_tokens,
    eos_id=None,
    generator=None,
    temperature=1.0,
    top_k=None,
    top_p=1.0,
    min_p=0.0,
    repetition_penalty=1.0,
    cached=True,
):
    """Sample one unpadded prompt; cache/fresh-prefix paths use identical distributions."""
    if prompt.ndim != 2 or prompt.shape[0] != 1 or not prompt.shape[1]:
        raise ValueError("sampling requires one nonempty prompt")
    if (
        type(max_new_tokens) is not int
        or max_new_tokens < 0
        or prompt.shape[1] + max_new_tokens > model.config.max_length
    ):
        raise ValueError("requested generation must fit context")
    if eos_id is not None and (
        type(eos_id) is not int or not 0 <= eos_id < model.config.vocab_size
    ):
        raise ValueError("EOS must belong to the vocabulary")
    was_training = model.training
    model.eval()
    result, cache = prompt.clone(), None
    try:
        for _ in range(max_new_tokens):
            current = result[:, -1:] if cached and cache is not None else result
            output = model(current, cache=cache, use_cache=cached)
            cache = output.cache
            probabilities = distribution(
                output.logits[0, -1],
                temperature,
                top_k,
                top_p,
                min_p,
                repetition_penalty,
                result[0].tolist(),
            )
            token = torch.multinomial(probabilities, 1, generator=generator).reshape(1, 1)
            result = torch.cat((result, token), 1)
            if token.item() == eos_id:
                break
    finally:
        model.train(was_training)
    return result
