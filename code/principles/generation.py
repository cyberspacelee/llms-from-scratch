"""Filtering, reproducible sampling, and decoder generation stopping checks."""

import torch

from decoder import Decoder


def distribution(logits, temperature=1., top_k=None, top_p=1., min_p=0.,
                 repetition_penalty=1., history=()):
    if logits.ndim != 1 or not torch.isfinite(logits).all():
        raise ValueError("expected finite one-dimensional logits")
    if temperature <= 0 or not 0 < top_p <= 1 or not 0 <= min_p <= 1 or repetition_penalty <= 0:
        raise ValueError("temperature > 0, 0 < top_p <= 1, 0 <= min_p <= 1, penalty > 0 required")
    scores = logits.clone()
    # CTRL-style penalty on raw logits: divide positive, multiply negative, once per seen ID.
    seen = torch.tensor(sorted(set(history)), dtype=torch.long)
    if seen.numel():
        if seen.min() < 0 or seen.max() >= scores.numel():
            raise ValueError("history IDs must belong to the vocabulary")
        picked = scores[seen]
        scores[seen] = torch.where(picked > 0, picked / repetition_penalty, picked * repetition_penalty)
    scores = scores / temperature
    if top_k is not None:
        if not isinstance(top_k, int) or not 1 <= top_k <= scores.numel():
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
    kept = probs.masked_fill(probs < min_p * probs.max(), 0.)
    return kept / kept.sum()


@torch.no_grad()
def generate(model, prompt, max_new_tokens, eos_id=None, generator=None):
    if prompt.ndim != 2 or prompt.shape[0] != 1 or prompt.shape[1] == 0:
        raise ValueError("generation demo accepts one nonempty prompt")
    if max_new_tokens < 0 or prompt.shape[1] + max_new_tokens > model.max_length:
        raise ValueError("requested output must fit the learned context table")
    result = prompt.clone()
    for _ in range(max_new_tokens):
        probabilities = distribution(model(result)[0, -1])
        next_id = torch.multinomial(probabilities, 1, generator=generator).reshape(1, 1)
        result = torch.cat((result, next_id), 1)
        if eos_id is not None and next_id.item() == eos_id:
            break
    return result


def verify():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    logits = torch.tensor([.4, .3, .2, .1], dtype=torch.float64).log()
    torch.testing.assert_close(distribution(logits, top_p=.6), torch.tensor([4/7, 3/7, 0., 0.], dtype=torch.float64))
    torch.testing.assert_close(distribution(logits, top_k=2), distribution(logits, top_p=.6))
    assert distribution(logits, temperature=.5)[0] > distribution(logits)[0]
    torch.testing.assert_close(distribution(logits, min_p=.6), torch.tensor([4/7, 3/7, 0., 0.], dtype=torch.float64))
    # Temperature first: at tau=0.5 the maximum is 0.16/0.3, so min_p=0.6 keeps only it.
    torch.testing.assert_close(distribution(logits, temperature=.5, min_p=.6), torch.tensor([1., 0., 0., 0.], dtype=torch.float64))
    # Negative logit of a seen ID is multiplied: ln 0.4 * 2 = ln 0.16.
    penalized = distribution(logits, repetition_penalty=2., history=[0, 0])
    torch.testing.assert_close(penalized, torch.tensor([.16, .3, .2, .1], dtype=torch.float64) / .76)
    # The sign rule makes this penalty depend on a shift that softmax alone ignores.
    shifted = distribution(logits + 3, repetition_penalty=2., history=[0])
    assert not torch.allclose(shifted, penalized)
    torch.testing.assert_close(distribution(logits + 3), distribution(logits))
    model = Decoder(4, max_length=8).double().eval()
    prompt = torch.tensor([[0, 1]])
    a = generate(model, prompt, 3, generator=torch.Generator().manual_seed(8))
    b = generate(model, prompt, 3, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b) and torch.equal(a[:, :2], prompt)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(-1000)
        model.head.bias[3] = 0
    assert generate(model, prompt, 3, eos_id=3).shape[1] == 3
    assert generate(model, prompt, 0).shape[1] == 2
    print("PASS: top-p crossing, top-k support, temperature, min-p, repetition penalty, seeded generation, EOS")


if __name__ == "__main__":
    verify()
