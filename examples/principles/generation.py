"""Filtering, reproducible sampling, and decoder generation stopping checks."""

from types import SimpleNamespace

import torch

from llms_from_scratch import Transformer, basic_decoder_config
from llms_from_scratch.inference.sampling import distribution, sample_generate


def verify():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    # ID order: BOS, A, B, EOS; the prompt is [BOS, A].
    first = torch.tensor([0.1, 0.4, 0.3, 0.2], dtype=torch.float64)
    logits = first.log()
    kept = torch.tensor([0.0, 4 / 7, 3 / 7, 0.0], dtype=torch.float64)
    torch.testing.assert_close(distribution(logits), first)
    torch.testing.assert_close(distribution(logits, top_p=0.6), kept)
    torch.testing.assert_close(distribution(logits, top_k=2), distribution(logits, top_p=0.6))
    torch.testing.assert_close(
        distribution(logits, temperature=0.5), first.square() / first.square().sum()
    )
    torch.testing.assert_close(distribution(logits, min_p=0.6), kept)
    torch.testing.assert_close(
        distribution(logits, temperature=0.5, min_p=0.6),
        torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float64),
    )
    # BOS and A have appeared in the prompt; both negative logits are multiplied.
    penalized = distribution(logits, repetition_penalty=2.0, history=[0, 1, 1])
    torch.testing.assert_close(
        penalized, torch.tensor([0.01, 0.16, 0.3, 0.2], dtype=torch.float64) / 0.67
    )
    # The sign rule makes this penalty depend on a shift that softmax alone ignores.
    shifted = distribution(logits + 3, repetition_penalty=2.0, history=[0, 1])
    assert not torch.allclose(shifted, penalized)
    torch.testing.assert_close(distribution(logits + 3), distribution(logits))
    assert (distribution(logits, top_p=0.8) > 0).tolist() == [False, True, True, True]

    class TableModel(torch.nn.Module):
        config = SimpleNamespace(max_length=4, vocab_size=4)

        def forward(self, ids, **kwargs):
            if ids.shape[1] == 2:
                probabilities = first
            elif ids[0, -1] == 1:
                probabilities = torch.tensor([0.2, 0.2, 0.2, 0.4], dtype=torch.float64)
            else:
                probabilities = torch.tensor([0.1, 0.1, 0.2, 0.6], dtype=torch.float64)
            return SimpleNamespace(
                logits=probabilities.log().expand(ids.shape[1], -1)[None], cache=None
            )

    table = TableModel()
    prompt = torch.tensor([[0, 1]])
    torch.testing.assert_close(
        sample_generate(table, prompt, 2, eos_id=3, top_k=1, repetition_penalty=2.0),
        torch.tensor([[0, 1, 2, 3]]),
    )
    torch.testing.assert_close(
        sample_generate(table, prompt, 2, eos_id=3, top_k=1), torch.tensor([[0, 1, 1, 3]])
    )
    assert 0.4 * 0.4 < 0.3 * 0.6
    torch.testing.assert_close(sample_generate(table, prompt, 0), prompt)
    try:
        sample_generate(table, prompt, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("context overflow must fail")
    model = Transformer(basic_decoder_config(4, max_length=8)).double().eval()
    a = sample_generate(model, prompt, 3, top_p=0.6, generator=torch.Generator().manual_seed(8))
    b = sample_generate(model, prompt, 3, top_p=0.6, generator=torch.Generator().manual_seed(8))
    assert torch.equal(a, b) and torch.equal(a[:, :2], prompt)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(-1000)
        model.head.bias[3] = 0
    assert sample_generate(model, prompt, 3, eos_id=3).shape[1] == 3
    assert sample_generate(model, prompt, 0).shape[1] == 2
    print(
        "PASS: four-ID distribution, filters, penalty, generation loop, budget, context, seeded Decoder, EOS"
    )


if __name__ == "__main__":
    verify()
