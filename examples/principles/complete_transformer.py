"""One token stream through training, parameter counting, and cached decode."""

import torch

from llms_from_scratch import Transformer, decoder_config
from llms_from_scratch import token_loss as sequence_loss
from llms_from_scratch.analysis import parameter_count
from llms_from_scratch.inference.sampling import sample_generate


def main():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = Transformer(
        decoder_config(
            8, dim=12, ff_dim=20, layers=2, heads=4, kv_heads=2, head_dim=4, max_length=12
        )
    ).double()
    stream = torch.tensor([[0, 1, 2, 3, 4]])  # BOS, 春, 风, 来, EOS
    inputs, targets = stream[:, :-1], stream[:, 1:]

    logits = model(inputs).logits
    assert logits.shape == (1, 4, 8)
    assert parameter_count(model) == sum(p.numel() for p in model.parameters()) == 2748
    assert model.embedding.weight is model.head.weight
    initial_loss = sequence_loss(logits, targets).item()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(70):
        optimizer.zero_grad(set_to_none=True)
        loss = sequence_loss(model(inputs).logits, targets)
        loss.backward()
        optimizer.step()
    final_loss = sequence_loss(model(inputs).logits, targets).item()
    assert final_loss < initial_loss and final_loss < 0.05

    model.eval()
    with torch.no_grad():
        full = model(stream).logits
        for lengths in ((5,), (2, 1, 2), (1, 1, 1, 1, 1)):
            caches, outputs, offset = None, [], 0
            for length in lengths:
                result = model(stream[:, offset : offset + length], cache=caches, use_cache=True)
                part, caches = result.logits, result.cache
                outputs.append(part)
                offset += length
                assert all(
                    layer.self_attention.key.shape
                    == layer.self_attention.value.shape
                    == (1, 2, offset, 4)
                    for layer in caches.layers
                )
            torch.testing.assert_close(torch.cat(outputs, dim=1), full)
        prompt = stream[:, :2]
        sample = sample_generate(
            model, prompt, 3, eos_id=4, generator=torch.Generator().manual_seed(9)
        )
        assert sample.shape == (1, 5) and torch.equal(sample[:, :2], prompt)
    print(
        f"PASS: logits (1,4,8), 2748 shared parameters, loss {initial_loss:.6f} -> {final_loss:.6f}"
    )
    print(f"PASS: full/chunk/token logits, per-layer KV shapes, sample {sample.tolist()}")


if __name__ == "__main__":
    main()
