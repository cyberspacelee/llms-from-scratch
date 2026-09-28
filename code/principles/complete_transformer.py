"""One token stream through training, parameter counting, and cached decode."""

import torch

from language_model import sequence_loss
from modern_decoder import ModernDecoder, generate_cached


def main():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = ModernDecoder(8, width=12, ff_width=20, layers=2,
                          heads=4, kv_heads=2, head_width=4, max_length=12).double()
    stream = torch.tensor([[0, 1, 2, 3, 4]])  # BOS, 春, 风, 来, EOS
    inputs, targets = stream[:, :-1], stream[:, 1:]

    logits = model(inputs)
    assert logits.shape == (1, 4, 8)
    assert model.parameter_count() == sum(p.numel() for p in model.parameters()) == 2748
    assert model.embedding.weight is model.head.weight
    initial_loss = sequence_loss(logits, targets).item()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    for _ in range(70):
        optimizer.zero_grad(set_to_none=True)
        loss = sequence_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
    final_loss = sequence_loss(model(inputs), targets).item()
    assert final_loss < initial_loss and final_loss < 0.05

    model.eval()
    with torch.no_grad():
        full = model(stream)
        for lengths in ((5,), (2, 1, 2), (1, 1, 1, 1, 1)):
            caches, outputs, offset = None, [], 0
            for length in lengths:
                part, caches = model.forward_cached(stream[:, offset:offset + length], caches)
                outputs.append(part)
                offset += length
                assert all(k.shape == v.shape == (1, 2, offset, 4) for k, v in caches)
            torch.testing.assert_close(torch.cat(outputs, dim=1), full)
        prompt = stream[:, :2]
        sample = generate_cached(model, prompt, 3, generator=torch.Generator().manual_seed(9))
        assert sample.shape == (1, 5) and torch.equal(sample[:, :2], prompt)
    print(f"PASS: logits (1,4,8), 2748 shared parameters, loss {initial_loss:.6f} -> {final_loss:.6f}")
    print(f"PASS: full/chunk/token logits, per-layer KV shapes, sample {sample.tolist()}")


if __name__ == "__main__":
    main()
