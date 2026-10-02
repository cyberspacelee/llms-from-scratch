"""Trace one image through patches, visual vectors, answer labels, and loss."""

import torch
from torch import nn

from llms_from_scratch import Transformer, decoder_config, token_loss


def patches(image, patch_size):
    if image.ndim != 3 or patch_size <= 0:
        raise ValueError("image must be C,H,W with positive patch size")
    channels, height, width = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError("teaching extractor requires divisible image dimensions")
    return (
        image.unfold(1, patch_size, patch_size)
        .unfold(2, patch_size, patch_size)
        .permute(1, 2, 0, 3, 4)
        .reshape(-1, channels * patch_size * patch_size)
    )


def verify():
    torch.manual_seed(23)
    dtype = torch.float64
    image = torch.arange(16, dtype=dtype).reshape(1, 4, 4) / 15
    raw = patches(image, 2)
    torch.testing.assert_close(raw[0], torch.tensor([0, 1, 4, 5], dtype=dtype) / 15)
    assert raw.shape == (4, 4)
    vision = nn.Linear(4, 6).double()
    # This fixed teaching encoder exposes each patch's mean in feature 0.
    with torch.no_grad():
        vision.weight.zero_()
        vision.bias.zero_()
        vision.weight[0].fill_(0.25)
        vision.weight[1:5] = torch.eye(4, dtype=dtype)
    vision.requires_grad_(False)
    means = torch.tensor([2.5, 4.5, 10.5, 12.5], dtype=dtype) / 15
    torch.testing.assert_close(vision(raw)[:, 0], means)
    assert means[2:].mean() > means[:2].mean()
    # region visual_prefix
    connector = nn.Linear(6, 8).double()
    model = Transformer(
        decoder_config(
            10, dim=8, ff_dim=16, layers=1, heads=2, kv_heads=1, head_dim=4, max_length=16
        )
    ).double()
    model.requires_grad_(
        False
    )  # Train the connector while retaining its path through the frozen LM.
    embedding = model.embedding
    visual = connector(vision(raw))
    assert visual.shape == (4, 8)
    # Serialized template: USER, IMAGE, QUESTION, ASSISTANT, ANSWER, EOS.
    sequence = torch.cat(
        [embedding(torch.tensor([1])), visual, embedding(torch.tensor([2, 3, 4, 5]))]
    )
    assert sequence.shape == (9, 8)
    # The same causal decoder used by pretraining consumes the expanded embedding prefix.
    expanded_ids = torch.tensor([[1, 0, 0, 0, 0, 2, 3, 4, 5]])
    logits = model(expanded_ids, inputs_embeds=sequence.unsqueeze(0)).logits[0]
    labels = torch.full((9,), -100, dtype=torch.long)
    labels[6], labels[7] = 4, 5  # ID 4: "lower half"; ID 5: EOS.
    logits.retain_grad()
    loss = token_loss(logits.unsqueeze(0), labels.unsqueeze(0), (labels >= 0).unsqueeze(0))
    loss.backward()
    assert torch.equal(logits.grad[:6], torch.zeros_like(logits.grad[:6]))
    assert torch.equal(logits.grad[8], torch.zeros_like(logits.grad[8]))
    assert connector.weight.grad is not None and connector.weight.grad.abs().sum() > 0
    assert vision.weight.grad is None
    # endregion visual_prefix
    swapped = image.flip(1)
    swapped_visual = connector(vision(patches(swapped, 2)))
    swapped_sequence = torch.cat(
        [embedding(torch.tensor([1])), swapped_visual, embedding(torch.tensor([2, 3, 4, 5]))]
    )
    swapped_logits = model(expanded_ids, inputs_embeds=swapped_sequence.unsqueeze(0)).logits[0]
    assert not torch.allclose(swapped_logits[6], logits[6])
    # Prefill must include the four visual rows; resumed positions use the expanded prefix length.
    with torch.no_grad():
        prefix = model(expanded_ids[:, :7], inputs_embeds=sequence[:7].unsqueeze(0), use_cache=True)
        suffix = model(
            expanded_ids[:, 7:],
            inputs_embeds=sequence[7:].unsqueeze(0),
            cache=prefix.cache,
            use_cache=True,
        )
        torch.testing.assert_close(suffix.logits[0], logits[7:])
        assert prefix.cache.length == 7 and suffix.cache.length == 9
    print(
        f"VLM: patch means {means.tolist()}; 4 visual vectors; loss rows 6/7; loss={loss.item():.4f}"
    )


if __name__ == "__main__":
    verify()
