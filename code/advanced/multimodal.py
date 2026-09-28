"""Trace one image through patches, visual vectors, answer labels, and loss."""
import torch
from torch import nn
from torch.nn import functional as F


def patches(image, patch_size):
    if image.ndim != 3 or patch_size <= 0:
        raise ValueError("image must be C,H,W with positive patch size")
    channels, height, width = image.shape
    if height % patch_size or width % patch_size:
        raise ValueError("teaching extractor requires divisible image dimensions")
    return image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size).permute(
        1, 2, 0, 3, 4).reshape(-1, channels * patch_size * patch_size)


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
        vision.weight[0].fill_(.25)
        vision.weight[1:5] = torch.eye(4, dtype=dtype)
    vision.requires_grad_(False)
    means = torch.tensor([2.5, 4.5, 10.5, 12.5], dtype=dtype) / 15
    torch.testing.assert_close(vision(raw)[:, 0], means)
    assert means[2:].mean() > means[:2].mean()
    connector = nn.Linear(6, 8).double()
    embedding = nn.Embedding(10, 8).double()
    visual = connector(vision(raw))
    assert visual.shape == (4, 8)
    # Serialized template: USER, IMAGE, QUESTION, ASSISTANT, ANSWER, EOS.
    sequence = torch.cat([embedding(torch.tensor([1])), visual,
                          embedding(torch.tensor([2, 3, 4, 5]))])
    assert sequence.shape == (9, 8)
    # Causal text decoder: each query sees itself and past inputs only.
    scores = sequence @ sequence.T / (8 ** .5)
    causal = torch.arange(9)[None, :] <= torch.arange(9)[:, None]
    hidden = scores.masked_fill(~causal, -torch.inf).softmax(-1) @ sequence
    head = nn.Linear(8, 10).double()
    logits = head(hidden)
    labels = torch.full((9,), -100, dtype=torch.long)
    labels[6], labels[7] = 4, 5  # ID 4: "lower half"; ID 5: EOS.
    logits.retain_grad()
    loss = F.cross_entropy(logits, labels, ignore_index=-100)
    loss.backward()
    assert torch.equal(logits.grad[:6], torch.zeros_like(logits.grad[:6]))
    assert torch.equal(logits.grad[8], torch.zeros_like(logits.grad[8]))
    assert connector.weight.grad is not None and connector.weight.grad.abs().sum() > 0
    assert vision.weight.grad is None
    swapped = image.flip(1)
    swapped_visual = connector(vision(patches(swapped, 2)))
    swapped_sequence = torch.cat([embedding(torch.tensor([1])), swapped_visual,
                                  embedding(torch.tensor([2, 3, 4, 5]))])
    swapped_scores = swapped_sequence @ swapped_sequence.T / (8 ** .5)
    swapped_hidden = swapped_scores.masked_fill(~causal, -torch.inf).softmax(-1) @ swapped_sequence
    assert not torch.allclose(head(swapped_hidden)[6], logits[6])
    print(f"VLM: patch means {means.tolist()}; 4 visual vectors; loss rows 6/7; loss={loss.item():.4f}")


if __name__ == "__main__":
    verify()
