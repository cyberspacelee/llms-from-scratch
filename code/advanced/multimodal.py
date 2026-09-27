"""Patch extraction, visual connector, placeholder expansion and text-only loss."""
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
    connector = nn.Linear(6, 8).double()
    embedding = nn.Embedding(10, 8).double()
    visual = connector(vision(raw))
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
    labels[6], labels[7] = 4, 5  # ASSISTANT predicts ANSWER; ANSWER predicts EOS.
    logits.retain_grad()
    F.cross_entropy(logits, labels, ignore_index=-100).backward()
    assert torch.equal(logits.grad[:6], torch.zeros_like(logits.grad[:6]))
    assert connector.weight.grad is not None and connector.weight.grad.abs().sum() > 0
    assert vision.weight.grad is not None and torch.isfinite(vision.weight.grad).all()
    print("VLM: 4 image patches -> 4 visual vectors; 9-position sequence; text-only loss")


if __name__ == "__main__":
    verify()
