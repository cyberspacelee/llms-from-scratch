import numpy as np
import torch
import torch.nn.functional as F

from llms_from_scratch.transformer.data import get_batch, synthetic_stories
from llms_from_scratch.transformer.train import TrainConfig, cross_entropy, train


def test_cross_entropy_matches_torch():
    torch.manual_seed(0)
    logits = torch.randn(4, 7, 50) * 30
    y = torch.randint(0, 50, (4, 7))
    torch.testing.assert_close(cross_entropy(logits, y), F.cross_entropy(logits.flatten(0, 1), y.flatten()))


def test_get_batch_targets_are_shifted_inputs():
    tokens = np.arange(100, dtype=np.uint16)
    x, y = get_batch(tokens, 8, 10, generator=torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (8, 10) and x.dtype == torch.long
    torch.testing.assert_close(y, x + 1)


def test_stories_are_deterministic():
    assert synthetic_stories(3, seed=1) == synthetic_stories(3, seed=1)
    assert synthetic_stories(3, seed=1) != synthetic_stories(3, seed=2)


def test_tiny_training_run_reduces_loss():
    cfg = TrainConfig(n_stories=100, vocab_size=300, context_length=32, d_model=32, n_layers=1,
                      n_heads=2, batch_size=8, steps=25, warmup_steps=5, eval_every=25, eval_batches=2)
    result = train(cfg, log=lambda s: None)
    losses = result["history"]["train"]
    assert np.mean(losses[-5:]) < np.mean(losses[:5]) - 0.5
