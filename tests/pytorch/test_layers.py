import torch
import torch.nn.functional as F
from torch import nn

from llms_from_scratch.pytorch.layers import (
    MLP,
    Embedding,
    Linear,
    count_parameters,
    trunc_normal_init_,
)


def test_linear_matches_nn_linear_bit_for_bit():
    torch.manual_seed(0)
    ours = Linear(7, 5)
    torch.manual_seed(0)
    ref = nn.Linear(7, 5)
    torch.testing.assert_close(ours.weight, ref.weight, rtol=0, atol=1e-7)
    torch.testing.assert_close(ours.bias, ref.bias, rtol=0, atol=1e-7)
    x = torch.randn(3, 4, 7)
    torch.testing.assert_close(ours(x), ref(x))


def test_embedding_is_one_hot_matmul():
    emb = Embedding(10, 4)
    idx = torch.tensor([[1, 3], [9, 1]])
    out = emb(idx)
    assert out.shape == (2, 2, 4)
    torch.testing.assert_close(out, F.one_hot(idx, 10).float() @ emb.weight)
    out.sum().backward()
    assert emb.weight.grad[1].eq(2).all()  # token 1 出现两次，梯度累加
    assert emb.weight.grad[0].eq(0).all()


def test_module_tree_parameters_buffers_state_dict():
    model = MLP(4, 8, 2, mean=torch.arange(4.0), std=torch.full((4,), 2.0))
    names = [n for n, _ in model.named_parameters()]
    assert names == ["fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]
    assert [n for n, _ in model.named_buffers()] == ["norm.mean", "norm.std"]
    assert set(model.state_dict()) == set(names) | {"norm.mean", "norm.std"}
    assert count_parameters(model) == 4 * 8 + 8 + 8 * 2 + 2
    clone = MLP(4, 8, 2)
    clone.load_state_dict(model.state_dict())
    torch.testing.assert_close(clone.norm.mean, torch.arange(4.0))
    half = model.to(torch.float64)
    assert half.fc1.weight.dtype == torch.float64 and half.norm.mean.dtype == torch.float64


def test_train_eval_changes_dropout_only():
    torch.manual_seed(0)
    model = MLP(4, 64, 3, dropout=0.5)
    x = torch.randn(16, 4)
    model.eval()
    assert torch.equal(model(x), model(x))
    model.train()
    assert model.training and model.drop.training
    assert not torch.equal(model(x), model(x))


def test_requires_grad_freeze():
    model = MLP(4, 8, 2)
    model.fc1.requires_grad_(False)
    assert count_parameters(model) == 8 * 2 + 2


def test_trunc_normal_init_bounds_and_scale():
    torch.manual_seed(0)
    w = trunc_normal_init_(torch.empty(256, 512))
    std = (2 / (256 + 512)) ** 0.5
    assert w.abs().max() <= 3 * std + 1e-6
    assert abs(w.std().item() - std * 0.986) < 0.02 * std  # 截断使标准差略小于 σ
