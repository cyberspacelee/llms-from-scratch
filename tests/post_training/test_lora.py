import torch
from torch import nn

from llms_from_scratch.post_training.common import tiny_gpt
from llms_from_scratch.post_training.lora import (
    ATTENTION,
    LoRALinear,
    apply_lora,
    count_trainable,
    lora_modules,
    lora_param_count,
    merge_lora,
    unmerge_lora,
)


def test_zero_init_preserves_base_output():
    torch.manual_seed(0)
    base = nn.Linear(16, 8)
    x = torch.randn(4, 16)
    expected = base(x)
    layer = LoRALinear(base, r=4, alpha=8)
    torch.testing.assert_close(layer(x), expected)
    assert torch.all(layer.B == 0) and layer.A.abs().sum() > 0


def test_gradients_flow_only_to_adapters():
    torch.manual_seed(0)
    layer = LoRALinear(nn.Linear(16, 8), r=4, alpha=8)
    layer(torch.randn(4, 16)).pow(2).sum().backward()
    assert layer.base.weight.grad is None
    assert layer.B.grad.abs().sum() > 0  # B=0 时 B 仍有梯度
    assert layer.A.grad.abs().sum() == 0  # ∂L/∂A ∝ B^T，起点为 0


def test_merge_and_unmerge_round_trip():
    torch.manual_seed(0)
    model = tiny_gpt(20, seed=0)
    idx = torch.randint(0, 20, (2, 10))
    base_logits = model(idx).detach()
    apply_lora(model, r=4, alpha=8)
    with torch.no_grad():  # 模拟训练后的适配器
        for m in lora_modules(model):
            m.B.normal_(std=0.05)
    adapted = model(idx).detach()
    assert (adapted - base_logits).abs().max() > 1e-3
    merge_lora(model)
    torch.testing.assert_close(model(idx), adapted, atol=1e-5, rtol=1e-4)
    unmerge_lora(model)
    for m in lora_modules(model):
        m.B.data.zero_()
    torch.testing.assert_close(model(idx), base_logits, atol=1e-5, rtol=1e-4)


def test_parameter_count():
    model = tiny_gpt(20, seed=0, d_model=32, n_layers=2, n_heads=4)
    apply_lora(model, targets=ATTENTION, r=2, alpha=4)
    d, d_kv = 32, 16  # n_kv_heads = 2，head_dim = 8
    per_layer = (lora_param_count(d, d, 2) + 2 * lora_param_count(d, d_kv, 2)
                 + lora_param_count(d, d, 2))
    assert count_trainable(model) == 2 * per_layer


def test_lora_training_changes_only_adapters():
    torch.set_num_threads(1)
    model = tiny_gpt(20, seed=0)
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    apply_lora(model, r=4, alpha=8)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    idx = torch.randint(0, 20, (4, 12))
    first = None
    for _ in range(20):
        logits = model(idx[:, :-1])
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 20), idx[:, 1:].reshape(-1))
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first
    state = model.state_dict()
    for k, v in frozen.items():
        key = k if k in state else k.replace(".weight", ".base.weight")
        torch.testing.assert_close(state[key], v, rtol=0, atol=0)
