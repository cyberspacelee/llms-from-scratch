import math

import torch
from torch.utils.flop_counter import FlopCounterMode

from llms_from_scratch.transformer.accounting import count_parameters, forward_flops
from llms_from_scratch.transformer.attention import scaled_dot_product_attention
from llms_from_scratch.transformer.model import GPT, GPTConfig


def test_parameter_formula_matches_model():
    for kw in [dict(), dict(n_kv_heads=2), dict(tie_embeddings=False, d_ff=200)]:
        config = GPTConfig(vocab_size=100, d_model=64, n_layers=3, n_heads=4, **kw)
        model = GPT(config)
        assert count_parameters(config)["total"] == sum(p.numel() for p in model.parameters())


def test_flop_formula_matches_counter():
    config = GPTConfig(vocab_size=100, context_length=64, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2)
    model = GPT(config)
    B, T = 2, 32
    with FlopCounterMode(display=False) as counter:
        model(torch.randint(0, 100, (B, T)))
    f = forward_flops(config, T, B)
    # CPU 上的融合 SDPA 不计入计数器：线性层部分单独核对
    assert counter.get_total_flops() == f["total"] - f["attn_scores"] - f["attn_values"]
    q = k = v = torch.randn(B, config.n_heads, T, config.head_dim)
    with FlopCounterMode(display=False) as counter:
        scaled_dot_product_attention(q, k, v)
    assert counter.get_total_flops() * config.n_layers == f["attn_scores"] + f["attn_values"]


def test_initial_loss_is_near_uniform():
    # 小初始化下 logits 接近 0，预测接近均匀分布，交叉熵约为 ln V
    torch.manual_seed(0)
    config = GPTConfig(vocab_size=512, context_length=64, d_model=64, n_layers=2, n_heads=4)
    model = GPT(config)
    idx = torch.randint(0, 512, (4, 64))
    loss = torch.nn.functional.cross_entropy(model(idx)[:, :-1].flatten(0, 1), idx[:, 1:].flatten())
    assert abs(loss.item() - math.log(512)) < 0.05
