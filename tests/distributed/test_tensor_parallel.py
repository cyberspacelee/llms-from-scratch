import torch
import torch.nn.functional as F
from torch import nn

from llms_from_scratch.distributed.comm import launch
from llms_from_scratch.distributed.tensor_parallel import (
    ColumnParallelLinear,
    SequenceParallelFFN,
    TPAttention,
    TPSwiGLU,
    ring_attention,
    vocab_parallel_cross_entropy,
)
from llms_from_scratch.transformer.model import (
    CausalSelfAttention,
    GPTConfig,
    RMSNorm,
    SwiGLU,
    rope_frequencies,
)

CONFIG = GPTConfig(vocab_size=48, context_length=16, d_model=32, n_layers=1, n_heads=4,
                   n_kv_heads=2, d_ff=64)
B, T = 2, 8


def _modules():
    torch.manual_seed(0)
    mlp = SwiGLU(CONFIG.d_model, CONFIG.d_ff).double()
    attn = CausalSelfAttention(CONFIG, 0).double()
    head = nn.Linear(CONFIG.d_model, CONFIG.vocab_size, bias=False).double()
    norm = RMSNorm(CONFIG.d_model)
    norm.weight.data = 1 + 0.1 * torch.randn(CONFIG.d_model)
    norm = norm.double()
    x = torch.randn(B, T, CONFIG.d_model, dtype=torch.float64)
    target = torch.randint(0, CONFIG.vocab_size, (B * T,))
    q, k, v = torch.randn(3, B, 4, 2 * T, 8, dtype=torch.float64)  # Ring Attention 用更长的序列
    return mlp, attn, head, norm, x, target, (q, k, v)


def _grads(module):
    return {n: p.grad.clone() for n, p in module.named_parameters()}


def _tp_worker(rank: int, t: int) -> dict:
    mlp, attn, head, norm, x, target, (q, k, v) = _modules()
    freqs = rope_frequencies(CONFIG.head_dim, T)
    out = {}

    x1 = x.clone().requires_grad_()
    tp_mlp = TPSwiGLU(mlp)
    y = tp_mlp(x1)
    (y * y).sum().backward()
    out["mlp"] = (y.detach(), x1.grad, _grads(tp_mlp))

    x2 = x.clone().requires_grad_()
    tp_attn = TPAttention(attn)
    y = tp_attn(x2, freqs)
    (y * y).sum().backward()
    out["attn"] = (y.detach(), x2.grad, _grads(tp_attn))

    x3 = x.clone().requires_grad_()
    tp_head = ColumnParallelLinear(head)  # 词表并行的输出层
    loss = vocab_parallel_cross_entropy(tp_head(x3).flatten(0, 1), target)
    loss.backward()
    out["vocab"] = (loss.detach(), x3.grad, tp_head.weight.grad)

    shard = x.chunk(t, dim=1)[rank].clone().requires_grad_()  # 序列并行：每个 rank 一段序列
    sp = SequenceParallelFFN(norm, mlp)
    y = sp(shard)
    (y * y).sum().backward()
    sp.sync_replicated_grads()
    out["sp"] = (y.detach(), shard.grad, sp.norm.weight.grad, sp.ffn.w1.grad)

    seq = slice(rank * (2 * T // t), (rank + 1) * (2 * T // t))
    out["ring"] = ring_attention(q[:, :, seq], k[:, :, seq], v[:, :, seq])
    return out


def test_tensor_parallel_matches_single_device():
    t = 2
    mlp, attn, head, norm, x, target, (q, k, v) = _modules()
    freqs = rope_frequencies(CONFIG.head_dim, T)
    results = launch(_tp_worker, t)

    def check_branch(name, module, fn, shard_map):
        x_ref = x.clone().requires_grad_()
        y = fn(x_ref)
        (y * y).sum().backward()
        for rank, res in enumerate(results):
            y_tp, gx, grads = res[name]
            torch.testing.assert_close(y_tp, y.detach())
            torch.testing.assert_close(gx, x_ref.grad)
            for tp_name, (full_name, dim) in shard_map.items():
                full_grad = dict(module.named_parameters())[full_name].grad
                torch.testing.assert_close(grads[tp_name], full_grad.chunk(t, dim=dim)[rank])

    check_branch("mlp", mlp, mlp, {"w1": ("w1.weight", 0), "w3": ("w3.weight", 0),
                                   "w2": ("w2.weight", 1)})
    check_branch("attn", attn, lambda z: attn(z, freqs),
                 {"wq": ("q_proj.weight", 0), "wk": ("k_proj.weight", 0),
                  "wv": ("v_proj.weight", 0), "wo": ("o_proj.weight", 1)})

    # 词表并行交叉熵：损失、输入梯度、各词表分片的权重梯度
    x_ref = x.clone().requires_grad_()
    loss = F.cross_entropy(head(x_ref).flatten(0, 1), target)
    loss.backward()
    for rank, (loss_tp, gx, gw) in enumerate(r["vocab"] for r in results):
        torch.testing.assert_close(loss_tp, loss.detach().float())
        torch.testing.assert_close(gx, x_ref.grad)
        torch.testing.assert_close(gw, head.weight.grad.chunk(t, dim=0)[rank])

    # 序列并行：输出与输入梯度沿序列切分，复制的 norm 权重梯度经 all-reduce 后与单卡一致
    mlp.zero_grad()
    x_ref = x.clone().requires_grad_()
    y = x_ref + mlp(norm(x_ref))
    (y * y).sum().backward()
    for rank, (y_sp, gx, g_norm, g_w1) in enumerate(r["sp"] for r in results):
        torch.testing.assert_close(y_sp, y.detach().chunk(t, dim=1)[rank])
        torch.testing.assert_close(gx, x_ref.grad.chunk(t, dim=1)[rank])
        torch.testing.assert_close(g_norm, norm.weight.grad)
        torch.testing.assert_close(g_w1, mlp.w1.weight.grad.chunk(t, dim=0)[rank])

    # Ring Attention：拼起来等于完整的因果注意力
    full = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    ring = torch.cat([r["ring"] for r in results], dim=2)
    torch.testing.assert_close(ring, full)


def _ring4_worker(rank: int, t: int) -> torch.Tensor:
    q, k, v = _modules()[-1]
    seq = slice(rank * (2 * T // t), (rank + 1) * (2 * T // t))
    return ring_attention(q[:, :, seq], k[:, :, seq], v[:, :, seq])


def test_ring_attention_four_ranks():
    q, k, v = _modules()[-1]
    full = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.testing.assert_close(torch.cat(launch(_ring4_worker, 4), dim=2), full)
