"""MHA、MQA 与 GQA：KV Cache 账本、逐头参考实现、免复制的分组计算与 MHA→GQA 转换。

注意力模块本身复用 ``transformer.model.CausalSelfAttention``——它的 ``n_kv_heads``
参数同时覆盖了三种情形：等于 ``n_heads`` 是 MHA，等于 1 是 MQA，介于两者之间是 GQA。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from llms_from_scratch.transformer.model import CausalSelfAttention, GPTConfig, apply_rope


def kv_cache_bytes(batch: int, seq_len: int, n_layers: int, n_kv_heads: int, head_dim: int,
                   bytes_per_elem: int = 2) -> int:
    """KV Cache 字节数 = 2（K 与 V）· B · T · L · n_kv · d_h · 每元素字节。"""
    return 2 * batch * seq_len * n_layers * n_kv_heads * head_dim * bytes_per_elem


# region reference
def attention_reference(attn: CausalSelfAttention, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """逐个查询头循环的因果注意力：第 h 个查询头读取第 h // (H / H_kv) 个 K/V 头。"""
    B, T, _ = x.shape
    H, H_kv, d_h = attn.n_heads, attn.n_kv_heads, attn.head_dim
    q = apply_rope(attn.q_proj(x).view(B, T, H, d_h).transpose(1, 2), freqs)
    k = apply_rope(attn.k_proj(x).view(B, T, H_kv, d_h).transpose(1, 2), freqs)
    v = attn.v_proj(x).view(B, T, H_kv, d_h).transpose(1, 2)
    causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
    heads = []
    for h in range(H):
        g = h // (H // H_kv)  # 该查询头所属的组 = 共享的 K/V 头编号
        scores = q[:, h] @ k[:, g].transpose(-1, -2) / math.sqrt(d_h)
        scores = scores.masked_fill(~causal, float("-inf"))
        heads.append(scores.softmax(-1) @ v[:, g])
    return attn.o_proj(torch.cat(heads, dim=-1))
# endregion


# region grouped
def grouped_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """不复制 K/V 的 GQA：q[B, H, T, d]，k/v[B, H_kv, S, d]，返回 [B, H, T, d]。

    把查询头整理成 [B, H_kv, g, T, d]，同组的 g 个头与同一个 K/V 头做批量矩阵乘法，
    因此 K/V 只从显存读一次，这正是 GQA 在 decode 阶段省带宽的原因。
    """
    B, H, T, d = q.shape
    H_kv, S = k.shape[1], k.shape[2]
    qg = q.view(B, H_kv, H // H_kv, T, d)
    scores = qg @ k.unsqueeze(2).transpose(-1, -2) / math.sqrt(d)  # [B, H_kv, g, T, S]
    mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(diagonal=S - T)
    probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
    return (probs @ v.unsqueeze(2)).reshape(B, H, T, d)
# endregion


# region uptrain
@torch.no_grad()
def mha_to_gqa(mha: CausalSelfAttention, n_kv_heads: int) -> CausalSelfAttention:
    """把 MHA 检查点转换为 GQA：每组内 K/V 投影取平均（Ainslie 等 2023 的 mean-pooling）。

    转换后的模型还需用原预训练数据的约 5% 继续训练（uptraining）来恢复质量。
    """
    H, d_h = mha.n_heads, mha.head_dim
    d_model = mha.q_proj.in_features
    cfg = GPTConfig(d_model=d_model, n_heads=H, n_kv_heads=n_kv_heads)
    gqa = CausalSelfAttention(cfg, mha.layer)
    gqa.q_proj.load_state_dict(mha.q_proj.state_dict())
    gqa.o_proj.load_state_dict(mha.o_proj.state_dict())
    for name in ("k_proj", "v_proj"):
        w = getattr(mha, name).weight.view(n_kv_heads, H // n_kv_heads, d_h, d_model)
        getattr(gqa, name).weight.copy_(w.mean(dim=1).reshape(n_kv_heads * d_h, d_model))
    return gqa
# endregion


def sdpa_with_repeat(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """对照实现：先 repeat_interleave 复制 K/V 头，再调用 PyTorch 的 SDPA。"""
    repeat = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
    T, S = q.shape[2], k.shape[2]
    mask = torch.ones(T, S, dtype=torch.bool, device=q.device).tril(diagonal=S - T)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
