"""后训练各章共用的小工具：逐 token 对数概率、掩码平均、带自定义掩码的前向、采样。

这些函数只依赖全书共用的 ``transformer/model.py``，不复制模型代码：
``gpt_hidden`` 复用 GPT 的各个子模块，只是把因果掩码与位置换成调用方给定的版本。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from llms_from_scratch.transformer.model import GPT, GPTConfig, apply_rope


def tiny_gpt(vocab_size: int, seed: int = 0, context_length: int = 64, d_model: int = 32,
             n_layers: int = 2, n_heads: int = 4) -> GPT:
    """测试与示例用的极小 GPT（约两万参数），在 CPU 上一步只需毫秒级。"""
    torch.manual_seed(seed)
    config = GPTConfig(vocab_size=vocab_size, context_length=context_length, d_model=d_model,
                       n_layers=n_layers, n_heads=n_heads, n_kv_heads=n_heads // 2)
    return GPT(config)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    """只对 mask 为真的位置求平均。"""
    mask = mask.to(x.dtype)
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp_min(1)
    return (x * mask).sum(dim) / mask.sum(dim).clamp_min(1)


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits[B, T, V] 与 targets[B, T] → 每个目标 token 的 log π(target)，形状 [B, T]。

    用 log_softmax + gather 而不是 softmax 后取 log，避免小概率下溢。
    """
    return logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def response_logprobs(model: GPT, input_ids: torch.Tensor) -> torch.Tensor:
    """序列 x_0..x_{T-1} 中每个位置 t ≥ 1 的 log π(x_t | x_<t)，形状 [B, T-1]。

    第 t-1 个位置的 logits 预测第 t 个 token，所以输出与 ``input_ids[:, 1:]`` 对齐。
    """
    logits = model(input_ids)[:, :-1]
    return token_logprobs(logits, input_ids[:, 1:])


def gpt_hidden(model: GPT, idx: torch.Tensor, attn_mask: torch.Tensor | None = None,
               position_ids: torch.Tensor | None = None) -> torch.Tensor:
    """返回 GPT 最后一层归一化后的隐藏状态 [B, T, d]，可指定注意力掩码与位置。

    attn_mask: [B, T, T] 布尔张量，True 表示允许注意；None 时为普通因果掩码。
    position_ids: [B, T]，每个 token 的 RoPE 位置；None 时为 0..T-1。
    """
    B, T = idx.shape
    if position_ids is None:
        position_ids = torch.arange(T, device=idx.device).expand(B, T)
    if attn_mask is None:
        attn_mask = torch.ones(T, T, dtype=torch.bool, device=idx.device).tril().expand(B, T, T)
    freqs = model.freqs[position_ids].unsqueeze(1)  # [B, 1, T, head_dim/2]，对所有头广播
    mask = attn_mask.unsqueeze(1)  # [B, 1, T, T]
    x = model.embed(idx)
    for block in model.blocks:
        attn = block.attn
        h = block.attn_norm(x)
        q = attn.q_proj(h).view(B, T, attn.n_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(h).view(B, T, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
        v = attn.v_proj(h).view(B, T, attn.n_kv_heads, attn.head_dim).transpose(1, 2)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        repeat = attn.n_heads // attn.n_kv_heads
        k, v = k.repeat_interleave(repeat, dim=1), v.repeat_interleave(repeat, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + attn.o_proj(out.transpose(1, 2).reshape(B, T, -1))
        x = x + block.ffn(block.ffn_norm(x))
    return model.norm(x)


@dataclass
class Rollout:
    """一批采样结果。sequences = prompt + response；response_mask 标出需要计算损失的回复 token。"""

    sequences: torch.Tensor  # [N, P + R]
    response_mask: torch.Tensor  # [N, P + R - 1]，与 response_logprobs 的输出对齐
    responses: list[list[int]]  # 每条回复（截到 EOS，含 EOS）


@torch.no_grad()
def sample_responses(model: GPT, prompts: torch.Tensor, max_new_tokens: int, eos_id: int,
                     temperature: float = 1.0, generator: torch.Generator | None = None) -> Rollout:
    """对等长的 prompts[N, P] 各采样一条回复；EOS 之后的 token 在掩码中被排除。"""
    model.eval()
    seq = model.generate(prompts, max_new_tokens, temperature=temperature, generator=generator)
    model.train()
    P = prompts.shape[1]
    gen = seq[:, P:]
    # EOS 之后（不含 EOS 本身）的位置无效：cumsum 统计“此前是否已经出现过 EOS”
    is_eos = gen == eos_id
    after_eos = (is_eos.cumsum(1) - is_eos.long()) > 0
    valid = ~after_eos
    mask = torch.zeros(seq.shape[0], seq.shape[1] - 1, dtype=torch.bool, device=seq.device)
    mask[:, P - 1:] = valid
    responses = [g[v].tolist() for g, v in zip(gen, valid, strict=True)]
    return Rollout(seq, mask, responses)
