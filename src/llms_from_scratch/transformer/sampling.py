"""解码策略与带 / 不带 KV Cache 的自回归生成。

所有过滤函数都作用在 logits 的最后一维（词表维）上，把被排除的 token 置为 -inf，
随后的 softmax 自动把剩余概率重新归一化。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import GPT, KVCache


# region filters
def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """只保留最大的 k 个 logit（并列时可能多于 k 个）。"""
    k = min(k, logits.shape[-1])
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]
    return logits.masked_fill(logits < kth, float("-inf"))


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """核采样：保留概率从大到小累加、首次达到 p 的最小集合。概率最大的 token 总会保留。"""
    sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    # 某个 token 之前（不含自身）的累积概率已经 >= p，它就不再需要
    mass_before = probs.cumsum(-1) - probs
    drop_sorted = mass_before >= p
    drop = drop_sorted.scatter(-1, order, drop_sorted)  # 把排序后的掩码放回原位置
    return logits.masked_fill(drop, float("-inf"))


def apply_repetition_penalty(logits: torch.Tensor, history: torch.Tensor, penalty: float) -> torch.Tensor:
    """CTRL 式重复惩罚：出现过的 token，正 logit 除以 penalty、负 logit 乘以 penalty。

    logits: [B, V]，history: [B, T] 已生成（含提示）的 token。
    """
    if penalty == 1.0:
        return logits
    seen = torch.gather(logits, -1, history)
    seen = torch.where(seen > 0, seen / penalty, seen * penalty)
    return logits.scatter(-1, history, seen)
# endregion


# region sample
def sample_next(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    history: torch.Tensor | None = None,
    repetition_penalty: float = 1.0,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """从 logits [B, V] 选出下一个 token [B, 1]。temperature=0 表示贪心。"""
    if history is not None:
        logits = apply_repetition_penalty(logits, history, repetition_penalty)
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    logits = logits / temperature  # 温度先作用，再截断：截断看到的是最终分布
    if top_k is not None:
        logits = top_k_filter(logits, top_k)
    if top_p is not None:
        logits = top_p_filter(logits, top_p)
    return torch.multinomial(F.softmax(logits, dim=-1), 1, generator=generator)
# endregion


# region generate
@torch.no_grad()
def generate(
    model: GPT,
    prompt: torch.Tensor,
    max_new_tokens: int,
    use_cache: bool = True,
    eos_id: int | None = None,
    **sampling,
) -> torch.Tensor:
    """自回归生成。use_cache=False 时每步把整个前缀重新送入模型（O(T²) 的重复计算）。

    prompt: [B, T0]；返回 [B, T0 + 生成长度]。``sampling`` 透传给 ``sample_next``。
    """
    model.eval()
    idx = prompt
    limit = model.config.context_length
    cache = KVCache(model.config, idx.shape[0], device=idx.device) if use_cache else None
    if use_cache:
        logits = model(idx, cache, 0)[:, -1]  # prefill：一次处理整段提示
    else:
        logits = model(idx[:, -limit:])[:, -1]
    for _ in range(max_new_tokens):
        nxt = sample_next(logits, history=idx, **sampling)
        idx = torch.cat([idx, nxt], dim=1)
        if eos_id is not None and bool((nxt == eos_id).all()):
            break
        if idx.shape[1] >= limit:
            break
        if use_cache:
            logits = model(nxt, cache, cache.length)[:, -1]  # decode：只送入一个新 token
        else:
            logits = model(idx)[:, -1]
    return idx
# endregion


# region cache_bytes
def kv_cache_bytes(n_layers: int, n_kv_heads: int, head_dim: int, seq_len: int,
                   batch: int = 1, bytes_per_elem: int = 2) -> int:
    """KV Cache 的字节数 = 2（K 与 V）· L · B · S · n_kv_heads · d_h · 每元素字节数。"""
    return 2 * n_layers * batch * seq_len * n_kv_heads * head_dim * bytes_per_elem
# endregion
