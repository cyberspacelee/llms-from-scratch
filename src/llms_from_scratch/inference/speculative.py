"""投机解码：接受-拒绝采样、期望接受长度、树形注意力掩码，以及基于共享 GPT 的草稿-验证生成。"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from llms_from_scratch.transformer.model import GPT, KVCache


# region accept
def residual_distribution(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """拒绝后重采样所用的分布 norm(max(0, p − q))。p == q 时永不拒绝，此时随便返回 p。"""
    r = (p - q).clamp(min=0)
    total = r.sum(-1, keepdim=True)
    return torch.where(total > 0, r / total.clamp(min=1e-30), p)


def accept_or_resample(p: torch.Tensor, q: torch.Tensor, x: torch.Tensor,
                       generator: torch.Generator | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """批量地对一个位置做接受-拒绝。p、q: [B, V]；x: [B] 是从 q 采出的草稿 token。

    以概率 min(1, p(x)/q(x)) 接受 x，否则从残差分布重采样。返回 (是否接受, 最终 token)。
    """
    px = p.gather(-1, x[:, None])[:, 0]
    qx = q.gather(-1, x[:, None])[:, 0]
    u = torch.rand(x.shape, generator=generator, device=p.device)
    accepted = u * qx < px  # 等价于 u < p(x)/q(x)，避免除零
    fallback = torch.multinomial(residual_distribution(p, q), 1, generator=generator)[:, 0]
    return accepted, torch.where(accepted, x, fallback)
# endregion accept


def verify(target_probs: torch.Tensor, draft_probs: torch.Tensor, draft_tokens: list[int],
           generator: torch.Generator | None = None) -> list[int]:
    """对 k 个草稿 token 逐个接受-拒绝；全部接受时再从 target_probs[k] 采一个“奖励” token。

    target_probs: [k+1, V]，draft_probs: [k, V]。返回本轮被采纳的 token（1 到 k+1 个）。
    """
    out = []
    for i, tok in enumerate(draft_tokens):
        ok, final = accept_or_resample(target_probs[i:i + 1], draft_probs[i:i + 1],
                                       torch.tensor([tok], device=target_probs.device), generator)
        out.append(int(final))
        if not ok:
            return out
    bonus = torch.multinomial(target_probs[len(draft_tokens)], 1, generator=generator)
    return out + [int(bonus)]


# region expected
def acceptance_rate(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """单个位置的接受概率 α = Σ_x min(p(x), q(x)) = 1 − TV(p, q)。"""
    return torch.minimum(p, q).sum(-1)


def expected_tokens_per_step(alpha: float, k: int) -> float:
    """每轮验证平均产出的 token 数 (1 − α^{k+1}) / (1 − α)（假设各位置独立、接受率相同）。"""
    return k + 1.0 if alpha == 1 else (1 - alpha ** (k + 1)) / (1 - alpha)


def expected_speedup(alpha: float, k: int, c: float) -> float:
    """相对普通解码的加速比：一轮耗时 = k 次草稿（每次 c 倍目标模型步长）+ 1 次验证。"""
    return expected_tokens_per_step(alpha, k) / (k * c + 1)
# endregion expected


# region tree
def tree_attention_mask(parents: list[int]) -> torch.Tensor:
    """树形草稿的注意力掩码。parents[i] 是节点 i 的父节点下标（-1 表示接在已确认前缀之后）。

    节点 i 只能看到自己与祖先：mask[i, j] = True 当且仅当 j 在从根到 i 的路径上。
    """
    n = len(parents)
    mask = torch.zeros(n, n, dtype=torch.bool)
    for i in range(n):
        j = i
        while j != -1:
            mask[i, j] = True
            j = parents[j]
    return mask


def tree_depths(parents: list[int]) -> list[int]:
    """每个节点的深度，用作它的位置偏移（RoPE 位置 = 前缀长度 + 深度）。"""
    depth = []
    for p in parents:
        depth.append(0 if p == -1 else depth[p] + 1)
    return depth
# endregion tree


@dataclass
class SpecStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0

    @property
    def tokens_per_round(self) -> float:
        return (self.accepted + self.rounds) / max(1, self.rounds)


def _probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.softmax(logits.float() / temperature, -1)


# region generate
@torch.no_grad()
def speculative_generate(target: GPT, draft: GPT, idx: torch.Tensor, max_new_tokens: int,
                         k: int = 4, temperature: float = 0.0,
                         generator: torch.Generator | None = None
                         ) -> tuple[torch.Tensor, SpecStats]:
    """草稿模型每轮自回归地提议 k 个 token，目标模型一次前向验证 k+1 个位置。

    temperature == 0 时用贪心规则（草稿 token 等于目标 argmax 才接受），输出与
    target.generate(temperature=0) 完全一致；否则用接受-拒绝采样，输出分布与目标模型一致。
    两个 KV Cache 都只“回滚”长度：被拒绝位置的 K/V 留在缓存里，下次写入时被覆盖。
    """
    assert idx.shape[0] == 1, "示例实现只处理单个序列"
    ctx = target.config.context_length
    seq = idx[0].tolist()
    t_cache = KVCache(target.config, 1, device=idx.device, dtype=target.embed.weight.dtype)
    d_cache = KVCache(draft.config, 1, device=idx.device, dtype=draft.embed.weight.dtype)
    t_len = d_len = 0  # 两个缓存中有效（已写入且仍在序列里）的位置数
    produced, stats = 0, SpecStats()
    as_input = lambda toks: torch.tensor([toks], device=idx.device)  # noqa: E731
    while produced < max_new_tokens and len(seq) < ctx:
        kk = max(0, min(k, max_new_tokens - produced - 1, ctx - len(seq)))
        # 1. 草稿：先补上缓存里缺的 token，再逐个提议
        drafts, q_probs = [], []
        feed = seq[d_len:]
        for _ in range(kk):
            logits = draft(as_input(feed), d_cache, len(seq) + len(drafts) - len(feed))[0, -1]
            if temperature == 0:
                tok = int(logits.argmax())
            else:
                q = _probs(logits, temperature)
                tok = int(torch.multinomial(q, 1, generator=generator))
                q_probs.append(q)
            drafts.append(tok)
            feed = [tok]
        # 2. 验证：目标模型一次前向算出 kk+1 个位置的分布
        logits = target(as_input(seq[t_len:] + drafts), t_cache, t_len)[0, -(kk + 1):]
        if temperature == 0:
            choice = logits.argmax(-1).tolist()
            m = 0
            while m < kk and drafts[m] == choice[m]:
                m += 1
            new = drafts[:m] + [choice[m]]  # 前 m 个被接受，再加目标模型自己的下一个 token
        else:
            new = verify(_probs(logits, temperature), torch.stack(q_probs) if q_probs else
                         logits.new_zeros(0, logits.shape[-1]), drafts, generator)
            m = len(new) - 1
        # 3. 更新序列与两个缓存的有效长度
        before = len(seq)
        t_len = before + m
        d_len = before + min(m, max(kk - 1, 0)) if kk > 0 else d_len
        new = new[:max_new_tokens - produced][:ctx - len(seq)]
        seq += new
        produced += len(new)
        stats.rounds += 1
        stats.drafted += kk
        stats.accepted += m
    return torch.tensor([seq], device=idx.device), stats
# endregion generate
