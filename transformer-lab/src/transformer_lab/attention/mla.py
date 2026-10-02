"""MLA：归一化 KV latent、解耦 RoPE 与矩阵吸收。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..cache import KVCache
from ..config import AttentionConfig
from ..layers.normalization import RMSNorm
from .patterns import attention_mask, masked_softmax
from .position import apply_rope


class MultiHeadLatentAttention(nn.Module):
    """DeepSeek 风格的归一化 KV latent + 共享的解耦 rotary key。

    内容维 c=head_dim，rotary 维 r=rope_dim，latent 维 L=kv_rank。
    输入 [B,T,D] 经 down-projection + RMSNorm 得 C[B,1,T,L]；
    单独投影得到 Kr[B,1,T,r]，只对 Kr 和 Qr 应用 RoPE。
    每 token 缓存 L+r 个元素，而不是每头分别保存 K/V。
    Query 可先压缩到 q_rank 再上投影到 H*(c+r)，q_rank=0 表示直接投影。

    naive 从 C 重建每头 Kc/V；absorbed 把 W_UK 移到 query 侧，
    把 W_UV 与 W_O 合并，使 decode 全程读取 latent。两条路径共享参数
    和 cache，且都保留 autograd。训练通常适合展开后的 matmul；本实现
    允许两条路径用于所有阶段，以检查等价性，不宣称谁在 CPU 上更快。
    参数/FLOPs/缓存账本见 analysis.attention_cost 和 docs/principles.md。
    """

    def __init__(
        self, dim: int, config: AttentionConfig, dropout: float = 0.0, cross: bool = False
    ) -> None:
        super().__init__()
        self.config, self.dim, self.cross, self.dropout = config, dim, cross, dropout
        h, d, r, rank = config.heads, config.head_dim, config.rope_dim, config.kv_rank
        if config.q_rank:
            self.q_down = nn.Linear(dim, config.q_rank, bias=False)
            self.q_norm = RMSNorm(config.q_rank)
            self.q_up = nn.Linear(config.q_rank, h * (d + r), bias=False)
        else:
            self.q_down = nn.Identity()
            self.q_norm = nn.Identity()
            self.q_up = nn.Linear(dim, h * (d + r), bias=False)
        self.kv_down = nn.Linear(dim, rank, bias=False)
        self.kv_norm = RMSNorm(rank)
        self.k_rope = nn.Linear(dim, r, bias=False)
        self.k_up = nn.Linear(rank, h * d, bias=False)
        self.v_up = nn.Linear(rank, h * config.value_dim, bias=False)
        self.output = nn.Linear(h * config.value_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        cache: KVCache | None = None,
        use_cache: bool = False,
        query_offset: int = 0,
        causal: bool = False,
        key_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, KVCache | None]:
        """返回 [B,T,D] 与可选 latent cache；offset/static 约定与 MHA 相同。"""
        c = self.config
        b, t, _ = x.shape
        if not self.cross and cache is None and query_offset != 0:
            raise ValueError("nonzero MLA position requires a prefix cache")
        if cache is not None:
            if not isinstance(cache, KVCache):
                raise ValueError("expected a latent KVCache")
            cache.validate(b, 1, c.kv_rank, c.rope_dim, x, self.cross)
            if not self.cross and query_offset != cache.length:
                raise ValueError("append position must equal cached prefix length")
        q = (
            self.q_up(self.q_norm(self.q_down(x)))
            .reshape(b, t, c.heads, c.head_dim + c.rope_dim)
            .transpose(1, 2)
        )
        # 不能把普通 RoPE 施加到压缩后的内容 latent：位置旋转会阻碍矩阵吸收。
        qc, qr = q.split((c.head_dim, c.rope_dim), -1)
        qp = torch.arange(query_offset, query_offset + t, device=x.device)
        if c.position.kind == "rope":
            qr = apply_rope(qr, qp, c.position)
        if self.cross and cache is not None:
            next_cache = cache
        else:
            source = memory if self.cross else x
            if (
                source is None
                or source.ndim != 3
                or source.shape[0] != b
                or source.shape[-1] != self.dim
            ):
                raise ValueError("cross MLA requires compatible encoder memory")
            # key/value 字段在 MLA 中分别装 C 和 Kr；value 字段不是展开后的 V。
            latent = self.kv_norm(self.kv_down(source)).unsqueeze(1)
            kr = self.k_rope(source).unsqueeze(1)
            start = 0 if self.cross else query_offset
            kp = torch.arange(start, start + source.shape[1], device=x.device)
            if c.position.kind == "rope":
                kr = apply_rope(kr, kp, c.position)
            next_cache = (
                KVCache(latent, kr, self.cross) if cache is None else cache.append(latent, kr)
            )
        latent, kr = next_cache.key, next_cache.value
        s = next_cache.length
        visible = attention_mask(qp, torch.arange(s, device=x.device), c, causal, key_valid)
        scale = (c.head_dim + c.rope_dim) ** -0.5 * c.position.attention_factor**2
        if c.mla_impl == "naive":
            # Kc=C@W_UKᵀ，V=C@W_UVᵀ。每个 query 对内容和位置两项之和做 softmax。
            kc = self.k_up(latent[:, 0]).reshape(b, s, c.heads, c.head_dim).transpose(1, 2)
            v = self.v_up(latent[:, 0]).reshape(b, s, c.heads, c.value_dim).transpose(1, 2)
            scores = (qc @ kc.transpose(-1, -2) + qr @ kr.transpose(-1, -2)) * scale
            p = F.dropout(masked_softmax(scores, visible), self.dropout, self.training)
            y = p @ v
            y = self.output(y.transpose(1, 2).reshape(b, t, -1))
        else:
            # qc@(C@W_UKᵀ)ᵀ = (qc@W_UK)@Cᵀ，省掉 [B,H,S,c] 的重建。
            wk = self.k_up.weight.reshape(c.heads, c.head_dim, c.kv_rank)
            q_latent = torch.matmul(qc, wk)
            scores = (q_latent @ latent.transpose(-1, -2) + qr @ kr.transpose(-1, -2)) * scale
            p = F.dropout(masked_softmax(scores, visible), self.dropout, self.training)
            context = p @ latent  # [B,H,T,L]，在 latent 空间汇聚 value 信息。
            wv = self.v_up.weight.reshape(c.heads, c.value_dim, c.kv_rank)
            wo = self.output.weight.T.reshape(c.heads, c.value_dim, self.dim)
            # (P@C)@W_UVᵀ@W_Oᵀ：先合并静态线性映射，再求各头的输出之和。
            # 从当前参数重新计算，避免 optimizer 更新后保留过期的吸收矩阵。
            wvo = torch.matmul(wv.transpose(-1, -2), wo)  # [H,L,D]
            y = torch.matmul(context, wvo).sum(1)
        return y, next_cache if use_cache else None
