"""显式 Softmax Attention 与统一 MHA/MQA/GQA 实现。"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..cache import KVCache
from ..config import AttentionConfig
from ..layers.normalization import RMSNorm
from .patterns import attention_mask, masked_softmax
from .position import apply_rope


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    visible: torch.Tensor | None = None,
    dropout: float = 0.0,
    training: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Attention(Q,K,V) = softmax(QKᵀ / sqrt(Dk) + mask)V。

    Q [B,H,Q,Dk]、K [B,H,S,Dk]、V [B,H,S,Dv] -> [B,H,Q,Dv]。
    visible 为可广播到 [B,H,Q,S] 的 bool 张量，True 表示允许读取。
    原语没有参数；两次 matmul 约 2*B*H*Q*S*(Dk+Dv) FLOPs，
    scores/weights 占 O(B*H*Q*S) 空间。训练、prefill、decode 的数学相同，
    区别是调用者提供的 Q/S 长度及缓存；此函数自身不持有状态。
    """
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or q.shape[:2] != k.shape[:2]
        or k.shape[:3] != v.shape[:3]
        or q.shape[-1] != k.shape[-1]
    ):
        raise ValueError("incompatible attention shapes")
    if visible is None:
        visible = torch.ones((q.shape[-2], k.shape[-2]), dtype=torch.bool, device=q.device)
    if visible.dtype != torch.bool:
        raise ValueError("attention masks use True=visible")
    scores = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5 if scale is None else scale)
    weights = masked_softmax(scores, visible)
    return F.dropout(weights, dropout, training) @ v


def gathered_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Actually gather selected keys: indices [Q,M], no [Q,K] score allocation.

    Same key selection for all batch/head entries; valid [Q,M] masks fillers.
    """
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.ndim != 4
        or q.shape[:2] != k.shape[:2]
        or k.shape[:3] != v.shape[:3]
        or q.shape[-1] != k.shape[-1]
    ):
        raise ValueError("incompatible gathered attention shapes")
    if (
        indices.ndim != 2
        or indices.dtype != torch.long
        or indices.shape[0] != q.shape[-2]
        or indices.shape[1] < 1
    ):
        raise ValueError("indices must be nonempty long [Q,M]")
    if indices.device != q.device or indices.min() < 0 or indices.max() >= k.shape[-2]:
        raise ValueError("sparse index outside key sequence")
    if valid is None:
        valid = torch.ones_like(indices, dtype=torch.bool)
    if valid.shape != indices.shape or valid.dtype != torch.bool or valid.device != q.device:
        raise ValueError("valid must be boolean [Q,M]")
    # Duplicate real indices change normalization; filler duplicates are allowed.
    for row, mask in zip(indices, valid, strict=True):
        if row[mask].unique().numel() != mask.sum():
            raise ValueError("duplicate selected keys")
    selected_k, selected_v = k[:, :, indices, :], v[:, :, indices, :]
    scores = (q.unsqueeze(-2) * selected_k).sum(-1) * (
        q.shape[-1] ** -0.5 if scale is None else scale
    )
    return (masked_softmax(scores, valid).unsqueeze(-1) * selected_v).sum(-2)


class MultiHeadAttention(nn.Module):
    """统一 MHA/MQA/GQA，以及 self/cross、训练/prefill/decode。

    H 是 query head 数，G 是 KV head 数，d 是每头宽度：
    Q: [B,T,D] -> [B,H,T,d]；K/V: [B,S,D] -> [B,G,S,d]。
    MHA G=H，MQA G=1，GQA 1<G<H；缓存始终保留 G 个头。
    bias-free 投影共 2*D*d*(H+G) 参数，QK-Norm 额外增加 2*d。
    普通 KV 存储为 2*B*G*S*d 个元素；计算时临时扩展到 H 个头。
    成本的精确配置对照见 analysis.attention_cost。
    """

    def __init__(
        self, dim: int, config: AttentionConfig, dropout: float = 0.0, cross: bool = False
    ) -> None:
        super().__init__()
        self.config, self.dim, self.cross, self.dropout = config, dim, cross, dropout
        h, g, d = config.heads, config.effective_kv_heads, config.head_dim
        self.q = nn.Linear(dim, h * d, bias=False)
        self.k = nn.Linear(dim, g * d, bias=False)
        self.v = nn.Linear(dim, g * d, bias=False)
        self.output = nn.Linear(h * d, dim, bias=False)
        self.q_norm = RMSNorm(d) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(d) if config.qk_norm else nn.Identity()

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
        """x 仅包含本次 query；key_valid 覆盖完整 key 前缀。

        无 cache：训练/首次 prefill，从 x 或 encoder memory 投影 K/V。
        self cache：query_offset 必须等于缓存长度，再追加本次 K/V。
        cross cache：K/V 是静态 source 投影，decode 只更新 Q 的位置。
        use_cache 控制是否返回状态；状态不会原地修改。
        """
        c = self.config
        b, t, _ = x.shape
        h, g, d = c.heads, c.effective_kv_heads, c.head_dim
        if not self.cross and cache is None and query_offset != 0:
            raise ValueError("nonzero self-attention position requires a prefix cache")
        if cache is not None:
            if not isinstance(cache, KVCache):
                raise ValueError("expected a KVCache")
            cache.validate(b, g, d, d, x, self.cross)
            if not self.cross and query_offset != cache.length:
                raise ValueError("append position must equal cached prefix length")
        q = self.q_norm(self.q(x).reshape(b, t, h, d).transpose(1, 2))
        qp = torch.arange(query_offset, query_offset + t, device=x.device)
        if c.position.kind == "rope":
            q = apply_rope(q, qp, c.position)
        # Cross 的 S 来自输入序列，T 来自输出序列，两侧 RoPE 都从自身位置计数。
        # 静态 Cross cache 不必在每个输出 token 上重算 encoder 的 K/V。
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
                raise ValueError("cross attention requires compatible encoder memory")
            s = source.shape[1]
            k = self.k_norm(self.k(source).reshape(b, s, g, d).transpose(1, 2))
            v = self.v(source).reshape(b, s, g, d).transpose(1, 2)
            kp = torch.arange(
                0 if self.cross else query_offset,
                (0 if self.cross else query_offset) + s,
                device=x.device,
            )
            if c.position.kind == "rope":
                k = apply_rope(k, kp, c.position)
            next_cache = KVCache(k, v, self.cross) if cache is None else cache.append(k, v)
        k, v = next_cache.key, next_cache.value
        visible = attention_mask(
            qp, torch.arange(next_cache.length, device=x.device), c, causal, key_valid
        )
        # ponytail: expanded KV is temporary; grouped GPU kernels avoid the H/G replication.
        k, v = (a.repeat_interleave(h // g, 1) for a in (k, v))
        # attention_factor 同时缩放 Q/K 的振幅，等价于 logits 乘其平方。
        scale = d**-0.5 * c.position.attention_factor**2
        if c.backend == "sdpa":
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=visible,
                dropout_p=self.dropout if self.training else 0.0,
                scale=scale,
            )
        else:
            y = scaled_dot_product_attention(q, k, v, visible, self.dropout, self.training, scale)
        return self.output(y.transpose(1, 2).reshape(b, t, -1)), next_cache if use_cache else None
