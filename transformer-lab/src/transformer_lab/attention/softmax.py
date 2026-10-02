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

    Q [B,H_q,S_q,D_h]、K [B,H_q,S_kv,D_h]、V [B,H_q,S_kv,D_v] -> [B,H_q,S_q,D_v]。
    visible 为可广播到 [B,H_q,S_q,S_kv] 的 bool 张量，True 表示允许读取。
    原语没有参数；两次 matmul 约 2*B*H_q*S_q*S_kv*(D_h+D_v) FLOPs，
    scores/weights 占 O(B*H_q*S_q*S_kv) 空间。训练、prefill、decode 的数学相同，
    区别是调用者提供的 S_q/S_kv 长度及缓存；此函数自身不持有状态。

    Args:
        q: float Q[B,H_q,S_q,D_h]。
        k: float K[B,H_q,S_kv,D_h]。
        v: float V[B,H_q,S_kv,D_v]。
        visible: 可选 bool [S_q,S_kv] 或 [B,1,S_q,S_kv]，可广播到 scores；True=可见。
        dropout: 训练期 dropout 概率，推理期关闭。
        training: 是否启用训练期 dropout。
        scale: 可选 Attention logits 缩放因子，None 使用 1/sqrt(D_h)。

    Returns:
        float context[B,H_q,S_q,D_v]，与 V 的 dtype 相同。
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
    # K transpose: [B,H_q,S_kv,D_h] -> [B,H_q,D_h,S_kv]；scores[B,H_q,S_q,S_kv]。
    scores = (q @ k.transpose(-1, -2)) * (q.shape[-1] ** -0.5 if scale is None else scale)
    weights = masked_softmax(scores, visible)  # [B,H_q,S_q,S_kv]，按 key 轴归一化
    return F.dropout(weights, dropout, training) @ v  # [B,H_q,S_q,D_v]


def gathered_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    valid: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Actually gather selected keys: indices [S_q,M], no [S_q,S_kv] score allocation.

    Same key selection for all batch/head entries; valid [S_q,M] masks fillers.

    Args:
        q: float Q[B,H_q,S_q,D_h]。
        k: float K[B,H_q,S_kv,D_h]。
        v: float V[B,H_q,S_kv,D_v]。
        indices: long [S_q,M]，每个 query 选择的 key 下标。
        valid: 可选 bool [S_q,M]，屏蔽 filler。
        scale: 可选 Attention logits 缩放因子，None 使用 1/sqrt(D_h)。

    Returns:
        float context[B,H_q,S_q,D_v]；只分配 [B,H_q,S_q,M] scores。
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
        raise ValueError("indices must be nonempty long [S_q,M]")
    if indices.device != q.device or indices.min() < 0 or indices.max() >= k.shape[-2]:
        raise ValueError("sparse index outside key sequence")
    if valid is None:
        valid = torch.ones_like(indices, dtype=torch.bool)
    if valid.shape != indices.shape or valid.dtype != torch.bool or valid.device != q.device:
        raise ValueError("valid must be boolean [S_q,M]")
    # Duplicate real indices change normalization; filler duplicates are allowed.
    for row, mask in zip(indices, valid, strict=True):
        if row[mask].unique().numel() != mask.sum():
            raise ValueError("duplicate selected keys")
    # gather: K/V[B,H_q,S_kv,D_h/D_v] -> [B,H_q,S_q,M,D_h/D_v]。
    selected_k, selected_v = k[:, :, indices, :], v[:, :, indices, :]
    scores = (q.unsqueeze(-2) * selected_k).sum(-1) * (
        q.shape[-1] ** -0.5 if scale is None else scale
    )
    return (masked_softmax(scores, valid).unsqueeze(-1) * selected_v).sum(-2)


class MultiHeadAttention(nn.Module):
    """统一 MHA/MQA/GQA，以及 self/cross、训练/prefill/decode。

    H_q 是 query head 数，H_kv 是 KV head 数，D_h 是每头宽度：
    Q: [B,S_q,D] -> [B,H_q,S_q,D_h]；K/V: [B,S_kv,D] -> [B,H_kv,S_kv,D_h]。
    MHA H_kv=H_q，MQA H_kv=1，GQA 1<H_kv<H_q；cache保留H_kv个头。
    bias-free 投影共 2*D*D_h*(H_q+H_kv) 参数，QK-Norm 额外增加 2*D_h。
    普通 KV 存储为 2*B*H_kv*S_kv*D_h 个元素；计算时临时扩展到 H_q 个头。
    成本的精确配置对照见 analysis.attention_cost。
    """

    def __init__(
        self, dim: int, config: AttentionConfig, dropout: float = 0.0, cross: bool = False
    ) -> None:
        """根据尺寸和配置创建参数/子层。

        Args:
            dim: 输入/输出 hidden width D；不要求等于 Attention 投影宽度。
            config: 本模块的显式配置对象。
            dropout: 训练期 dropout 概率，推理期关闭。
            cross: 是否使用 Encoder memory 作为 K/V 来源。

        Returns:
            None；参数与子层注册在 self 中。
        """
        super().__init__()
        self.config, self.dim, self.cross, self.dropout = config, dim, cross, dropout
        num_heads, num_kv_heads, head_dim = config.heads, config.effective_kv_heads, config.head_dim
        self.q = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.output = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.q_norm = RMSNorm(head_dim) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if config.qk_norm else nn.Identity()

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

        Args:
            x: float [B,S_q,D] 当前 query hidden。
            memory: Cross 的可选 float [B,S_kv,D]。
            cache: 已有单层/模型状态；None 表示没有历史。
            use_cache: 是否返回新状态；不原地修改既有缓存。
            query_offset: query 绝对位置起点 P；Self cache 时等于前缀长度。
            causal: 是否限制 key 绝对位置不超过 query。
            key_valid: 可选 bool [B,S_kv]，True 表示有效，含完整历史。

        Returns:
            tuple: output[B,S_q,D]，KVCache 或 None；K/V[B,H_kv,S_kv,D_h]。
        """
        config = self.config
        batch_size, query_len, _ = x.shape
        num_heads, num_kv_heads, head_dim = config.heads, config.effective_kv_heads, config.head_dim
        if not self.cross and cache is None and query_offset != 0:
            raise ValueError("nonzero self-attention position requires a prefix cache")
        if cache is not None:
            if not isinstance(cache, KVCache):
                raise ValueError("expected a KVCache")
            cache.validate(batch_size, num_kv_heads, head_dim, head_dim, x, self.cross)
            if not self.cross and query_offset != cache.length:
                raise ValueError("append position must equal cached prefix length")
        # Q projection[B,S_q,H_q*D_h] -> reshape[B,S_q,H_q,D_h] -> transpose[B,H_q,S_q,D_h]。
        q = self.q_norm(
            self.q(x).reshape(batch_size, query_len, num_heads, head_dim).transpose(1, 2)
        )
        qp = torch.arange(query_offset, query_offset + query_len, device=x.device)
        if config.position.kind == "rope":
            q = apply_rope(q, qp, config.position)
        # Cross 的 S_kv 来自输入侧，S_q 来自输出侧，两侧 RoPE 用各自位置。
        # 静态 Cross cache 不必在每个输出 token 上重算 encoder 的 K/V。
        if self.cross and cache is not None:
            next_cache = cache
        else:
            source = memory if self.cross else x
            if (
                source is None
                or source.ndim != 3
                or source.shape[0] != batch_size
                or source.shape[-1] != self.dim
            ):
                raise ValueError("cross attention requires compatible encoder memory")
            key_len = source.shape[1]
            # source[B,S_new,D] -> K/V[B,H_kv,S_new,D_h]，Self的S_new=S_q。
            k = self.k_norm(
                self.k(source).reshape(batch_size, key_len, num_kv_heads, head_dim).transpose(1, 2)
            )
            v = self.v(source).reshape(batch_size, key_len, num_kv_heads, head_dim).transpose(1, 2)
            kp = torch.arange(
                0 if self.cross else query_offset,
                (0 if self.cross else query_offset) + key_len,
                device=x.device,
            )
            if config.position.kind == "rope":
                k = apply_rope(k, kp, config.position)
            # Self append在sequence轴拼接P与S_q；静态Cross只在首次投影S_kv。
            next_cache = KVCache(k, v, self.cross) if cache is None else cache.append(k, v)
        k, v = next_cache.key, next_cache.value
        visible = attention_mask(
            qp, torch.arange(next_cache.length, device=x.device), config, causal, key_valid
        )
        # [B,H_kv,S_kv,D_h] -> [B,H_q,S_kv,D_h]；持久cache仍只有H_kv。
        # ponytail: expanded KV is temporary; grouped GPU kernels avoid H_q/H_kv replication.
        k, v = (a.repeat_interleave(num_heads // num_kv_heads, 1) for a in (k, v))
        # attention_factor 同时缩放 Q/K 的振幅，等价于 logits 乘其平方。
        scale = head_dim**-0.5 * config.position.attention_factor**2
        if config.backend == "sdpa":
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
        # context[B,H_q,S_q,D_h] -> [B,S_q,H_q*D_h] -> output[B,S_q,D]。
        return self.output(
            y.transpose(1, 2).reshape(batch_size, query_len, -1)
        ), next_cache if use_cache else None
