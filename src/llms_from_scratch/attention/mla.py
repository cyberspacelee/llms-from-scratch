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
    输入 [B,T_q,D] 经 down-projection + RMSNorm 得 C[B,1,T_q,L_kv]；
    单独投影得到 Kr[B,1,T_q,D_r]，只对 Kr 和 Qr 应用 RoPE。
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
        num_heads, head_dim, rope_dim, kv_rank = (
            config.heads,
            config.head_dim,
            config.rope_dim,
            config.kv_rank,
        )
        if config.q_rank:
            self.q_down = nn.Linear(dim, config.q_rank, bias=False)
            self.q_norm = RMSNorm(config.q_rank)
            self.q_up = nn.Linear(config.q_rank, num_heads * (head_dim + rope_dim), bias=False)
        else:
            self.q_down = nn.Identity()
            self.q_norm = nn.Identity()
            self.q_up = nn.Linear(dim, num_heads * (head_dim + rope_dim), bias=False)
        self.kv_down = nn.Linear(dim, kv_rank, bias=False)
        self.kv_norm = RMSNorm(kv_rank)
        self.k_rope = nn.Linear(dim, rope_dim, bias=False)
        self.k_up = nn.Linear(kv_rank, num_heads * head_dim, bias=False)
        self.v_up = nn.Linear(kv_rank, num_heads * config.value_dim, bias=False)
        self.output = nn.Linear(num_heads * config.value_dim, dim, bias=False)

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
        """返回 [B,T_q,D] 与可选 latent cache；offset/static 约定与 MHA 相同。

        Args:
            x: float [B,T_q,D]。
            memory: Cross 的可选 float [B,T_kv,D]。
            cache: 已有单层/模型状态；None 表示没有历史。
            use_cache: 是否返回新状态；不原地修改既有缓存。
            query_offset: query 绝对位置起点 P；Self cache 时等于前缀长度。
            causal: 是否限制 key 绝对位置不超过 query。
            key_valid: 可选 bool [B,T_kv]，True 表示有效，含完整历史。

        Returns:
            tuple: output[B,T_q,D]，latent KVCache 或 None；C[B,1,T_kv,L_kv]、Kr[B,1,T_kv,D_r]。
        """
        config = self.config
        batch_size, query_len, _ = x.shape
        if not self.cross and cache is None and query_offset != 0:
            raise ValueError("nonzero MLA position requires a prefix cache")
        if cache is not None:
            if not isinstance(cache, KVCache):
                raise ValueError("expected a latent KVCache")
            cache.validate(batch_size, 1, config.kv_rank, config.rope_dim, x, self.cross)
            if not self.cross and query_offset != cache.length:
                raise ValueError("append position must equal cached prefix length")
        # query down[B,T_q,L_q] -> up[B,T_q,H_q*(D_h+D_r)] -> split heads。
        q = (
            self.q_up(self.q_norm(self.q_down(x)))
            .reshape(batch_size, query_len, config.heads, config.head_dim + config.rope_dim)
            .transpose(1, 2)
        )
        # 不能把普通 RoPE 施加到压缩后的内容 latent：位置旋转会阻碍矩阵吸收。
        # Qc[B,H_q,T_q,D_h] 与 Qr[B,H_q,T_q,D_r]；只旋转Qr。
        qc, qr = q.split((config.head_dim, config.rope_dim), -1)
        qp = torch.arange(query_offset, query_offset + query_len, device=x.device)
        if config.position.kind == "rope":
            qr = apply_rope(qr, qp, config.position)
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
                raise ValueError("cross MLA requires compatible encoder memory")
            # key/value 字段在 MLA 中分别装 C 和 Kr；value 字段不是展开后的 V。
            # source[B,T_new,D] -> C[B,1,T_new,L_kv]，Kr[B,1,T_new,D_r]。
            latent = self.kv_norm(self.kv_down(source)).unsqueeze(1)
            kr = self.k_rope(source).unsqueeze(1)
            start = 0 if self.cross else query_offset
            kp = torch.arange(start, start + source.shape[1], device=x.device)
            if config.position.kind == "rope":
                kr = apply_rope(kr, kp, config.position)
            # Self在sequence轴追加为T_kv=P+T_q；Cross保持静态source长度。
            next_cache = (
                KVCache(latent, kr, self.cross) if cache is None else cache.append(latent, kr)
            )
        latent, kr = next_cache.key, next_cache.value
        key_len = next_cache.length
        visible = attention_mask(
            qp, torch.arange(key_len, device=x.device), config, causal, key_valid
        )
        scale = (config.head_dim + config.rope_dim) ** -0.5 * config.position.attention_factor**2
        if config.mla_impl == "naive":
            # Kc=C@W_UKᵀ，V=C@W_UVᵀ。每个 query 对内容和位置两项之和做 softmax。
            kc = (
                self.k_up(latent[:, 0])
                .reshape(batch_size, key_len, config.heads, config.head_dim)
                .transpose(1, 2)
            )
            v = (
                self.v_up(latent[:, 0])
                .reshape(batch_size, key_len, config.heads, config.value_dim)
                .transpose(1, 2)
            )
            # Kc[B,H_q,T_kv,D_h]、V[B,H_q,T_kv,D_v]；scores[B,H_q,T_q,T_kv]。
            scores = (qc @ kc.transpose(-1, -2) + qr @ kr.transpose(-1, -2)) * scale
            p = F.dropout(masked_softmax(scores, visible), self.dropout, self.training)
            y = p @ v  # [B,H_q,T_q,D_v]
            y = self.output(y.transpose(1, 2).reshape(batch_size, query_len, -1))
        else:
            # qc@(C@W_UKᵀ)ᵀ = (qc@W_UK)@Cᵀ，省掉 [B,H_q,T_kv,D_h] 的重建。
            wk = self.k_up.weight.reshape(config.heads, config.head_dim, config.kv_rank)
            q_latent = torch.matmul(qc, wk)  # [B,H_q,T_q,L_kv]
            scores = (q_latent @ latent.transpose(-1, -2) + qr @ kr.transpose(-1, -2)) * scale
            p = F.dropout(masked_softmax(scores, visible), self.dropout, self.training)
            context = p @ latent  # [B,H_q,T_q,L_kv]，在 latent 空间汇聚 value 信息。
            wv = self.v_up.weight.reshape(config.heads, config.value_dim, config.kv_rank)
            wo = self.output.weight.T.reshape(config.heads, config.value_dim, self.dim)
            # (P@C)@W_UVᵀ@W_Oᵀ：先合并静态线性映射，再求各头的输出之和。
            # 从当前参数重新计算，避免 optimizer 更新后保留过期的吸收矩阵。
            wvo = torch.matmul(wv.transpose(-1, -2), wo)  # [H_q,L_kv,D]
            y = torch.matmul(context, wvo).sum(1)  # [B,H_q,T_q,D] -> [B,T_q,D]
        return y, next_cache if use_cache else None
