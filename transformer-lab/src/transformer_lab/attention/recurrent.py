"""Token-by-token reference for linear attention and the (gated) delta rule."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..cache import RecurrentCache
from ..config import AttentionConfig


class RecurrentAttention(nn.Module):
    """用固定矩阵状态代替随长度增长的 KV 序列；只支持因果 self-attention。

    Linear: S+=phi(k)⊗v，z+=phi(k)，y=phi(q)ᵀS/(phi(q)ᵀz)。
    Delta: S+=beta*k⊗(v-kᵀS)，用预测误差更新关联记忆。
    Gated Delta: 先 S'=alpha*S，再 S=S'+beta*k⊗(v-kᵀS')。
    Delta 的 Q/K 做单位范数归一化；beta/alpha 是每 token、每 head 的标量。
    这是原理递归，不包含完整 KDA 的逐通道门或专用 chunkwise kernel。

    输入 [B,S,D]，输出同形状；state=[B,H_q,D_h,D_v]，linear 额外 z=[B,H_q,D_h]。
    训练/prefill 顺序遍历 S，decode 只更新一个 token，状态空间 O(BHdv)。
    所有步骤保留梯度；长序列训练仍需保存反向中间量，不能把推理状态
    的固定大小解释成训练总内存固定。精确成本见 analysis.attention_cost。
    """

    def __init__(self, dim: int, config: AttentionConfig) -> None:
        """根据尺寸和配置创建参数/子层。

        Args:
            dim: 输入/输出 hidden width D；不要求等于 Attention 投影宽度。
            config: 本模块的显式配置对象。

        Returns:
            None；参数与子层注册在 self 中。
        """
        super().__init__()
        self.config = config
        num_heads, key_dim = config.heads, config.head_dim
        self.q = nn.Linear(dim, num_heads * key_dim, bias=False)
        self.k = nn.Linear(dim, num_heads * key_dim, bias=False)
        self.v = nn.Linear(dim, num_heads * config.value_dim, bias=False)
        self.beta = nn.Linear(dim, num_heads) if config.kind != "linear" else None
        self.decay = nn.Linear(dim, num_heads) if config.kind == "gated_delta" else None
        self.output = nn.Linear(num_heads * config.value_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        cache: RecurrentCache | None = None,
        use_cache: bool = False,
        query_offset: int = 0,
        causal: bool = True,
        key_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RecurrentCache | None]:
        """执行本模块的前向计算；输入与输出 shape 约定如下。

        Args:
            x: float [B,S,D] 当前 chunk。
            memory: 必须为 None；递归实现只支持 Self。
            cache: 已有单层/模型状态；None 表示没有历史。
            use_cache: 是否返回新状态；不原地修改既有缓存。
            query_offset: query 绝对位置起点 P；Self cache 时等于前缀长度。
            causal: 是否限制 key 绝对位置不超过 query。
            key_valid: 可选 bool [B,P+S] 含完整历史。

        Returns:
            tuple: output[B,S,D]，RecurrentCache 或 None；state[B,H_q,D_h,D_v]。
        """
        if not causal or memory is not None:
            raise ValueError("recurrent reference supports causal self attention only")
        config = self.config
        batch_size, seq_len, _ = x.shape
        shape = (batch_size, config.heads, config.head_dim, config.value_dim)
        if cache is None and query_offset != 0:
            raise ValueError("nonzero recurrent position requires a state cache")
        if cache is not None:
            if not isinstance(cache, RecurrentCache):
                raise ValueError("expected a RecurrentCache")
            if (
                cache.state.shape != shape
                or cache.state.device != x.device
                or cache.state.dtype != x.dtype
                or cache.length != query_offset
            ):
                raise ValueError("invalid recurrent state or position")
            if config.kind == "linear" and (
                cache.normalizer is None
                or cache.normalizer.shape != shape[:-1]
                or cache.normalizer.device != x.device
                or cache.normalizer.dtype != x.dtype
            ):
                raise ValueError("invalid linear normalizer state")
            if config.kind != "linear" and cache.normalizer is not None:
                raise ValueError("delta state has no normalizer")
        state = x.new_zeros(shape) if cache is None else cache.state
        normalizer = x.new_zeros(shape[:-1]) if cache is None else cache.normalizer

        def project(layer: nn.Linear, width: int) -> torch.Tensor:
            """将 captured hidden 投影、拆头、转置。

            Args:
                layer: 应用于当前 hidden 的 nn.Linear 投影。
                width: 每个 head 投影宽度。

            Returns:
                float [B,H_q,S,width] 投影并拆头的 tensor。
            """
            return layer(x).reshape(batch_size, seq_len, config.heads, width).transpose(1, 2)

        q, k, v = (
            project(self.q, config.head_dim),
            project(self.k, config.head_dim),
            project(self.v, config.value_dim),
        )
        if config.kind == "linear":
            q, k = F.elu(q) + 1, F.elu(k) + 1
        else:
            q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        beta = self.beta(x).sigmoid().transpose(1, 2) if self.beta is not None else None
        decay = self.decay(x).sigmoid().transpose(1, 2) if self.decay is not None else None
        if key_valid is not None and (
            key_valid.shape != (batch_size, query_offset + seq_len)
            or key_valid.dtype != torch.bool
            or key_valid.device != x.device
        ):
            raise ValueError("key_valid must cover the complete recurrent prefix")
        outputs = []
        # ponytail: sequential recurrence demonstrates fixed state; chunkwise parallel kernels accelerate training.
        for i in range(seq_len):
            ki, qi, vi = k[:, :, i], q[:, :, i], v[:, :, i]  # [B,H_q,D_h/D_v]
            active = (
                torch.ones((batch_size, 1, 1, 1), device=x.device, dtype=torch.bool)
                if key_valid is None
                else key_valid[:, query_offset + i, None, None, None]
            )
            if config.kind == "linear":
                # [B,H_q,D_h,1] * [B,H_q,1,D_v] -> outer product[B,H_q,D_h,D_v]。
                updated = state + ki.unsqueeze(-1) * vi.unsqueeze(-2)
                z = normalizer + ki
                normalizer = torch.where(active.squeeze(-1), z, normalizer)
            else:
                # 先衰减旧记忆，再估计旧状态在当前 key 上的 value；只写入残差。
                decayed = state if decay is None else decay[:, :, i, None, None] * state
                residual = vi - (ki.unsqueeze(-2) @ decayed).squeeze(-2)
                updated = decayed + beta[:, :, i, None, None] * ki.unsqueeze(
                    -1
                ) * residual.unsqueeze(-2)
            state = torch.where(active, updated, state)
            yi = (qi.unsqueeze(-2) @ state).squeeze(-2)  # [B,H_q,1,D_v] -> [B,H_q,D_v]
            if config.kind == "linear":
                yi = yi / (qi * normalizer).sum(-1, keepdim=True).clamp_min(1e-6)
            outputs.append(yi * active.squeeze(-1))
        # stack[B,H_q,S,D_v] -> transpose[B,S,H_q,D_v] -> merge[B,S,H_q*D_v]。
        y = torch.stack(outputs, 2).transpose(1, 2).reshape(batch_size, seq_len, -1)
        result_cache = RecurrentCache(
            state, normalizer if config.kind == "linear" else None, query_offset + seq_len
        )
        return self.output(y), result_cache if use_cache else None
