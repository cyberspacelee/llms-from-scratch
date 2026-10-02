"""Analytical ledgers, not profiler readings. A multiply-add counts as 2 FLOPs."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from torch import nn

from .config import AttentionConfig, BlockConfig


def parameter_count(module: nn.Module) -> int:
    """统计独立参数元素数，绑定权重只计一次。

    Args:
        module: 要计数或初始化的 nn.Module。

    Returns:
        int 独立参数元素数。
    """
    return sum(p.numel() for p in module.parameters())  # PyTorch deduplicates tied weights.


@dataclass(frozen=True)
class AttentionCost:
    parameters: int
    projection_flops: int
    mixing_flops: int
    absorption_flops: int
    cache_bytes: int
    decode_cache_read_bytes: int
    score_elements: int

    @property
    def flops(self) -> int:
        """求投影、混合和矩阵吸收 matmul FLOPs之和。

        Args:
            无显式输入；读取实例字段。

        Returns:
            int matmul FLOPs。
        """
        return self.projection_flops + self.mixing_flops + self.absorption_flops

    def to_dict(self) -> dict[str, int]:
        """将成本字段和总 FLOPs 转为字典。

        Args:
            无显式输入；读取实例字段。

        Returns:
            dict[str,int] 成本明细。
        """
        return asdict(self) | {"flops": self.flops}


def attention_cost(
    config: AttentionConfig,
    dim: int,
    batch: int = 1,
    query_tokens: int = 1,
    key_tokens: int = 1024,
    new_kv_tokens: int | None = None,
    bytes_per_element: int = 2,
) -> AttentionCost:
    """Matmul FLOPs only; no norm/softmax/activation, allocator or kernel traffic.

    new_kv_tokens=0 for already-projected cross memory. Sparse manual masks still
    allocate and compute dense scores. Decode cache read is one ideal logical pass.

    Args:
        config: 本模块的显式配置对象。
        dim: 输入/输出 hidden width D；不要求等于 Attention 投影宽度。
        batch: batch size B，正整数。
        query_tokens: 本次 query 长度 T_q。
        key_tokens: 完整可用 KV 长度 T_kv。
        new_kv_tokens: 本次新投影的 KV token 数 N，None 使用 query_tokens；静态 Cross decode 为 0。
        bytes_per_element: 每个缓存元素占用字节数。

    Returns:
        AttentionCost：参数、投影/混合/吸收 FLOPs、cache/read bytes、score elements。
    """
    n = query_tokens if new_kv_tokens is None else new_kv_tokens
    if (
        any(
            type(v) is not int or v < 1
            for v in (dim, batch, query_tokens, key_tokens, bytes_per_element)
        )
        or type(n) is not int
        or n < 0
        or n > key_tokens
    ):
        raise ValueError("invalid ledger dimensions")
    c, b, q, s = config, batch, query_tokens, key_tokens
    h, d, v, rank, r = c.heads, c.head_dim, c.value_dim, c.kv_rank, c.rope_dim
    absorption = 0
    if c.kind in {"mha", "mqa", "gqa"}:
        g = c.effective_kv_heads
        params = 2 * dim * (h + g) * d + (2 * d if c.qk_norm else 0)
        if c.bias:
            params += (h + 2 * g) * d + dim
        projection = 2 * b * (2 * q * dim * h * d + 2 * n * dim * g * d)
        mixing = 4 * b * h * q * s * d
        elements, scores = 2 * b * g * s * d, b * h * q * s
    elif c.kind == "mla":
        qparams = (
            dim * h * (d + r)
            if c.q_rank == 0
            else dim * c.q_rank + c.q_rank * h * (d + r) + c.q_rank
        )
        params = qparams + dim * (rank + r) + rank + rank * h * (d + v) + h * v * dim
        qprojection = dim * h * (d + r) if not c.q_rank else dim * c.q_rank + c.q_rank * h * (d + r)
        projection = 2 * b * (q * qprojection + n * dim * (rank + r))
        if c.mla_impl == "naive":
            projection += 2 * b * (s * rank * h * (d + v) + q * h * v * dim)
            mixing = 2 * b * h * q * s * (d + r + v)
        else:
            projection += 2 * b * q * h * rank * (d + dim)
            mixing = 2 * b * h * q * s * (2 * rank + r)
            absorption = 2 * h * rank * v * dim  # W_UV @ W_O, currently recomputed each call.
        elements, scores = b * s * (rank + r), b * h * q * s
    else:
        gates = int(c.kind != "linear") + int(c.kind == "gated_delta")
        params = 2 * dim * h * (d + v) + gates * (dim + 1) * h
        projection = 2 * b * q * (2 * dim * h * (d + v) + gates * dim * h)
        # State update/read multiply-add terms; excludes decay/beta pointwise products.
        mixing = b * h * q * d * v * (4 if c.kind == "linear" else 6)
        elements = b * h * (d * v + (d if c.kind == "linear" else 0))
        scores = 0
    memory = elements * bytes_per_element
    return AttentionCost(params, projection, mixing, absorption, memory, memory, scores)


def ffn_cost(dim: int, config: BlockConfig, tokens: int = 1) -> dict[str, int]:
    """计算 Dense/MoE 的理论参数和 matmul FLOPs。

    Args:
        dim: 输入/输出 hidden width D；不要求等于 Attention 投影宽度。
        config: 本模块的显式配置对象。
        tokens: 待计算的有效 token 数 N，非负整数。

    Returns:
        dict[str,int]：total_parameters、active_parameters_per_token、matmul_flops。
    """
    multiplier = 3 if config.activation.endswith("glu") else 2
    expert_weights = multiplier * dim * config.ff_dim
    expert = expert_weights
    if config.bias:
        expert += (multiplier - 1) * config.ff_dim + dim
    if config.experts:
        router = dim * config.experts
        total = router + expert * (config.experts + config.shared_experts)
        active = router + expert * (config.top_k + config.shared_experts)
        active_weights = router + expert_weights * (config.top_k + config.shared_experts)
    else:
        total = active = expert
        active_weights = expert_weights
    return {
        "total_parameters": total,
        "active_parameters_per_token": active,
        "matmul_flops": 2 * tokens * active_weights,
    }
