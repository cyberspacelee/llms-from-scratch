"""训练不依赖缓存；推理 append 返回新状态，方便安全复用前缀。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVCache:
    """单层投影后的 key/value；static=True 表示不可追加的 Cross cache。

    MHA/MQA/GQA: K/V=[B,H_kv,S,D_h]；MLA: key=C[B,1,S,L_kv]，value=Kr[B,1,S,D_r]。
    frozen 限制字段替换，使用方也必须把内部 Tensor 当作只读。
    append 用 cat 返回新对象，不修改已共享前缀；这会产生 O(S) 拷贝，
    学习版重视状态语义，真正的服务缓存需预分配或分页 kernel。
    """

    key: torch.Tensor  # [B,H_kv,S,D_h] or MLA [B,1,S,L_kv]
    value: torch.Tensor  # [B,H_kv,S,D_v] or MLA shared rotary keys [B,1,S,D_r]
    static: bool = False

    def __post_init__(self) -> None:
        """核对 dataclass 字段与组合约束，非法值抛 ValueError。

        Args:
            无显式输入；读取实例字段。

        Returns:
            None；只验证现有字段。
        """
        if (
            self.key.ndim != 4
            or self.value.ndim != 4
            or self.key.shape[:3] != self.value.shape[:3]
            or self.key.device != self.value.device
            or self.key.dtype != self.value.dtype
        ):
            raise ValueError("KV cache requires aligned [B,H_kv,S,D] tensors")

    @property
    def length(self) -> int:
        """读取已处理/缓存的 sequence length。

        Args:
            无显式输入；读取实例字段。

        Returns:
            int sequence length。
        """
        return self.key.shape[-2]

    @property
    def nbytes(self) -> int:
        """统计实际 tensor 存储字节。

        Args:
            无显式输入；读取实例字段。

        Returns:
            int tensor存储字节数。
        """
        return sum(t.numel() * t.element_size() for t in (self.key, self.value))

    def validate(
        self,
        batch: int,
        heads: int,
        key_dim: int,
        value_dim: int,
        reference: torch.Tensor,
        static: bool,
    ) -> None:
        """核对 cache类型、尺寸、位置、device与dtype。

        Args:
            batch: batch size B，正整数。
            heads: 缓存头数 H_kv；MLA 使用1，不要求整除模型宽度。
            key_dim: 每头 key/query 宽度 D_h。
            value_dim: 每头 value 宽度 D_v。
            reference: float hidden Tensor，用来核对 cache 的 device/dtype。
            static: 是否为不可追加的 Cross cache。

        Returns:
            None；检查 static、heads/width、device/dtype，不符合时抛 ValueError。
        """
        expected_k = (batch, heads, self.length, key_dim)
        expected_v = (batch, heads, self.length, value_dim)
        if (
            self.static != static
            or self.length < 1
            or self.key.shape != expected_k
            or self.value.shape != expected_v
        ):
            raise ValueError("cache kind or tensor shapes do not match this attention")
        if any(
            t.device != reference.device or t.dtype != reference.dtype
            for t in (self.key, self.value)
        ):
            raise ValueError("cache device and dtype must match hidden states")

    def append(self, key: torch.Tensor, value: torch.Tensor) -> KVCache:
        """沿 sequence轴追加新的 K/V，维护状态语义。

        Args:
            key: float [B,H_kv,S_new,D_h] 待追加的 keys；MLA 为 [B,1,S_new,L_kv] latent。
            value: float [B,H_kv,S_new,D_v] 待追加的 values；MLA 为 [B,1,S_new,D_r] rotary key。

        Returns:
            新 KVCache，序列轴由 P 增长为 P+S_new；旧 tensor 不变。
        """
        if self.static:
            raise ValueError("cross-attention caches are static")
        # [B,H_kv,P,D_h/D_v] + [B,H_kv,S_new,D_h/D_v] -> [B,H_kv,P+S_new,D_h/D_v]。
        return KVCache(torch.cat((self.key, key), -2), torch.cat((self.value, value), -2))


@dataclass(frozen=True)
class RecurrentCache:
    """固定大小 S[B,H_q,D_h,D_v]，linear 额外保存 z[B,H_q,D_h]。

    length 只记录已处理的位置数，不决定 state 大小；不能与普通 KV 混用。
    """

    state: torch.Tensor  # [B,H_q,D_h,D_v]
    normalizer: torch.Tensor | None  # linear only: [B,H_q,D_h]
    length: int

    def __post_init__(self) -> None:
        """核对 dataclass 字段与组合约束，非法值抛 ValueError。

        Args:
            无显式输入；读取实例字段。

        Returns:
            None；只验证现有字段。
        """
        if self.state.ndim != 4 or type(self.length) is not int or self.length < 1:
            raise ValueError("recurrent cache requires a matrix state and positive length")

    @property
    def nbytes(self) -> int:
        """统计实际 tensor 存储字节。

        Args:
            无显式输入；读取实例字段。

        Returns:
            int tensor存储字节数。
        """
        return sum(
            t.numel() * t.element_size() for t in (self.state, self.normalizer) if t is not None
        )


@dataclass(frozen=True)
class LayerCache:
    self_attention: KVCache | RecurrentCache
    cross_attention: KVCache | None = None


@dataclass(frozen=True)
class ModelCache:
    """模型级 decode 状态：每层状态 + 完整有效前缀 + 可选 Encoder memory。

    kv_nbytes 只计 Attention 的张量存储，不包含 memory、mask、Python 对象、
    权重或分配器保留内存；它是分析账本里的 KV 指标而非进程显存峰值。
    """

    layers: tuple[LayerCache, ...]
    valid: torch.Tensor  # complete decoder prefix mask [B,S]
    memory: torch.Tensor | None = None
    memory_valid: torch.Tensor | None = None

    @property
    def length(self) -> int:
        """读取已处理/缓存的 sequence length。

        Args:
            无显式输入；读取实例字段。

        Returns:
            int sequence length。
        """
        return self.valid.shape[1]

    @property
    def kv_nbytes(self) -> int:
        """统计所有层 Attention cache字节，不包括 hidden memory。

        Args:
            无显式输入；读取实例字段。

        Returns:
            int 模型 Attention cache存储字节数。
        """
        return sum(
            c.nbytes
            for layer in self.layers
            for c in (layer.self_attention, layer.cross_attention)
            if c is not None
        )
