"""分页、前缀共享与 int8 量化的可读存储原语。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .state import KVCache


@dataclass(frozen=True)
class QuantizedTensor:
    """每个最后维向量使用一个 scale 的对称 int8 量化，不可微。

    scale=max(abs(x))/127，codes=round(x/scale)；重建误差至多 scale/2。
    nbytes 包括 codes 和 scale，因此小维度下不会恰好得到 4/8 倍压缩。
    这里只实现存储原语，Attention 尚需 decode 到浮点后计算。
    """

    codes: torch.Tensor
    scales: torch.Tensor
    dtype: torch.dtype

    @classmethod
    def encode(cls, x: torch.Tensor) -> QuantizedTensor:
        """对每个末维向量生成对称 int8 codes和scale。

        Args:
            x: 任意非空末维的有限 float tensor [...,D]。

        Returns:
            QuantizedTensor: int8 codes[...,D]、float scales[...,1]、原 dtype。
        """
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("quantization requires finite floating point tensors")
        work = x.float() if x.dtype != torch.float64 else x
        scale = work.abs().amax(-1, keepdim=True) / 127  # [...,D] -> [...,1]
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return cls((work / scale).round().clamp(-127, 127).to(torch.int8), scale, x.dtype)

    def decode(self) -> torch.Tensor:
        """将 int8 codes和scale还原为原dtype浮点值。

        Args:
            无显式输入；读取实例字段。

        Returns:
            重建 float tensor，与原 tensor shape/dtype相同。
        """
        return (self.codes.to(self.scales.dtype) * self.scales).to(self.dtype)

    @property
    def nbytes(self) -> int:
        """统计实际 tensor 存储字节。

        Args:
            无显式输入；读取实例字段。

        Returns:
            int tensor存储字节数。
        """
        return self.codes.numel() + self.scales.numel() * self.scales.element_size()


class PagedKVCache:
    """教学页表：K/V 按序列轴分成 [B,H_kv,page_size,D_h] 的 Tensor pages。

    fork 共享只读前缀；append 对不满页进行 copy-on-write，避免分支污染。
    materialize 拼回连续张量供普通 Attention 使用，会分配/拷贝完整前缀。
    block_table 是 Python 对象标识，不是 GPU 地址或生产级 allocator。
    页式存储改变分配和共享方式，不改变 Attention 数学或逻辑 KV 元素数。
    """

    def __init__(self, page_size: int = 16) -> None:
        """根据尺寸和配置创建参数/子层。

        Args:
            page_size: 每个 Tensor page 的 token 容量。

        Returns:
            None；参数与子层注册在 self 中。
        """
        if type(page_size) is not int or page_size < 1:
            raise ValueError("page_size must be positive")
        self.page_size = page_size
        self.pages: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fork(self) -> PagedKVCache:
        """创建共享只读前缀的分页缓存分支。

        Args:
            无显式输入；读取实例字段。

        Returns:
            新 PagedKVCache，对已有只读完整 pages共享引用。
        """
        result = PagedKVCache(self.page_size)
        result.pages = list(self.pages)
        return result

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """沿 sequence轴追加新的 K/V，维护状态语义。

        Args:
            key: float [B,H_kv,T_new,D_h] 待追加/压缩的 keys。
            value: float [B,H_kv,T_new,D_v] 待追加/压缩的 values。

        Returns:
            None；按 sequence轴增加 pages，部分页采用 copy-on-write。
        """
        if (
            key.ndim != 4
            or value.ndim != 4
            or key.shape[:3] != value.shape[:3]
            or key.shape[-2] == 0
        ):
            raise ValueError("expected nonempty aligned [B,H_q,T,D] tensors")
        if self.pages:
            k, v = self.pages[0]
            if any(
                a.shape[:2] != b.shape[:2]
                or a.shape[-1] != b.shape[-1]
                or a.dtype != b.dtype
                or a.device != b.device
                for a, b in ((k, key), (v, value))
            ):
                raise ValueError("page append shape/device/dtype mismatch")
        # ponytail: CPU reference uses cat for partial-page copy-on-write; a pool/kernel avoids allocation.
        if self.pages and self.pages[-1][0].shape[-2] < self.page_size:
            k, v = self.pages[-1]
            take = min(self.page_size - k.shape[-2], key.shape[-2])
            self.pages[-1] = (
                torch.cat((k, key[..., :take, :]), -2),
                torch.cat((v, value[..., :take, :]), -2),
            )
            key, value = key[..., take:, :], value[..., take:, :]
        for start in range(0, key.shape[-2], self.page_size):
            self.pages.append(
                (
                    key[..., start : start + self.page_size, :].clone(),
                    value[..., start : start + self.page_size, :].clone(),
                )
            )

    def materialize(self) -> KVCache:
        """将分页 K/V拼接为连续张量。

        Args:
            无显式输入；读取实例字段。

        Returns:
            连续 KVCache：K[B,H_kv,T_kv,D_h]、V[B,H_kv,T_kv,D_v]；分配并拷贝。
        """
        if not self.pages:
            raise ValueError("cache is empty")
        return KVCache(*(torch.cat([p[i] for p in self.pages], -2) for i in (0, 1)))

    @property
    def block_table(self) -> tuple[int, ...]:
        """读取教学页表中的 tensor对象标识。

        Args:
            无显式输入；读取实例字段。

        Returns:
            tuple[int,...]，各页 key tensor 的对象标识，不是 GPU 地址。
        """
        return tuple(id(k) for k, _ in self.pages)
