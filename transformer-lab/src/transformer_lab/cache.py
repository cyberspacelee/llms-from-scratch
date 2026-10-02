"""Functional caches: append returns a new cache, so prefixes can be shared safely."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVCache:
    key: torch.Tensor  # [B,Hkv,S,Dk] or MLA [B,1,S,L]
    value: torch.Tensor  # [B,Hkv,S,Dv] or MLA shared rotary keys [B,1,S,R]
    static: bool = False

    def __post_init__(self) -> None:
        if (
            self.key.ndim != 4
            or self.value.ndim != 4
            or self.key.shape[:3] != self.value.shape[:3]
            or self.key.device != self.value.device
            or self.key.dtype != self.value.dtype
        ):
            raise ValueError("KV cache requires aligned [B,H,S,D] tensors")

    @property
    def length(self) -> int:
        return self.key.shape[-2]

    @property
    def nbytes(self) -> int:
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
        if self.static:
            raise ValueError("cross-attention caches are static")
        return KVCache(torch.cat((self.key, key), -2), torch.cat((self.value, value), -2))


@dataclass(frozen=True)
class RecurrentCache:
    state: torch.Tensor  # [B,H,Dk,Dv]
    normalizer: torch.Tensor | None  # linear only: [B,H,Dk]
    length: int

    def __post_init__(self) -> None:
        if self.state.ndim != 4 or type(self.length) is not int or self.length < 1:
            raise ValueError("recurrent cache requires a matrix state and positive length")

    @property
    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size() for t in (self.state, self.normalizer) if t is not None
        )


@dataclass(frozen=True)
class LayerCache:
    self_attention: KVCache | RecurrentCache
    cross_attention: KVCache | None = None


@dataclass(frozen=True)
class ModelCache:
    layers: tuple[LayerCache, ...]
    valid: torch.Tensor  # complete decoder prefix mask [B,S]
    memory: torch.Tensor | None = None
    memory_valid: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return self.valid.shape[1]

    @property
    def kv_nbytes(self) -> int:
        return sum(
            c.nbytes
            for layer in self.layers
            for c in (layer.self_attention, layer.cross_attention)
            if c is not None
        )


def compress_sequence(
    key: torch.Tensor, value: torch.Tensor, block_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mean pooling completed blocks; never include an unfinished/future block."""
    if key.ndim != 4 or value.ndim != 4 or key.shape[:3] != value.shape[:3] or block_size < 1:
        raise ValueError("expected aligned [B,H,S,D] and positive block size")
    count = key.shape[-2] // block_size

    def pool(x: torch.Tensor) -> torch.Tensor:
        return (
            x[..., : count * block_size, :]
            .reshape(*x.shape[:2], count, block_size, x.shape[-1])
            .mean(-2)
        )

    ends = torch.arange(count, device=key.device) * block_size + block_size - 1
    return pool(key), pool(value), ends


@dataclass(frozen=True)
class QuantizedTensor:
    codes: torch.Tensor
    scales: torch.Tensor
    dtype: torch.dtype

    @classmethod
    def encode(cls, x: torch.Tensor) -> QuantizedTensor:
        if not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError("quantization requires finite floating point tensors")
        work = x.float() if x.dtype != torch.float64 else x
        scale = work.abs().amax(-1, keepdim=True) / 127
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        return cls((work / scale).round().clamp(-127, 127).to(torch.int8), scale, x.dtype)

    def decode(self) -> torch.Tensor:
        return (self.codes.to(self.scales.dtype) * self.scales).to(self.dtype)

    @property
    def nbytes(self) -> int:
        return self.codes.numel() + self.scales.numel() * self.scales.element_size()


class PagedKVCache:
    """Educational tensor pages. Prefix forks share full/partial pages without mutation."""

    def __init__(self, page_size: int = 16) -> None:
        if type(page_size) is not int or page_size < 1:
            raise ValueError("page_size must be positive")
        self.page_size = page_size
        self.pages: list[tuple[torch.Tensor, torch.Tensor]] = []

    def fork(self) -> PagedKVCache:
        result = PagedKVCache(self.page_size)
        result.pages = list(self.pages)
        return result

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        if (
            key.ndim != 4
            or value.ndim != 4
            or key.shape[:3] != value.shape[:3]
            or key.shape[-2] == 0
        ):
            raise ValueError("expected nonempty aligned [B,H,T,D] tensors")
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
        if not self.pages:
            raise ValueError("cache is empty")
        return KVCache(*(torch.cat([p[i] for p in self.pages], -2) for i in (0, 1)))

    @property
    def block_table(self) -> tuple[int, ...]:
        return tuple(id(k) for k, _ in self.pages)
