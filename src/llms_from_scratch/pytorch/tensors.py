"""张量的内存布局：storage、shape、stride、offset，以及广播与 rearrange。

本模块用最朴素的 Python 重写 PyTorch 在底层做的几件事，测试把它们与 torch 的结果逐一对照：

- ``element_offset``：多维下标如何经 stride 映射到一维存储；
- ``as_strided_reference``：按 (shape, stride, offset) 从一维存储“读出”一个张量；
- ``contiguous_strides`` / ``is_contiguous``：行优先连续布局的判定；
- ``shares_storage``：两个张量是否共用同一块内存；
- ``broadcast_shape``：广播规则；
- ``rearrange``：einops 风格的轴重排，展开成 view/reshape + permute。
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Sequence

import torch


# region offset
def element_offset(index: Sequence[int], stride: Sequence[int], offset: int = 0) -> int:
    """下标 (i_0, …, i_{n-1}) 在一维存储中的位置：offset + Σ i_k · stride_k。"""
    if len(index) != len(stride):
        raise ValueError("index 与 stride 的维数不同")
    return offset + sum(i * s for i, s in zip(index, stride, strict=True))


def as_strided_reference(storage: torch.Tensor, shape: Sequence[int], stride: Sequence[int],
                         offset: int = 0) -> torch.Tensor:
    """逐元素地按 stride 从一维 storage 中取数，得到一个新的（复制出来的）张量。

    这正是 ``torch.as_strided`` 的语义，只不过 torch 不复制，而是返回共享 storage 的视图。
    """
    flat = storage.reshape(-1)
    out = torch.empty(tuple(shape), dtype=storage.dtype)
    for index in itertools.product(*(range(n) for n in shape)):
        out[index] = flat[element_offset(index, stride, offset)]
    return out
# endregion offset


# region contiguous
def contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """行优先（C 顺序）布局的 stride：最后一维步长为 1，前一维步长 = 后面各维长度之积。"""
    strides, step = [], 1
    for n in reversed(shape):
        strides.append(step)
        step *= n
    return tuple(reversed(strides))


def is_contiguous(shape: Sequence[int], stride: Sequence[int]) -> bool:
    """与 ``Tensor.is_contiguous()`` 相同的判定：长度为 1 的维度的 stride 无关紧要。"""
    expected = 1
    for n, s in zip(reversed(shape), reversed(stride), strict=True):
        if n == 1:
            continue
        if s != expected:
            return False
        expected *= n
    return True


def shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """两个张量是否指向同一块底层 storage（视图关系）。"""
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
# endregion contiguous


# region broadcast
def broadcast_shape(*shapes: Sequence[int]) -> tuple[int, ...]:
    """广播规则：右对齐；每一维要么相等，要么其中一个为 1（缺失的维度视为 1）。"""
    ndim = max(len(s) for s in shapes)
    padded = [(1,) * (ndim - len(s)) + tuple(s) for s in shapes]
    result = []
    for dims in zip(*padded, strict=True):
        sizes = {d for d in dims if d != 1}
        if len(sizes) > 1:
            raise ValueError(f"无法广播的形状：{shapes}")
        result.append(sizes.pop() if sizes else 1)
    return tuple(result)
# endregion broadcast


# region rearrange
_TOKEN = re.compile(r"\(([^)]*)\)|(\S+)")


def _parse(side: str) -> list[list[str]]:
    """'b t (h d)' -> [['b'], ['t'], ['h', 'd']]：每个元素是一个实际轴包含的命名轴。"""
    groups = []
    for group, name in _TOKEN.findall(side):
        groups.append(group.split() if group else [name])
    return groups


def rearrange(x: torch.Tensor, pattern: str, **sizes: int) -> torch.Tensor:
    """einops.rearrange 的最小实现，只支持拆分、合并与重排轴。

    三步完成：① reshape 把左边的组合轴拆成命名轴；② permute 按右边的顺序排列命名轴；
    ③ reshape 把右边括号里的命名轴合并。例如 ``rearrange(x, 'b t (h d) -> b h t d', h=4)``。
    """
    left, right = (_parse(side) for side in pattern.split("->"))
    if len(left) != x.dim():
        raise ValueError(f"模式左边有 {len(left)} 个轴，张量有 {x.dim()} 维")
    known = dict(sizes)
    for group, n in zip(left, x.shape, strict=True):
        missing = [name for name in group if name not in known]
        product = 1
        for name in group:
            product *= known.get(name, 1)
        if len(missing) > 1:
            raise ValueError(f"轴组 {group} 中有多个未知长度")
        if missing:
            known[missing[0]] = n // product
        elif product != n:
            raise ValueError(f"轴组 {group} 的长度 {product} 与张量长度 {n} 不符")
    names = [name for group in left for name in group]
    out_names = [name for group in right for name in group]
    if sorted(names) != sorted(out_names):
        raise ValueError("左右两边的命名轴必须相同")
    x = x.reshape([known[name] for name in names])  # ① 拆分
    x = x.permute([names.index(name) for name in out_names])  # ② 重排（不复制）
    final = []
    for group in right:
        n = 1
        for name in group:
            n *= known[name]
        final.append(n)
    return x.reshape(final)  # ③ 合并（必要时才复制）
# endregion rearrange


# region heads
def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """[B, T, d] -> [B, h, T, d/h]：view 拆分最后一维，transpose 交换 T 与 h，均不复制。"""
    B, T, d = x.shape
    return x.view(B, T, n_heads, d // n_heads).transpose(1, 2)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """[B, h, T, d_h] -> [B, T, h·d_h]。transpose 后不再连续，必须用 reshape（会复制）。"""
    B, h, T, dh = x.shape
    return x.transpose(1, 2).reshape(B, T, h * dh)


def attention_scores_einsum(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """S[b,h,i,j] = Σ_d q[b,h,i,d] k[b,h,j,d]，与 q @ k.transpose(-2, -1) 相同。"""
    return torch.einsum("bhid,bhjd->bhij", q, k)
# endregion heads
