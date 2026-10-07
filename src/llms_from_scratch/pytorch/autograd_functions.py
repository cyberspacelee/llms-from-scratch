"""自定义 autograd.Function、梯度钩子与“反向传播要保存多少激活”的测量。"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch


# region silu
class SiLUFunction(torch.autograd.Function):
    """SiLU(x) = x·σ(x)，手写前向与反向。

    导数：d/dx [x σ(x)] = σ(x) + x σ(x)(1 − σ(x)) = σ(x)(1 + x(1 − σ(x)))。
    前向只保存输入 x，反向时重新计算 σ(x)——用一点计算换一份激活显存。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        (x,) = ctx.saved_tensors
        s = torch.sigmoid(x)
        return grad_out * s * (1 + x * (1 - s))


def silu(x: torch.Tensor) -> torch.Tensor:
    return SiLUFunction.apply(x)
# endregion silu


# region ste
class RoundSTE(torch.autograd.Function):
    """直通估计器（straight-through estimator）：前向取整，反向假装是恒等函数。

    round 的真实导数几乎处处为 0，梯度传不回去；STE 让梯度原样通过，
    是量化感知训练的基本技巧。
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> torch.Tensor:
        return grad_out


def fake_quantize(x: torch.Tensor, scale: float) -> torch.Tensor:
    """把 x 量化到 scale 的整数倍，再反量化回来；梯度经 STE 直通。"""
    return RoundSTE.apply(x / scale) * scale
# endregion ste


# region hooks
def clamp_grad_hook(limit: float) -> Callable[[torch.Tensor], torch.Tensor]:
    """返回一个张量钩子：在梯度流过该张量时把每个分量截断到 [-limit, limit]。"""

    def hook(grad: torch.Tensor) -> torch.Tensor:
        return grad.clamp(-limit, limit)

    return hook
# endregion hooks


# region saved
def saved_tensor_bytes(fn: Callable[..., object], *inputs: torch.Tensor,
                       exclude: Iterable[torch.Tensor] = ()) -> int:
    """运行 fn(*inputs)，统计 autograd 为反向保存的张量占多少字节。

    ``saved_tensors_hooks`` 的 pack 钩子会看到每一个被保存的张量。多个算子可能保存
    同一块 storage（例如同一个输入被两个算子使用），按 storage 指针去重，只算一次。
    ``exclude`` 中张量的 storage 不计入：模型参数本来就常驻显存，不属于激活。
    """
    skip = {t.untyped_storage().data_ptr() for t in exclude}
    seen: dict[int, int] = {}

    def pack(t: torch.Tensor) -> torch.Tensor:
        storage = t.untyped_storage()
        if storage.data_ptr() not in skip:
            seen[storage.data_ptr()] = storage.nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn(*inputs)
    return sum(seen.values())
# endregion saved
