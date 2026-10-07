"""从零写 Linear 与 Embedding，并演示 nn.Module 的参数、子模块、buffer 与 train/eval。"""

from __future__ import annotations

import math

import torch
from torch import nn


# region linear
class Linear(nn.Module):
    """y = x Wᵀ + b，W 形状 [out, in]，与 ``nn.Linear`` 的布局和默认初始化完全一致。

    nn.Linear 的默认初始化是 kaiming_uniform_(a=√5)，化简后就是
    W, b ~ U(−1/√in, 1/√in)。按同样的顺序从同一个随机数流取数，结果逐位相同。
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        factory = {"device": device, "dtype": dtype}
        self.in_features, self.out_features = in_features, out_features
        # 给属性赋值一个 nn.Parameter，Module.__setattr__ 就会把它登记进 _parameters
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory))
        self.bias = nn.Parameter(torch.empty(out_features, **factory)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.in_features)
        with torch.no_grad():  # 初始化是对参数的原地修改，不应进入计算图
            self.weight.uniform_(-bound, bound)
            if self.bias is not None:
                self.bias.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"
# endregion linear


# region embedding
class Embedding(nn.Module):
    """查表：第 i 个 token 取 W 的第 i 行。等价于 one_hot(i) @ W，但不做乘法。

    与 nn.Embedding 一样用 N(0, 1) 初始化。
    """

    def __init__(self, num_embeddings: int, embedding_dim: int,
                 device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        with torch.no_grad():
            self.weight.normal_()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.weight[idx]  # 高级索引：输出形状 idx.shape + (embedding_dim,)
# endregion embedding


# region init
def trunc_normal_init_(weight: torch.Tensor) -> torch.Tensor:
    """CS336 作业采用的线性层初始化：N(0, σ²)，σ² = 2/(in+out)，截断在 ±3σ。"""
    out_features, in_features = weight.shape
    std = math.sqrt(2 / (in_features + out_features))
    return nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
# endregion init


# region mlp
class Standardize(nn.Module):
    """用固定的均值与标准差标准化输入。mean/std 不是可训练参数，而是 buffer：
    会随 state_dict 保存、随 .to() 迁移，但不会出现在 parameters() 里，优化器看不到它们。"""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mean", mean.clone())
        self.register_buffer("std", std.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class MLP(nn.Module):
    """Standardize → Linear → GELU → Dropout → Linear，用来观察模块树、state_dict 与 train/eval。"""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float = 0.1,
                 mean: torch.Tensor | None = None, std: torch.Tensor | None = None) -> None:
        super().__init__()
        self.norm = Standardize(torch.zeros(d_in) if mean is None else mean,
                                torch.ones(d_in) if std is None else std)
        self.fc1 = Linear(d_in, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.fc2 = Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.gelu(self.fc1(self.norm(x)))
        return self.fc2(self.drop(h))
# endregion mlp


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    """参数个数；parameters() 会对共享（tied）的参数去重。"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad or not trainable_only)
