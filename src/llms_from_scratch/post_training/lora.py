"""LoRA：给冻结的线性层加一个低秩旁路 ΔW = (α/r)·B A。"""

from __future__ import annotations

import math

import torch
from torch import nn


# region lora_linear
class LoRALinear(nn.Module):
    """y = x W^T + (α/r) · x A^T B^T，W 冻结，只训练 A[r, in] 与 B[out, r]。

    A 用 Kaiming 均匀初始化，B 初始化为 0：训练开始时 ΔW = 0，模型与基座完全等价，
    而 B 的梯度 ∂L/∂B = (α/r)·g^T (x A^T) 不为零，训练可以立刻启动。
    """

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if r <= 0:
            raise ValueError("rank 必须为正")
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))  # 与 nn.Linear 默认初始化相同
        self.dropout = nn.Dropout(dropout)
        self.merged = False

    def delta_weight(self) -> torch.Tensor:
        return self.scaling * (self.B @ self.A)  # [out, in]，与 base.weight 同形

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.merged:
            return y
        # 先乘 A 再乘 B：中间结果只有 r 维，计算量 O(r·(in+out)) 而不是 O(in·out)
        return y + self.scaling * (self.dropout(x) @ self.A.T @ self.B.T)

    @torch.no_grad()
    def merge(self) -> None:
        """把 ΔW 加回 W：推理时没有额外开销。"""
        if not self.merged:
            self.base.weight += self.delta_weight()
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """从 W 中减去 ΔW，恢复基座权重（可切换到另一个适配器）。"""
        if self.merged:
            self.base.weight -= self.delta_weight()
            self.merged = False
# endregion lora_linear


ATTENTION = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP = ("w1", "w2", "w3")


def apply_lora(model: nn.Module, targets: tuple[str, ...] = ATTENTION + MLP, r: int = 8,
               alpha: float = 16.0) -> nn.Module:
    """冻结整个模型，再把名字以 targets 结尾的 nn.Linear 替换成 LoRALinear。"""
    model.requires_grad_(False)
    for module in list(model.modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child_name in targets:
                setattr(module, child_name, LoRALinear(child, r, alpha))
    return model


def lora_modules(model: nn.Module) -> list[LoRALinear]:
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def merge_lora(model: nn.Module) -> None:
    for m in lora_modules(model):
        m.merge()


def unmerge_lora(model: nn.Module) -> None:
    for m in lora_modules(model):
        m.unmerge()


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def lora_param_count(d_in: int, d_out: int, r: int) -> int:
    """一个 d_out×d_in 矩阵的 LoRA 参数量 r·(d_in + d_out)。"""
    return r * (d_in + d_out)
