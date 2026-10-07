"""超连接（Hyper-Connections, HC）与流形约束超连接（mHC）。

残差流从一条 x ∈ R^d 扩成 n 条 X ∈ R^{n×d}。每层用三组系数读写这些流：
    x_in  = H_pre X              （1×n：把 n 条流混合成层输入）
    X'    = H_res X + H_post^T F(x_in)
              （n×n 的流间混合 + 把层输出按 1×n 的权重写回各条流）
n = 1 且三者都取 1 时就是普通残差 x + F(x)。mHC 把 H_res 投影到双随机矩阵
（行和、列和都为 1 的非负矩阵），H_pre 用 sigmoid、H_post 用 2·sigmoid 约束为非负。
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from llms_from_scratch.transformer.model import RMSNorm


# region sinkhorn
def sinkhorn(logits: torch.Tensor, n_iter: int = 20) -> torch.Tensor:
    """Sinkhorn-Knopp：M = exp(logits)，交替做行归一化与列归一化，逼近双随机矩阵。"""
    M = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))  # 减最大值防溢出，不影响结果
    for _ in range(n_iter):
        M = M / M.sum(-1, keepdim=True)  # 行和 = 1
        M = M / M.sum(-2, keepdim=True)  # 列和 = 1
    return M
# endregion


class HyperConnection(nn.Module):
    """静态 HC（Zhu 等 2024）：H_pre、H_post、H_res 都是不受约束的可学习参数。

    初始化成与 Pre-Norm 残差等价：层只读第 (layer mod n) 条流，输出写回所有流，H_res = I。
    """

    def __init__(self, n: int, layer: int, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        super().__init__()
        self.fn = fn
        self.h_pre = nn.Parameter(torch.eye(n)[layer % n])
        self.h_post = nn.Parameter(torch.ones(n))
        self.h_res = nn.Parameter(torch.eye(n))

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # X[..., n, d]
        x_in = torch.einsum("n,...nd->...d", self.h_pre, X)
        y = self.fn(x_in)
        return torch.einsum("mn,...nd->...md", self.h_res, X) + self.h_post.unsqueeze(-1) * y.unsqueeze(-2)


# region mhc
class ManifoldHyperConnection(nn.Module):
    """mHC（DeepSeek 2025）：系数由当前残差流动态生成，再投影到约束集合上。

    H̃ = α · (RMSNorm(vec X) W) + b；H_pre = σ(H̃_pre)，H_post = 2σ(H̃_post)，
    H_res = Sinkhorn(H̃_res)。α 初始化为小值，b 初始化让 H_res ≈ I（对角占优）。
    """

    def __init__(self, n: int, d: int, fn: Callable[[torch.Tensor], torch.Tensor],
                 n_iter: int = 20) -> None:
        super().__init__()
        self.n, self.fn, self.n_iter = n, fn, n_iter
        self.norm = RMSNorm(n * d)
        self.w = nn.Linear(n * d, n + n + n * n, bias=False)
        nn.init.zeros_(self.w.weight)
        self.alpha = nn.Parameter(torch.full((3,), 0.01))
        self.b_pre = nn.Parameter(torch.zeros(n))
        self.b_post = nn.Parameter(torch.zeros(n))
        self.b_res = nn.Parameter(torch.eye(n) * 4.0)  # exp(4) ≫ exp(0)：初始 H_res 接近单位阵

    def coefficients(self, X: torch.Tensor):
        n = self.n
        raw = self.w(self.norm(X.flatten(-2)))  # [..., 2n + n²]
        pre, post, res = raw.split([n, n, n * n], dim=-1)
        h_pre = torch.sigmoid(self.alpha[0] * pre + self.b_pre)
        h_post = 2 * torch.sigmoid(self.alpha[1] * post + self.b_post)
        h_res = sinkhorn(self.alpha[2] * res.unflatten(-1, (n, n)) + self.b_res, self.n_iter)
        return h_pre, h_post, h_res

    def forward(self, X: torch.Tensor) -> torch.Tensor:  # X[..., n, d]
        h_pre, h_post, h_res = self.coefficients(X)
        y = self.fn(torch.einsum("...n,...nd->...d", h_pre, X))
        return h_res @ X + h_post.unsqueeze(-1) * y.unsqueeze(-2)
# endregion


def expand_streams(x: torch.Tensor, n: int) -> torch.Tensor:
    """嵌入输出复制成 n 条流：[..., d] → [..., n, d]。"""
    return x.unsqueeze(-2).expand(*x.shape[:-1], n, x.shape[-1]).clone()


def collapse_streams(X: torch.Tensor) -> torch.Tensor:
    """最后把 n 条流相加，交给最终的归一化与输出头。"""
    return X.sum(-2)


def composite_gain(mats: list[torch.Tensor]) -> float:
    """多层 H_res 连乘后的最大绝对行和（mHC 论文用它衡量信号放大），恒等映射为 1。"""
    P = torch.eye(mats[0].shape[0])
    for M in mats:
        P = M @ P
    return P.abs().sum(-1).max().item()
