"""反向传播：手写常用层的前向与反向（向量-雅可比积）。

对应《反向传播：从标量到矩阵》一章。每个层由一对函数组成：

* ``*_forward(...) -> (out, cache)``：计算输出，并缓存反向需要的中间量；
* ``*_backward(dout, cache) -> grads``：给定上游梯度 dL/dout，返回对各输入的梯度。

反向函数从不构造完整的雅可比矩阵，只计算 VJP。``as_autograd`` 把一对函数包装成
``torch.autograd.Function``，以便用 ``torch.autograd.gradcheck`` 检验。
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import Tensor


# region linear
def linear_forward(x: Tensor, w: Tensor, b: Tensor) -> tuple[Tensor, tuple]:
    """Y = X Wᵀ + b。X: (N, d_in)，W: (d_out, d_in)，b: (d_out,)。"""
    return x @ w.T + b, (x, w)


def linear_backward(dy: Tensor, cache: tuple) -> tuple[Tensor, Tensor, Tensor]:
    """dX = dY W，dW = dYᵀ X，db = Σ_n dY[n]。"""
    x, w = cache
    dx = dy @ w  # (N, d_out) @ (d_out, d_in)
    dw = dy.T @ x  # (d_out, N) @ (N, d_in)：对样本求和
    db = dy.sum(dim=0)  # b 被广播到每一行，反向时把各行加回来
    return dx, dw, db


# endregion


# region activations
def relu_forward(x: Tensor) -> tuple[Tensor, Tensor]:
    return x.clamp_min(0), x


def relu_backward(dy: Tensor, x: Tensor) -> Tensor:
    return dy * (x > 0).to(dy.dtype)


def _normal_cdf(x: Tensor) -> Tensor:
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))


def _normal_pdf(x: Tensor) -> Tensor:
    return torch.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def gelu_forward(x: Tensor) -> tuple[Tensor, Tensor]:
    """精确 GELU：x Φ(x)，Φ 是标准正态分布函数（与 ``F.gelu`` 默认一致）。"""
    return x * _normal_cdf(x), x


def gelu_backward(dy: Tensor, x: Tensor) -> Tensor:
    """d/dx [x Φ(x)] = Φ(x) + x φ(x)。"""
    return dy * (_normal_cdf(x) + x * _normal_pdf(x))


def silu_forward(x: Tensor) -> tuple[Tensor, Tensor]:
    """SiLU（Swish）：x σ(x)。"""
    return x * torch.sigmoid(x), x


def silu_backward(dy: Tensor, x: Tensor) -> Tensor:
    """d/dx [x σ(x)] = σ(x) + x σ(x)(1 − σ(x)) = σ(x)(1 + x(1 − σ(x)))。"""
    s = torch.sigmoid(x)
    return dy * s * (1 + x * (1 - s))


# endregion


# region softmax
def softmax_forward(z: Tensor) -> tuple[Tensor, Tensor]:
    """沿最后一维做 softmax（先减最大值保证数值稳定）。"""
    e = torch.exp(z - z.amax(dim=-1, keepdim=True))
    s = e / e.sum(dim=-1, keepdim=True)
    return s, s


def softmax_backward(ds: Tensor, s: Tensor) -> Tensor:
    """雅可比 J = diag(s) − s sᵀ，VJP 为 Jᵀ g = s ⊙ (g − ⟨g, s⟩)。"""
    return s * (ds - (ds * s).sum(dim=-1, keepdim=True))


def cross_entropy_forward(logits: Tensor, targets: Tensor) -> tuple[Tensor, tuple]:
    """批平均交叉熵 L = mean_n [logsumexp(z_n) − z_n[y_n]]。logits: (N, V)，targets: (N,)。"""
    m = logits.amax(dim=-1, keepdim=True)
    lse = m.squeeze(-1) + torch.log(torch.exp(logits - m).sum(dim=-1))
    picked = logits.gather(-1, targets[:, None]).squeeze(-1)
    loss = (lse - picked).mean()
    probs = torch.exp(logits - lse[:, None])
    return loss, (probs, targets)


def cross_entropy_backward(dloss: Tensor, cache: tuple) -> Tensor:
    """dL/dz = (p − onehot(y)) / N，再乘上游标量梯度。"""
    probs, targets = cache
    n = probs.shape[0]
    grad = probs.clone()
    grad[torch.arange(n), targets] -= 1
    return dloss * grad / n


# endregion


# region norms
def layernorm_forward(
    x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5
) -> tuple[Tensor, tuple]:
    """y = γ ⊙ (x − μ)/σ + β，μ、σ² 沿最后一维计算（有偏方差，与 PyTorch 一致）。"""
    mu = x.mean(dim=-1, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(var + eps)
    xhat = (x - mu) * inv_std
    return gamma * xhat + beta, (xhat, inv_std, gamma)


def layernorm_backward(dy: Tensor, cache: tuple) -> tuple[Tensor, Tensor, Tensor]:
    """dx = (1/σ)(g − mean(g) − x̂ ⊙ mean(g ⊙ x̂))，其中 g = dy ⊙ γ。"""
    xhat, inv_std, gamma = cache
    g = dy * gamma
    dx = inv_std * (
        g - g.mean(dim=-1, keepdim=True) - xhat * (g * xhat).mean(dim=-1, keepdim=True)
    )
    reduce_dims = tuple(range(dy.ndim - 1))
    dgamma = (dy * xhat).sum(dim=reduce_dims)
    dbeta = dy.sum(dim=reduce_dims)
    return dx, dgamma, dbeta


def rmsnorm_forward(x: Tensor, gamma: Tensor, eps: float = 1e-6) -> tuple[Tensor, tuple]:
    """y = γ ⊙ x / rms(x)，rms(x) = sqrt(mean(x²) + ε)。"""
    inv_rms = torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + eps)
    xhat = x * inv_rms
    return gamma * xhat, (xhat, inv_rms, gamma)


def rmsnorm_backward(dy: Tensor, cache: tuple) -> tuple[Tensor, Tensor]:
    """dx = (1/r)(g − x̂ ⊙ mean(g ⊙ x̂))：比 LayerNorm 少了减均值那一项。"""
    xhat, inv_rms, gamma = cache
    g = dy * gamma
    dx = inv_rms * (g - xhat * (g * xhat).mean(dim=-1, keepdim=True))
    dgamma = (dy * xhat).sum(dim=tuple(range(dy.ndim - 1)))
    return dx, dgamma


# endregion


# region as-autograd
def as_autograd(
    forward: Callable[..., tuple[Tensor, object]],
    backward: Callable[[Tensor, object], Tensor | tuple[Tensor, ...]],
) -> Callable[..., Tensor]:
    """把一对手写的 forward/backward 包装成可被 autograd 与 gradcheck 使用的函数。"""

    class Op(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *inputs):  # type: ignore[override]
            out, cache = forward(*inputs)
            ctx.cache = cache
            return out

        @staticmethod
        def backward(ctx, dout):  # type: ignore[override]
            grads = backward(dout, ctx.cache)
            return grads if isinstance(grads, tuple) else (grads,)

    return Op.apply


# endregion
