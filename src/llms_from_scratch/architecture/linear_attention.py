"""线性注意力的三种等价形式、衰减门、Mamba-2 式选择性扫描与（门控）DeltaNet。

单头布局：q/k[T, d_k]，v[T, d_v]；状态 S[d_v, d_k]，读出 o_t = S_t q_t。
衰减 alpha[T] ∈ (0, 1]：alpha 全为 1 时退化为无门控的线性注意力。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def feature_map(x: torch.Tensor) -> torch.Tensor:
    """Katharopoulos 等（2020）的 φ(x) = elu(x) + 1，保证特征非负。"""
    return F.elu(x) + 1


# region parallel_recurrent
def decay_matrix(alpha: torch.Tensor) -> torch.Tensor:
    """D[t, s] = Π_{r=s+1}^{t} α_r（s ≤ t），否则 0。用累积对数避免连乘下溢。"""
    b = torch.cumsum(torch.log(alpha), 0)
    D = torch.exp(b.unsqueeze(1) - b.unsqueeze(0))
    return D * torch.ones_like(D).tril()


def linear_attention_parallel(q, k, v, alpha=None, normalize=False):
    """并行形式 O = ((Q K^T) ⊙ D) V：像 softmax 注意力一样一次算出 T×T 矩阵（训练用）。"""
    T = q.shape[0]
    D = decay_matrix(alpha) if alpha is not None else torch.ones(T, T).tril()
    A = (q @ k.t()) * D
    out = A @ v
    if normalize:  # 原始线性注意力的分母 q_t · Σ_s k_s
        out = out / A.sum(-1, keepdim=True)
    return out


def linear_attention_recurrent(q, k, v, alpha=None, normalize=False):
    """递推形式 S_t = α_t S_{t-1} + v_t k_t^T，o_t = S_t q_t：每步 O(d_k d_v)（推理用）。"""
    T, d_k = k.shape
    S = torch.zeros(v.shape[1], d_k)
    z = torch.zeros(d_k)
    outs = []
    for t in range(T):
        a = alpha[t] if alpha is not None else 1.0
        S = a * S + torch.outer(v[t], k[t])
        z = a * z + k[t]
        o = S @ q[t]
        outs.append(o / (z @ q[t]) if normalize else o)
    return torch.stack(outs)
# endregion


# region chunked
def linear_attention_chunked(q, k, v, alpha, chunk: int):
    """分块形式：块内用并行形式，块间只传递一个 d_v × d_k 的状态（训练 kernel 的做法）。"""
    T = q.shape[0]
    S = torch.zeros(v.shape[1], k.shape[1])
    outs = []
    for start in range(0, T, chunk):
        sl = slice(start, min(start + chunk, T))
        qc, kc, vc, ac = q[sl], k[sl], v[sl], alpha[sl]
        b = torch.cumsum(torch.log(ac), 0)  # 块内累积衰减（对数）
        inter = (qc @ S.t()) * torch.exp(b).unsqueeze(1)  # 读取前面所有块留下的状态
        intra = ((qc @ kc.t()) * decay_matrix(ac)) @ vc  # 块内的因果部分
        outs.append(inter + intra)
        w = torch.exp(b[-1] - b).unsqueeze(1)  # 每个位置衰减到块末的系数
        S = torch.exp(b[-1]) * S + (vc * w).t() @ kc
    return torch.cat(outs)
# endregion


# region ssm
def mamba2_scan(x, dt, a, B, C):
    """标量衰减的选择性状态空间（Mamba-2 / SSD 的单头形式）。

    x[T, P] 输入通道，dt[T] > 0 步长，a < 0 标量，B/C[T, N] 输入相关的投影：
        h_t = exp(dt_t · a) h_{t-1} + dt_t · x_t B_t^T，  y_t = h_t C_t。
    “选择性”指 dt、B、C 都随输入变化，模型可以按内容决定记住或忘掉什么。
    """
    h = torch.zeros(x.shape[1], B.shape[1])
    ys = []
    for t in range(x.shape[0]):
        h = torch.exp(dt[t] * a) * h + dt[t] * torch.outer(x[t], B[t])
        ys.append(h @ C[t])
    return torch.stack(ys)
# endregion


# region delta
def delta_rule_recurrent(q, k, v, beta, alpha=None):
    """（门控）DeltaNet：S_t = α_t S_{t-1} (I - β_t k_t k_t^T) + β_t v_t k_t^T。

    等价写法 S_t = α_t S_{t-1} + β_t (v_t - α_t S_{t-1} k_t) k_t^T：先用旧状态预测 k_t
    对应的值，再把预测误差写回去——这是在线最小化 ||S k_t - v_t||² 的一步梯度下降。
    要求 k_t 已 L2 归一化，β_t ∈ [0, 1]；alpha 为 None 时就是不带遗忘门的 DeltaNet。
    """
    T, d_k = k.shape
    S = torch.zeros(v.shape[1], d_k)
    eye = torch.eye(d_k)
    outs = []
    for t in range(T):
        a = alpha[t] if alpha is not None else 1.0
        S = a * S @ (eye - beta[t] * torch.outer(k[t], k[t])) + beta[t] * torch.outer(v[t], k[t])
        outs.append(S @ q[t])
    return torch.stack(outs)
# endregion


def state_bytes(n_layers: int, n_heads: int, d_k: int, d_v: int, bytes_per_elem: int = 2) -> int:
    """线性注意力层的推理状态大小：与序列长度无关。"""
    return n_layers * n_heads * d_k * d_v * bytes_per_elem
