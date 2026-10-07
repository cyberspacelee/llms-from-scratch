import torch
import torch.nn.functional as F

from llms_from_scratch.architecture.linear_attention import (
    delta_rule_recurrent,
    feature_map,
    linear_attention_chunked,
    linear_attention_parallel,
    linear_attention_recurrent,
    mamba2_scan,
)


def data(T=11, dk=4, dv=3):
    torch.manual_seed(0)
    return torch.randn(T, dk), torch.randn(T, dk), torch.randn(T, dv), torch.rand(T) * 0.5 + 0.5


def test_parallel_equals_recurrent_normalized():
    q, k, v, _ = data()
    q, k = feature_map(q), feature_map(k)
    torch.testing.assert_close(linear_attention_parallel(q, k, v, normalize=True),
                               linear_attention_recurrent(q, k, v, normalize=True))


def test_parallel_equals_recurrent_with_decay():
    q, k, v, alpha = data()
    torch.testing.assert_close(linear_attention_parallel(q, k, v, alpha),
                               linear_attention_recurrent(q, k, v, alpha))


def test_chunked_equals_recurrent():
    q, k, v, alpha = data(T=13)
    ref = linear_attention_recurrent(q, k, v, alpha)
    for chunk in (1, 4, 5, 13):
        torch.testing.assert_close(linear_attention_chunked(q, k, v, alpha, chunk), ref)


def test_mamba2_is_gated_linear_attention():
    torch.manual_seed(0)
    T, P, N = 9, 3, 4
    x, B, C = torch.randn(T, P), torch.randn(T, N), torch.randn(T, N)
    dt, a = F.softplus(torch.randn(T)), -0.7
    # 对应关系：q = C，k = B，v = dt·x，α_t = exp(dt_t · a)
    gla = linear_attention_recurrent(C, B, dt.unsqueeze(1) * x, torch.exp(dt * a))
    torch.testing.assert_close(mamba2_scan(x, dt, a, B, C), gla)


def test_delta_rule_overwrites_instead_of_accumulating():
    k = torch.eye(4)[[0, 1, 0]]  # 第 3 步再次写入键 0
    v = torch.tensor([[1.0, 0.0], [0.0, 1.0], [5.0, 5.0]])
    q = torch.eye(4)[[0, 0, 0]]
    beta = torch.ones(3)
    delta = delta_rule_recurrent(q, k, v, beta)
    plain = linear_attention_recurrent(q, k, v)
    torch.testing.assert_close(delta[-1], torch.tensor([5.0, 5.0]))  # 新值覆盖旧值
    torch.testing.assert_close(plain[-1], torch.tensor([6.0, 5.0]))  # 线性注意力把两次写入叠加


def test_gated_delta_with_unit_gate_equals_delta():
    q, k, v, alpha = data()
    k = F.normalize(k, dim=-1)
    beta = torch.sigmoid(torch.randn(11))
    torch.testing.assert_close(delta_rule_recurrent(q, k, v, beta, torch.ones(11)),
                               delta_rule_recurrent(q, k, v, beta))
    # α_t 越小遗忘越快：α 全 0 时只剩当前一步的写入
    out = delta_rule_recurrent(q, k, v, beta, torch.zeros(11))
    torch.testing.assert_close(out, beta.unsqueeze(1) * v * (k * q).sum(-1, keepdim=True))
