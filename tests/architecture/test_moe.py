import pytest
import torch

from llms_from_scratch.architecture.moe import (
    MoE,
    MoEConfig,
    Routing,
    capacity_mask,
    expert_param_counts,
    load_balance_loss,
    moe_reference,
    update_bias,
)


@pytest.mark.parametrize("score", ["softmax", "sigmoid"])
@pytest.mark.parametrize("n_shared", [0, 1])
def test_dispatch_matches_token_loop(score, n_shared):
    torch.manual_seed(0)
    moe = MoE(MoEConfig(score=score, n_shared=n_shared))
    with torch.no_grad():
        moe.router.weight.normal_()
    x = torch.randn(2, 7, 32)
    torch.testing.assert_close(moe(x), moe_reference(moe, x), atol=1e-5, rtol=1e-5)


def test_routing_picks_topk_and_gates_sum_to_one():
    torch.manual_seed(0)
    moe = MoE(MoEConfig(top_k=3))
    with torch.no_grad():
        moe.router.weight.normal_()
    x = torch.randn(10, 32)
    r = moe.router(x)
    expected = torch.topk(x @ moe.router.weight.t(), 3, dim=-1).indices
    assert torch.equal(r.indices.sort(-1).values, expected.sort(-1).values)
    torch.testing.assert_close(r.gates.sum(-1), torch.ones(10))


def test_bias_changes_selection_but_not_gate_values():
    torch.manual_seed(0)
    moe = MoE(MoEConfig(score="sigmoid", top_k=2))
    x = torch.randn(6, 32)
    before = moe.router(x)
    moe.router.bias[5] = 10.0  # 强行把专家 5 推进每个 token 的 top-k
    after = moe.router(x)
    assert (after.indices == 5).any(-1).all()
    scores = torch.sigmoid(x @ moe.router.weight.t())
    picked = scores.gather(-1, after.indices)
    torch.testing.assert_close(after.gates, picked / picked.sum(-1, keepdim=True))
    assert not torch.equal(before.indices, after.indices)


def test_load_balance_loss_extremes():
    E, N = 4, 8
    uniform = Routing(torch.arange(N).remainder(E).unsqueeze(1), torch.ones(N, 1),
                      torch.full((N, E), 1 / E), torch.ones(N, 1, dtype=torch.bool))
    assert load_balance_loss(uniform, E).item() == pytest.approx(1.0)
    collapsed = Routing(torch.zeros(N, 1, dtype=torch.long), torch.ones(N, 1),
                        torch.eye(E)[[0] * N], torch.ones(N, 1, dtype=torch.bool))
    assert load_balance_loss(collapsed, E).item() == pytest.approx(E)


def test_bias_update_balances_load():
    torch.manual_seed(0)
    moe = MoE(MoEConfig(score="sigmoid", top_k=2, n_experts=8))
    with torch.no_grad():
        moe.router.weight.normal_()
        moe.router.weight[0] += 2.0  # 让专家 0 系统性地更受欢迎
    x = torch.randn(512, 32) + 0.5
    first = update_bias(moe.router, moe.router(x).indices, gamma=0.0)
    for _ in range(300):
        load = update_bias(moe.router, moe.router(x).indices, gamma=0.01)
    assert load.max() / load.mean() < 0.5 * first.max() / first.mean()


def test_capacity_drops_overflow_in_order():
    indices = torch.tensor([[0], [0], [0], [1]])
    kept = capacity_mask(indices, n_experts=2, capacity_factor=1.0)  # C = ceil(4/2) = 2
    assert kept.squeeze(1).tolist() == [True, True, False, True]


def test_dropped_tokens_pass_through_zero_from_routed_experts():
    torch.manual_seed(0)
    moe = MoE(MoEConfig(top_k=1, capacity_factor=0.25))
    x = torch.randn(16, 32)
    y = moe(x)
    dropped = ~moe.last_routing.kept.squeeze(1)
    assert dropped.any()
    assert torch.all(y[dropped] == 0)  # 被丢弃的 token 只剩残差连接
    torch.testing.assert_close(y, moe_reference(moe, x), atol=1e-5, rtol=1e-5)


def test_param_counts():
    total, active = expert_param_counts(MoEConfig(n_experts=64, top_k=6, n_shared=2, d_expert=16))
    assert total == 66 * 3 * 32 * 16 and active == 8 * 3 * 32 * 16
