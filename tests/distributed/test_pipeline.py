import pytest
import torch
import torch.nn.functional as F

from llms_from_scratch.distributed.comm import launch
from llms_from_scratch.distributed.pipeline import (
    bubble_fraction,
    gpipe_schedule,
    one_f_one_b_schedule,
    peak_activations,
    run_pipeline_local,
    run_pipeline_stage,
    simulate,
    split_gpt,
    zero_bubble_h1_schedule,
)
from llms_from_scratch.transformer.model import GPT, GPTConfig

CONFIG = dict(vocab_size=40, context_length=8, d_model=16, n_layers=4, n_heads=2,
              tie_embeddings=False)
M, MB, T = 4, 2, 8  # 4 个微批，每个 2 条序列


def _model() -> GPT:
    torch.manual_seed(0)
    return GPT(GPTConfig(**CONFIG)).double()


def _batches():
    g = torch.Generator().manual_seed(3)
    data = torch.randint(0, CONFIG["vocab_size"], (M, MB, T + 1), generator=g)
    return [d[:, :-1] for d in data], [d[:, 1:] for d in data]


def _loss(logits, target):
    return F.cross_entropy(logits.flatten(0, 1), target.flatten())


def _reference():
    model = _model()
    inputs, targets = _batches()
    loss = _loss(model(torch.cat(inputs)), torch.cat(targets))
    loss.backward()
    return loss.detach(), {n: p.grad for n, p in model.named_parameters()}


# ---- 调度表与气泡 ----

def _makespan(schedule, cost):
    return max(e.end for e in simulate(schedule, cost))


@pytest.mark.parametrize("p,m", [(2, 4), (4, 8), (4, 12), (8, 16)])
def test_bubble_counts(p, m):
    f, b, w = 1.0, 1.0, 1.0
    full = {"F": f, "B": b + w}
    # GPipe 与 1F1B 的总时长都是 (m + p - 1)(F + B)，气泡率 (p-1)/(m+p-1)
    for schedule in [gpipe_schedule(p, m), one_f_one_b_schedule(p, m)]:
        events = simulate(schedule, full)
        assert max(e.end for e in events) == (m + p - 1) * (f + b + w)
        assert abs(bubble_fraction(events) - (p - 1) / (m + p - 1)) < 1e-12
    # ZB-H1：每个 stage 的气泡从 (p-1)(F+B+W) 降到 (p-1)(F+B-W)
    zb = zero_bubble_h1_schedule(p, m)
    assert _makespan(zb, {"F": f, "B": b, "W": w}) == m * (f + b + w) + (p - 1) * (f + b - w)
    # 不同的耗时比例下同样成立（W 不长于 F、B 时）
    cost = {"F": 1.0, "B": 1.5, "W": 0.5}
    assert _makespan(zb, cost) == m * 3.0 + (p - 1) * (1.0 + 1.5 - 0.5)


def test_activation_peaks():
    p, m = 4, 8
    assert [peak_activations(a) for a in gpipe_schedule(p, m)] == [m] * p
    assert [peak_activations(a) for a in one_f_one_b_schedule(p, m)] == [4, 3, 2, 1]
    # ZB-H1 推迟了 W，后面的 stage 多存几份激活，但峰值不超过 1F1B 的最大值 p
    assert max(peak_activations(a) for a in zero_bubble_h1_schedule(p, m)) == p


def test_every_schedule_runs_each_action_once():
    p, m = 4, 6
    for schedule in [gpipe_schedule(p, m), one_f_one_b_schedule(p, m),
                     zero_bubble_h1_schedule(p, m)]:
        for actions in schedule:
            kinds = {k for k, _ in actions}
            for kind in kinds:
                assert sorted(mb for k, mb in actions if k == kind) == list(range(m))


# ---- 执行：与不切分的模型梯度一致 ----

@pytest.mark.parametrize("p", [2, 4])
@pytest.mark.parametrize("make", [gpipe_schedule, one_f_one_b_schedule, zero_bubble_h1_schedule])
def test_local_pipeline_matches_unsplit(p, make):
    ref_loss, ref_grads = _reference()
    model = _model()
    stages = split_gpt(model, p)
    inputs, targets = _batches()
    loss = run_pipeline_local(stages, make(p, M), inputs, targets, _loss)
    torch.testing.assert_close(loss, ref_loss)
    for name, param in model.named_parameters():
        torch.testing.assert_close(param.grad, ref_grads[name])


def _pipeline_worker(rank: int, world_size: int) -> dict:
    model = _model()
    stage = split_gpt(model, world_size)[rank]
    inputs, targets = _batches()
    actions = one_f_one_b_schedule(world_size, M)[rank]
    loss = run_pipeline_stage(stage, actions, inputs, targets, _loss,
                              (MB, T, CONFIG["d_model"]), torch.float64)
    prefix = {id(p): n for n, p in model.named_parameters()}
    return {"loss": loss, "grads": {prefix[id(p)]: p.grad for p in stage.parameters()}}


def test_multiprocess_1f1b_matches_unsplit():
    ref_loss, ref_grads = _reference()
    results = launch(_pipeline_worker, 2)
    torch.testing.assert_close(results[-1]["loss"], ref_loss)
    seen = set()
    for out in results:
        for name, grad in out["grads"].items():
            torch.testing.assert_close(grad, ref_grads[name])
            seen.add(name)
    assert seen == set(ref_grads)
