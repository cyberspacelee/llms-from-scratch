import pytest

from llms_from_scratch.pretraining.schedules import multi_step, warmup_cosine, wsd


def test_warmup_cosine_endpoints():
    assert warmup_cosine(0, 1.0, 0.1, 10, 100) == pytest.approx(0.1)
    assert warmup_cosine(9, 1.0, 0.1, 10, 100) == pytest.approx(1.0)
    assert warmup_cosine(55, 1.0, 0.1, 10, 100) == pytest.approx(0.55)   # 余弦段的中点
    assert warmup_cosine(100, 1.0, 0.1, 10, 100) == pytest.approx(0.1)


@pytest.mark.parametrize("shape", ["linear", "cosine", "sqrt"])
def test_wsd_is_flat_then_decays(shape):
    lrs = [wsd(s, 1.0, 0.0, 10, 100, decay_frac=0.2, shape=shape) for s in range(101)]
    assert all(lr == 1.0 for lr in lrs[10:80])
    assert all(a >= b for a, b in zip(lrs[80:], lrs[81:]))
    assert lrs[100] == 0.0


def test_multi_step_matches_deepseek_llm():
    assert multi_step(50, 1.0, 10, 100) == 1.0
    assert multi_step(85, 1.0, 10, 100) == pytest.approx(0.316)
    assert multi_step(95, 1.0, 10, 100) == pytest.approx(0.1)
