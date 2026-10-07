import torch

from llms_from_scratch.architecture.mtp import ParallelHeads, SequentialMTP, mtp_loss, mtp_targets
from llms_from_scratch.transformer.model import GPT, GPTConfig


def tiny():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=20, context_length=16, d_model=32, n_layers=1, n_heads=2)).eval()


def test_target_alignment():
    idx = torch.tensor([[10, 11, 12, 13, 14]])
    assert mtp_targets(idx, 0).tolist() == [[11, 12, 13, 14]]  # 位置 i → t_{i+1}
    assert mtp_targets(idx, 1).tolist() == [[12, 13, 14]]  # 位置 i → t_{i+2}
    assert mtp_targets(idx, 2).tolist() == [[13, 14]]


def test_sequential_shapes_and_main_logits_unchanged():
    model = tiny()
    mtp = SequentialMTP(model, depth=2).eval()
    idx = torch.randint(0, 20, (2, 10))
    outs = mtp(idx)
    assert [o.shape[1] for o in outs] == [10, 9, 8]
    torch.testing.assert_close(outs[0], model(idx))  # MTP 不改变主模型的预测


def test_sequential_mtp_is_causal_up_to_its_input_token():
    model = tiny()
    mtp = SequentialMTP(model, depth=1).eval()
    idx = torch.randint(0, 20, (1, 10))
    changed = idx.clone()
    changed[0, 6] = (idx[0, 6] + 1) % 20
    a, b = mtp(idx)[1], mtp(changed)[1]
    # 深度 1 在位置 i 读到 t_0 … t_{i+1}；改动 t_6 只影响 i >= 5 的预测
    torch.testing.assert_close(a[:, :5], b[:, :5])
    assert not torch.allclose(a[:, 5], b[:, 5])


def test_loss_weighting_and_backward():
    model = tiny()
    mtp = SequentialMTP(model, depth=2)
    idx = torch.randint(0, 20, (2, 12))
    outs = mtp(idx)
    main, extra = mtp_loss(outs, idx, lam=0.3)
    _, unweighted = mtp_loss(outs, idx, lam=2.0)  # λ/D = 1：两个深度损失之和
    torch.testing.assert_close(extra, 0.15 * unweighted)
    (main + extra).backward()
    assert mtp.modules_[1].proj.weight.grad is not None


def test_parallel_heads():
    model = tiny()
    heads = ParallelHeads(model, n_future=3)
    idx = torch.randint(0, 20, (2, 8))
    outs = heads(idx)
    assert len(outs) == 3 and all(o.shape == (2, 8, 20) for o in outs)
    main, extra = mtp_loss(outs, idx, lam=1.0)
    assert torch.isfinite(main + extra)
