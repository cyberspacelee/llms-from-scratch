import torch
from torch import nn

from llms_from_scratch.pytorch.training import (
    ResumableBatchSampler,
    SGDMomentum,
    TokenWindows,
    accumulated_step,
    clip_grad_norm,
    load_checkpoint,
    train,
    warmup_cosine,
)


def test_token_windows_shift_by_one():
    ds = TokenWindows(torch.arange(10), 4)
    assert len(ds) == 6
    x, y = ds[2]
    assert x.tolist() == [2, 3, 4, 5] and y.tolist() == [3, 4, 5, 6]


def test_sampler_is_deterministic_and_covers_epoch():
    s = ResumableBatchSampler(10, 3, seed=1)
    it = iter(s)
    first = [next(it) for _ in range(3)]
    assert len({i for b in first for i in b}) == 9  # 一个 epoch 内不重复
    nxt = next(it)
    assert s.state_dict() == {"seed": 1, "epoch": 1, "cursor": 3}
    again = iter(ResumableBatchSampler(10, 3, seed=1))
    assert [next(again) for _ in range(4)] == first + [nxt]


def test_handwritten_sgd_matches_torch():
    torch.manual_seed(0)
    a = nn.Linear(5, 3)
    b = nn.Linear(5, 3)
    b.load_state_dict(a.state_dict())
    opt_a = SGDMomentum(a.parameters(), lr=0.1, momentum=0.9, weight_decay=0.01)
    opt_b = torch.optim.SGD(b.parameters(), lr=0.1, momentum=0.9, weight_decay=0.01)
    for _ in range(5):
        x = torch.randn(8, 5)
        for model, opt in ((a, opt_a), (b, opt_b)):
            opt.zero_grad()
            model(x).pow(2).mean().backward()
            opt.step()
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q)


def test_clip_grad_norm_matches_torch():
    torch.manual_seed(0)
    a, b = nn.Linear(6, 4), nn.Linear(6, 4)
    b.load_state_dict(a.state_dict())
    x = torch.randn(3, 6) * 10
    for m in (a, b):
        m(x).pow(2).sum().backward()
    total = clip_grad_norm(a.parameters(), 1.0)
    ref = nn.utils.clip_grad_norm_(b.parameters(), 1.0)
    torch.testing.assert_close(total, ref)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_gradient_accumulation_equals_full_batch():
    torch.manual_seed(0)
    a, b = nn.Linear(4, 2), nn.Linear(4, 2)
    b.load_state_dict(a.state_dict())
    x, y = torch.randn(8, 4), torch.randn(8, 2)
    loss = nn.functional.mse_loss
    opt_a = torch.optim.SGD(a.parameters(), lr=0.1)
    opt_b = torch.optim.SGD(b.parameters(), lr=0.1)
    accumulated_step(a, loss, [(x[i:i + 2], y[i:i + 2]) for i in range(0, 8, 2)], opt_a)
    accumulated_step(b, loss, [(x, y)], opt_b)
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q)


def test_warmup_cosine_shape():
    lrs = [warmup_cosine(s, warmup=10, total=100) for s in range(120)]
    assert lrs[0] == 0.1 and lrs[9] == 1.0
    assert abs(lrs[100] - 0.1) < 1e-9 and lrs[119] == lrs[100]
    assert all(x >= y for x, y in zip(lrs[9:], lrs[10:]))


class TinyLM(nn.Module):
    def __init__(self, vocab: int = 16, d: int = 32) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.drop = nn.Dropout(0.2)  # 让 RNG 状态影响训练轨迹
        self.head = nn.Linear(d, vocab)

    def forward(self, x):
        return self.head(self.drop(self.embed(x)))


def _setup(seed: int = 0):
    torch.manual_seed(seed)
    model = TinyLM()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: warmup_cosine(s, 3, 12))
    data = TokenWindows(torch.randint(0, 16, (200,), generator=torch.Generator().manual_seed(1)), 8)
    sampler = ResumableBatchSampler(len(data), 16, seed=7)
    return model, opt, sched, data, sampler


def test_resume_reproduces_uninterrupted_run(tmp_path):
    # 不中断：一口气训练 12 步（跨越一个 epoch 边界：192 个窗口 / 每步 32 个）
    model, opt, sched, data, sampler = _setup()
    full = train(model, data, opt, sched, sampler, 0, 12, accum_steps=2)
    reference = {k: v.clone() for k, v in model.state_dict().items()}

    # 中断：训练 5 步并保存，然后在“新进程”中从不同的随机种子重建一切并恢复
    path = tmp_path / "ckpt.pt"
    model, opt, sched, data, sampler = _setup()
    first = train(model, data, opt, sched, sampler, 0, 5, accum_steps=2, checkpoint=(5, path))
    torch.manual_seed(123)  # 搅乱全局 RNG，模拟新进程
    model, opt, sched, data, sampler = _setup(seed=99)
    step = load_checkpoint(path, model, opt, sched, sampler)
    assert step == 5
    rest = train(model, data, opt, sched, sampler, step, 12, accum_steps=2)

    assert first + rest == full  # 每一步的损失逐位相同
    for k, v in model.state_dict().items():
        assert torch.equal(v, reference[k]), k
