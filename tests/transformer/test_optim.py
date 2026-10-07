import math

import torch

from llms_from_scratch.transformer.optim import AdamW, clip_grad_norm_, cosine_lr


def test_adamw_matches_torch():
    torch.manual_seed(0)
    w0 = torch.randn(5, 3)
    ours, ref = w0.clone().requires_grad_(), w0.clone().requires_grad_()
    kw = dict(lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    o1, o2 = AdamW([ours], **kw), torch.optim.AdamW([ref], foreach=False, **kw)
    x = torch.randn(16, 5)
    for _ in range(20):
        for p, opt in ((ours, o1), (ref, o2)):
            opt.zero_grad()
            (x @ p).pow(2).mean().backward()
            opt.step()
    torch.testing.assert_close(ours, ref, atol=1e-6, rtol=1e-6)


def test_cosine_schedule_shape():
    f = lambda t: cosine_lr(t, 1.0, 0.1, 10, 110)  # noqa: E731
    assert f(0) == 0 and f(5) == 0.5 and f(10) == 1.0
    assert math.isclose(f(60), 0.55)  # 余弦段的中点
    assert math.isclose(f(110), 0.1) and f(500) == 0.1
    assert all(f(t) >= f(t + 1) for t in range(10, 120))


def test_clip_grad_norm_matches_torch():
    torch.manual_seed(0)
    ps = [torch.randn(4, 4, requires_grad=True) for _ in range(3)]
    qs = [p.detach().clone().requires_grad_() for p in ps]
    for group in (ps, qs):
        sum((p**3).sum() for p in group).backward()
    n1 = clip_grad_norm_(ps, 1.0)
    n2 = torch.nn.utils.clip_grad_norm_(qs, 1.0)
    assert math.isclose(n1, n2.item(), rel_tol=1e-5)
    for p, q in zip(ps, qs):
        torch.testing.assert_close(p.grad, q.grad)
    assert math.isclose(math.sqrt(sum(float(p.grad.pow(2).sum()) for p in ps)), 1.0, rel_tol=1e-5)
