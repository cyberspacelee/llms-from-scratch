import torch
import torch.nn.functional as F

from llms_from_scratch.distributed.comm import launch
from llms_from_scratch.distributed.ddp import BucketedDDP, NaiveDDP
from llms_from_scratch.transformer.model import GPT, GPTConfig

CONFIG = dict(vocab_size=64, context_length=16, d_model=32, n_layers=2, n_heads=4, n_kv_heads=2)
STEPS, BATCH, SEQ = 3, 8, 16


def _data():
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, CONFIG["vocab_size"], (STEPS, BATCH, SEQ + 1), generator=g)


def _loss(model, batch):
    logits = model(batch[:, :-1])
    return F.cross_entropy(logits.flatten(0, 1), batch[:, 1:].flatten())


def _reference():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**CONFIG)).double()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    for batch in _data():
        opt.zero_grad()
        _loss(model, batch).backward()
        opt.step()
    return model.state_dict()


def _ddp_worker(rank: int, world_size: int) -> dict:
    out = {}
    for name in ["naive", "bucketed"]:
        torch.manual_seed(rank)  # 故意让各 rank 初始化不同：DDP 构造时的 broadcast 要把它们统一
        model = GPT(GPTConfig(**CONFIG)).double()
        if rank == 0:
            torch.manual_seed(0)
            model.load_state_dict(GPT(GPTConfig(**CONFIG)).double().state_dict())
        ddp = NaiveDDP(model) if name == "naive" else BucketedDDP(model, bucket_size_mb=0.01)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        per_rank = BATCH // world_size
        for batch in _data():
            opt.zero_grad()
            local = batch[rank * per_rank:(rank + 1) * per_rank]  # 每个 rank 只看自己那份数据
            _loss(ddp, local).backward()
            if name == "naive":
                ddp.sync_gradients()
            else:
                ddp.finish_gradient_synchronization()
            opt.step()
        out[name] = model.state_dict()
        if name == "bucketed":
            out["buckets"] = len(ddp.buckets)
            out["launched_in_backward"] = ddp.launched_in_backward
    return out


def test_ddp_matches_single_process():
    reference = _reference()
    results = launch(_ddp_worker, 2)
    for out in results:
        for name in ["naive", "bucketed"]:
            for key, value in reference.items():
                torch.testing.assert_close(out[name][key], value, atol=1e-10, rtol=1e-8)
        assert out["buckets"] > 1
        # 所有桶都是在 backward 进行中由 hook 发起的，而不是等 backward 结束后才发
        assert out["launched_in_backward"] == out["buckets"] * STEPS
