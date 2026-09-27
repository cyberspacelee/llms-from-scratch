"""Verify batching, optimizer state, safe local checkpoints and CPU continuation."""

import copy
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def new_run():
    model = nn.Linear(1, 1).double()
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.01, foreach=False)
    return model, optimizer


def update(model, optimizer, batch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in model.parameters())
    x, y = batch
    loss = (model(x) - y).square().mean()
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
    assert torch.isfinite(norm)
    optimizer.step()
    return loss.item()


def verify():
    torch.set_num_threads(1)
    torch.manual_seed(41)
    x = torch.arange(-2, 3, dtype=torch.float64).reshape(-1, 1)
    y = 2 * x + 1
    dataset = TensorDataset(x, y)
    generator = torch.Generator().manual_seed(41)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, generator=generator)
    batches = list(loader)
    assert [len(batch[0]) for batch in batches] == [2, 2, 1]
    assert torch.equal(torch.cat([batch[0] for batch in batches]), x)
    assert len(list(DataLoader(dataset, batch_size=2, drop_last=True))) == 2
    assert len(list(DataLoader(dataset, batch_size=8, drop_last=True))) == 0
    sums = [batch[1].square().sum().item() for batch in batches]
    weighted = sum(sums) / len(dataset)
    unweighted = sum(s / len(batch[0]) for s, batch in zip(sums, batches)) / len(batches)
    assert weighted == 9.0 and abs(unweighted - 35 / 3) < 1e-12
    assert loader.batch_size == 2 and batches[0][0].shape == (2, 1)

    snapshot = generator.get_state()
    order = torch.randperm(5, generator=generator)
    generator.set_state(snapshot)
    assert torch.equal(torch.randperm(5, generator=generator), order)
    assert torch.equal(torch.randperm(5, generator=torch.Generator().manual_seed(9)),
                       torch.randperm(5, generator=torch.Generator().manual_seed(9)))

    probe = nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    sgd = torch.optim.SGD([probe], lr=0.1)
    (probe * 2).sum().backward()
    sgd.step()
    torch.testing.assert_close(probe, torch.tensor([0.8], dtype=torch.float64))
    assert probe.grad.item() == 2.0
    sgd.zero_grad(set_to_none=True)
    assert probe.grad is None
    (probe * 3).sum().backward()
    sgd.zero_grad(set_to_none=False)
    assert probe.grad.item() == 0.0

    model, optimizer = new_run()
    initial = (model(x) - y).square().mean().item()
    for batch in batches[:2]:
        update(model, optimizer, batch)
    frozen = copy.deepcopy(model.state_dict())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "own-checkpoint.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "cursor": 2, "step": 2, "torch_rng": torch.get_rng_state(),
                    "loader_rng": generator.get_state(), "config": {"in": 1, "out": 1}}, path)
        expected_loss = update(model, optimizer, batches[2])
        expected = copy.deepcopy(model.state_dict())
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        resumed, resumed_optimizer = new_run()
        resumed.load_state_dict(checkpoint["model"])
        resumed_optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        generator.set_state(checkpoint["loader_rng"])
        assert checkpoint["config"] == {"in": 1, "out": 1}
        assert checkpoint["cursor"] == checkpoint["step"] == 2
        actual_loss = update(resumed, resumed_optimizer, batches[checkpoint["cursor"]])
        assert actual_loss == expected_loss
        for key in expected:
            torch.testing.assert_close(resumed.state_dict()[key], expected[key], rtol=0, atol=0)
        for parameter in resumed.parameters():
            assert resumed_optimizer.state[parameter]["step"].item() == 3
        final = (resumed(x) - y).square().mean().item()
        assert final < initial
        assert any(not torch.equal(frozen[key], expected[key]) for key in frozen)

    alias_model = nn.Linear(1, 1).double()
    state = alias_model.state_dict()
    before = state["weight"].clone()
    with torch.no_grad():
        alias_model.weight.add_(1)
    torch.testing.assert_close(state["weight"], before + 1)
    print(f"Data/training API: batches 2/2/1, weighted loss {weighted:g}, exact AdamW resume passed")
    print(f"Runtime: NumPy-independent; torch {torch.__version__}, CPU float64; initial {initial:.6f}, final {final:.6f}")


if __name__ == "__main__":
    verify()
