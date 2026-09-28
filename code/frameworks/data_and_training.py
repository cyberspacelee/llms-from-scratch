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
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
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

    generator = torch.Generator().manual_seed(41)
    order = torch.randperm(len(dataset), generator=generator)
    shuffled = list(DataLoader(dataset, batch_size=2, sampler=order.tolist(), num_workers=0))
    assert sorted(order.tolist()) == list(range(len(dataset)))
    assert torch.equal(torch.cat([batch[0] for batch in shuffled]), x[order])

    probe = nn.Linear(1, 1).double()
    with torch.no_grad():
        probe.weight.zero_()
        probe.bias.zero_()
    sgd = torch.optim.SGD(probe.parameters(), lr=0.1)
    sgd.zero_grad(set_to_none=True)
    first_x, first_y = batches[0]
    ((probe(first_x) - first_y).square().mean()).backward()
    torch.testing.assert_close(probe.weight.grad, torch.tensor([[-7.0]], dtype=torch.float64))
    torch.testing.assert_close(probe.bias.grad, torch.tensor([4.0], dtype=torch.float64))
    sgd.step()
    torch.testing.assert_close(probe.weight, torch.tensor([[0.7]], dtype=torch.float64))
    torch.testing.assert_close(probe.bias, torch.tensor([-0.4], dtype=torch.float64))
    assert probe.weight.grad.item() == -7.0
    sgd.zero_grad(set_to_none=True)
    assert all(p.grad is None for p in probe.parameters())
    ((probe(first_x) - first_y).square().mean()).backward()
    sgd.zero_grad(set_to_none=False)
    assert all(torch.count_nonzero(p.grad) == 0 for p in probe.parameters())

    model, optimizer = new_run()
    initial = (model(x) - y).square().mean().item()
    for batch in shuffled[:2]:
        update(model, optimizer, batch)
    frozen = copy.deepcopy(model.state_dict())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "own-checkpoint.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "order": order, "cursor": 2, "step": 2,
                    "torch_rng": torch.get_rng_state(),
                    "order_rng": generator.get_state(), "config": {"in": 1, "out": 1}}, path)
        expected_loss = update(model, optimizer, shuffled[2])
        expected = copy.deepcopy(model.state_dict())
        next_order = torch.randperm(len(dataset), generator=generator)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        resumed, resumed_optimizer = new_run()
        resumed.load_state_dict(checkpoint["model"])
        resumed_optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng"])
        generator.set_state(checkpoint["order_rng"])
        assert torch.equal(torch.randperm(len(dataset), generator=generator), next_order)
        assert checkpoint["config"] == {"in": 1, "out": 1}
        assert checkpoint["cursor"] == checkpoint["step"] == 2
        resumed_batches = list(DataLoader(dataset, batch_size=2,
                                          sampler=checkpoint["order"].tolist(), num_workers=0))
        actual_loss = update(resumed, resumed_optimizer, resumed_batches[checkpoint["cursor"]])
        assert actual_loss == expected_loss
        for key in expected:
            torch.testing.assert_close(resumed.state_dict()[key], expected[key], rtol=0, atol=0)
        weights_only_model, empty_history_optimizer = new_run()
        weights_only_model.load_state_dict(checkpoint["model"])
        assert update(weights_only_model, empty_history_optimizer,
                      resumed_batches[checkpoint["cursor"]]) == expected_loss
        assert any(not torch.equal(weights_only_model.state_dict()[key], expected[key])
                   for key in expected)
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
