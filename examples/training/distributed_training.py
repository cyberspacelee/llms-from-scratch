"""Two DDP ranks reproduce one global update with unequal target counts."""

import tempfile
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

WORLD_SIZE = 2
MICROBATCHES = ((3, 2), (5, 4))
TARGETS = (1.0, 3.0)
LEARNING_RATE = 0.1


class ScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, targets):
        return 0.5 * (self.theta - targets).square()


def local_batches(rank):
    return [
        torch.full((count,), TARGETS[rank], dtype=torch.float64) for count in MICROBATCHES[rank]
    ]


def worker(rank, rendezvous, result_path):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=45),
    )
    try:
        model = DistributedDataParallel(ScalarModel())
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
        batches = local_batches(rank)
        global_count = torch.tensor(sum(len(batch) for batch in batches), dtype=torch.float64)
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
        assert global_count.item() == 14

        optimizer.zero_grad(set_to_none=True)
        for index, batch in enumerate(batches):
            context = model.no_sync() if index + 1 < len(batches) else nullcontext()
            with context:
                local_sum = model(batch).sum()
                (WORLD_SIZE * local_sum / global_count).backward()
        gradient = model.module.theta.grad.detach().clone()
        optimizer.step()

        root_theta = model.module.theta.detach().clone()
        dist.broadcast(root_theta, src=0)
        torch.testing.assert_close(model.module.theta, root_theta, atol=0, rtol=0)
        if rank == 0:
            torch.save({"gradient": gradient, "theta": root_theta}, result_path)
    finally:
        dist.destroy_process_group()


def verify():
    torch.set_num_threads(1)
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    batches = [batch for rank in range(WORLD_SIZE) for batch in local_batches(rank)]
    losses = torch.cat([model(batch) for batch in batches])
    assert len(losses) == 14
    mean_loss = losses.mean()
    torch.testing.assert_close(mean_loss, torch.tensor(43 / 14, dtype=torch.float64))
    mean_loss.backward()
    gradient = model.theta.grad.detach().clone()
    torch.testing.assert_close(gradient, torch.tensor(-32 / 14, dtype=torch.float64))
    optimizer.step()
    torch.testing.assert_close(model.theta, torch.tensor(3.2 / 14, dtype=torch.float64))

    with tempfile.TemporaryDirectory() as directory:
        result_path = str(Path(directory) / "result.pt")
        mp.spawn(
            worker,
            args=(str(Path(directory) / "rendezvous"), result_path),
            nprocs=WORLD_SIZE,
            join=True,
        )
        result = torch.load(result_path, weights_only=True)
    torch.testing.assert_close(result["gradient"], gradient, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(result["theta"], model.theta, atol=1e-12, rtol=1e-12)

    wrong_loss = (0.5 + 4.5) / 2
    wrong_gradient = (-1.0 - 3.0) / 2
    assert wrong_loss != mean_loss.item() and wrong_gradient != gradient.item()
    print("PASS: 14 targets, two DDP ranks, gradient -32/14, SGD theta 3.2/14")


if __name__ == "__main__":
    verify()
