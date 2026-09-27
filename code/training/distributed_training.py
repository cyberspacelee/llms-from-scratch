"""Two real Gloo ranks: token-weighted DDP accumulation vs a serial LM update."""

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "principles"))
from modern_decoder import ModernDecoder
from language_model import sequence_loss


DOCUMENTS = [torch.tensor([[0, 1, 2, 3]]), torch.tensor([[1, 2, 3, 4, 5, 6]]),
             torch.tensor([[2, 3, 4]]), torch.tensor([[3, 4, 5, 6, 7]])]
CONFIG = dict(vocab_size=8, width=8, ff_width=16, heads=4, kv_heads=2, head_width=2, max_length=8)


def new_model():
    torch.manual_seed(19)
    return ModernDecoder(**CONFIG).double()


def worker(rank, world_size, rendezvous, result_path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=world_size, timeout=timedelta(seconds=45))
    try:
        model = DistributedDataParallel(new_model())
        optimizer = torch.optim.SGD(model.parameters(), lr=.02)
        local_documents = DOCUMENTS[rank::world_size]
        count = torch.tensor(sum(doc.shape[1] - 1 for doc in local_documents), dtype=torch.float64)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        optimizer.zero_grad(set_to_none=True)
        for index, document in enumerate(local_documents):
            # Both forward and backward must be inside no_sync; the final microbatch reduces once.
            with model.no_sync() if index + 1 < len(local_documents) else nullcontext():
                local_mean = sequence_loss(model(document[:, :-1]), document[:, 1:])
                (local_mean * (document.shape[1] - 1) * world_size / count).backward()
        gradients = {name: p.grad.clone() for name, p in model.module.named_parameters()}
        optimizer.step()
        for parameter in model.module.parameters():
            reference = parameter.detach().clone()
            dist.broadcast(reference, src=0)
            torch.testing.assert_close(parameter, reference, atol=0, rtol=0)
        if rank == 0:
            torch.save(dict(model=model.module.state_dict(), gradients=gradients, targets=count.item()), result_path)
    finally:
        dist.destroy_process_group()


def verify():
    torch.set_num_threads(1)
    serial = new_model()
    optimizer = torch.optim.SGD(serial.parameters(), lr=.02)
    total = sum(doc.shape[1] - 1 for doc in DOCUMENTS)
    for document in DOCUMENTS:
        (sequence_loss(serial(document[:, :-1]), document[:, 1:]) * (document.shape[1] - 1) / total).backward()
    gradients = {name: p.grad.clone() for name, p in serial.named_parameters()}
    optimizer.step()
    with tempfile.TemporaryDirectory() as directory:
        result_path = str(Path(directory) / "result.pt")
        mp.spawn(worker, args=(2, str(Path(directory) / "rendezvous"), result_path), nprocs=2, join=True)
        result = torch.load(result_path, weights_only=True)
    assert result["targets"] == total == 14
    for name, gradient in gradients.items():
        torch.testing.assert_close(gradient, result["gradients"][name], atol=1e-12, rtol=1e-10)
    for name, parameter in serial.state_dict().items():
        torch.testing.assert_close(parameter, result["model"][name], atol=1e-12, rtol=1e-10)
    # A deliberate counterexample: ranks contain 5 and 9 targets, so equal rank means are wrong.
    means, counts = [], []
    with torch.no_grad():
        for rank in range(2):
            docs = DOCUMENTS[rank::2]
            count = sum(d.shape[1] - 1 for d in docs)
            counts.append(count)
            means.append(sum(sequence_loss(serial(d[:, :-1]), d[:, 1:]).item() * (d.shape[1] - 1) for d in docs) / count)
    assert counts == [5, 9] and abs(sum(means) / 2 - sum(m * c for m, c in zip(means, counts)) / total) > 1e-8
    print("PASS: two real Gloo/DDP ranks, no_sync accumulation, 5/9 token weighting, gradients and update match serial LM")


if __name__ == "__main__":
    verify()
