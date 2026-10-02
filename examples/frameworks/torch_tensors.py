"""Verify one sequence's tensor storage and multi-head axis contract on CPU."""

import numpy as np
import torch


def main():
    a = np.arange(6, dtype=np.float64).reshape(2, 3)
    shared = torch.from_numpy(a)
    snapshot = torch.tensor(a)
    shared[0, 1] = 9
    assert a[0, 1] == 9 and snapshot[0, 1].item() == 1
    a[0, 1] = 1
    assert torch.as_tensor(a).data_ptr() == shared.data_ptr()
    assert shared.dtype == torch.float64 and shared.device.type == "cpu"
    assert shared.to(dtype=torch.float64) is shared
    assert shared.to(copy=True).data_ptr() != shared.data_ptr()
    assert torch.as_tensor(a, dtype=torch.float32).dtype == torch.float32

    # region head_axes
    x3 = shared.unsqueeze(0)
    assert x3.shape == (1, 2, 3) and x3.data_ptr() == shared.data_ptr()
    zero = torch.zeros((1, 2, 1), dtype=x3.dtype, device=x3.device)
    x = torch.cat([x3, zero], dim=-1)
    assert x.tolist() == [[[0.0, 1.0, 2.0, 0.0], [3.0, 4.0, 5.0, 0.0]]]
    assert x.data_ptr() != shared.data_ptr()
    assert x.stride() == (8, 4, 1)

    by_position = x.reshape(1, 2, 2, 2)
    q = by_position.permute(0, 2, 1, 3)
    assert by_position.stride() == (8, 4, 2, 1)
    assert q.stride() == (8, 2, 4, 1)
    assert q[0, 1, 1, 0].item() == x[0, 1, 2].item() == 5
    assert q.data_ptr() == x.data_ptr()
    assert torch.equal(q.permute(0, 2, 1, 3).reshape(1, 2, 4), x)
    assert not torch.equal(q.reshape(1, 2, 4), x)
    assert q.reshape(1, 2, 4)[0, 1].tolist() == [2.0, 0.0, 5.0, 0.0]

    scores = q @ q.transpose(-2, -1) / (2**0.5)
    assert scores.shape == (1, 2, 2, 2)
    future = torch.triu(torch.ones((2, 2), dtype=torch.bool), diagonal=1)
    weights = scores.masked_fill(future, -torch.inf).softmax(dim=-1)
    assert torch.all(weights[..., 0, 1] == 0)
    assert torch.allclose(weights.sum(dim=-1), torch.ones((1, 2, 2), dtype=x.dtype))
    context = weights @ q
    restored = context.permute(0, 2, 1, 3).reshape(1, 2, 4)
    assert restored.shape == x.shape
    assert torch.allclose(restored[0, 0], x[0, 0])
    key_index = torch.zeros((1, 2, 2, 1), dtype=torch.long)
    first = weights.gather(-1, key_index)
    assert first.shape == (1, 2, 2, 1)
    assert torch.equal(first[..., 0], weights[..., 0])

    # endregion head_axes
    # Unequal axis lengths expose swaps hidden by the 2-by-2 teaching example.
    unequal = torch.arange(12, dtype=torch.float64).reshape(1, 3, 4)
    heads = unequal.reshape(1, 3, 2, 2).permute(0, 2, 1, 3)
    assert heads.shape == (1, 2, 3, 2)
    assert heads[0, 1, 2, 0].item() == unequal[0, 2, 2].item() == 10
    assert torch.equal(heads.permute(0, 2, 1, 3).reshape(1, 3, 4), unequal)

    sliced = shared[:, 1:]
    assert sliced.shape == (2, 2) and sliced.stride() == (3, 1)
    assert sliced.storage_offset() == 1 and sliced[1, 1].item() == 5
    transposed = shared.T
    assert transposed.stride() == (1, 3) and not transposed.is_contiguous()
    try:
        transposed.view(6)
    except RuntimeError:
        pass
    else:
        raise AssertionError("The transposed layout should not flatten as a view")
    assert transposed.reshape(6).tolist() == [0.0, 3.0, 1.0, 4.0, 2.0, 5.0]

    base = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    expanded = base.expand(2, 3)
    repeated = base.repeat(2, 1)
    assert expanded.stride() == (0, 1)
    base[0, 0] = 9
    assert expanded[1, 0].item() == 9 and repeated[1, 0].item() == 1
    assert torch.stack([base, base]).shape == (2, 1, 3)
    print(
        f"torch_tensors: shared storage, head axes, causal mask, gather OK (torch {torch.__version__})"
    )


if __name__ == "__main__":
    main()
