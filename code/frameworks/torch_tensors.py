"""CPU float64 tensor contracts: aliases, strides, axis layout and indexing."""
import numpy as np
import torch


def main():
    source = np.arange(6, dtype=np.float64).reshape(2, 3)
    shared = torch.from_numpy(source)
    converted = torch.as_tensor(source)
    copied = torch.tensor(source)
    shared[0, 1] = 20
    assert source[0, 1] == converted[0, 1].item() == 20
    assert copied[0, 1].item() == 1
    assert shared.dtype == torch.float64 and shared.device.type == 'cpu'
    assert shared.to(dtype=torch.float64) is shared
    assert shared.to(copy=True).data_ptr() != shared.data_ptr()
    assert torch.as_tensor(source, dtype=torch.float32).data_ptr() != shared.data_ptr()

    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    assert x.stride() == (3, 1)
    sliced = x[:, 1:]
    assert sliced.shape == (2, 2) and sliced.stride() == (3, 1)
    assert sliced.storage_offset() == 1 and sliced[1, 1].item() == 5
    transposed = x.transpose(0, 1)
    assert transposed.shape == (3, 2) and transposed.stride() == (1, 3)
    assert not transposed.is_contiguous()
    try:
        transposed.view(6)
    except RuntimeError:
        pass
    else:
        raise AssertionError('Non-compatible strides unexpectedly flattened with view')
    flattened = transposed.reshape(6)
    assert flattened.tolist() == [0, 3, 1, 4, 2, 5]
    assert flattened.data_ptr() != x.data_ptr()
    assert x.contiguous() is x
    assert transposed.contiguous().stride() == (2, 1)

    states = torch.arange(24, dtype=torch.float64).reshape(1, 3, 2, 4)
    heads = states.permute(0, 2, 1, 3)
    assert heads.shape == (1, 2, 3, 4) and heads.stride() == (24, 4, 8, 1)
    assert heads[0, 1, 2, 3] == states[0, 2, 1, 3] == 23
    assert torch.equal(heads.permute(0, 2, 1, 3).reshape(1, 3, 8), states.reshape(1, 3, 8))
    assert states.squeeze().shape == (3, 2, 4)
    assert states.squeeze(0).unsqueeze(0).shape == states.shape
    base = torch.tensor([[1., 2., 3.]], dtype=torch.float64)
    expanded, repeated = base.expand(2, 3), base.repeat(2, 1)
    assert expanded.stride() == (0, 1) and expanded.data_ptr() == base.data_ptr()
    assert repeated.data_ptr() != base.data_ptr()
    base[0, 0] = 9
    assert expanded[1, 0] == 9 and repeated[1, 0] == 1
    assert torch.cat([base, base], dim=0).shape == (2, 3)
    assert torch.stack([base, base], dim=0).shape == (2, 1, 3)

    scores = torch.tensor([[10., 20., 30.], [40., 50., 60.]], dtype=torch.float64)
    index = torch.tensor([[2, 0], [1, 1]], dtype=torch.long)
    assert scores.gather(1, index).tolist() == [[30, 10], [50, 50]]
    masked = scores.masked_fill(torch.tensor([[False, True, False]]), -torch.inf)
    assert torch.isneginf(masked[:, 1]).all() and torch.isfinite(scores).all()
    print(f'torch_tensors: aliases, strides, B/T/head layout, gather/mask OK (torch {torch.__version__})')


if __name__ == '__main__':
    main()
