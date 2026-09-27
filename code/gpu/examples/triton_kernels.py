"""Optional NVIDIA GPU example: pip-installed torch + triton are required.

Run: python code/gpu/examples/triton_kernels.py
No benchmarks: tests compare float32 GPU outputs with float64 references.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, out, n, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n
    a = tl.load(x + offsets, mask=valid, other=0)
    b = tl.load(y + offsets, mask=valid, other=0)
    tl.store(out + offsets, a + b, mask=valid)


def vector_add(x, y):
    if (x.ndim != 1 or y.shape != x.shape or x.dtype != torch.float32
            or y.dtype != x.dtype or not x.is_cuda or y.device != x.device
            or not x.is_contiguous() or not y.is_contiguous()):
        raise ValueError("Expected equal contiguous float32 CUDA vectors on one device")
    out = torch.empty_like(x)
    if x.numel():
        with torch.cuda.device(x.device):
            add_kernel[(triton.cdiv(x.numel(), 256),)](x, y, out, x.numel(), BLOCK_SIZE=256, num_warps=4)
    return out


@triton.jit
def softmax_kernel(x, valid_ptr, out, cols, xs0, xs1, ms0, ms1, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    in_bounds = offsets < cols
    active = tl.load(valid_ptr + row * ms0 + offsets * ms1, mask=in_bounds, other=0)
    values = tl.load(x + row * xs0 + offsets * xs1, mask=in_bounds, other=0).to(tl.float32)
    has_values = tl.sum(active.to(tl.int32), axis=0) > 0
    maximum = tl.max(tl.where(active, values, -float('inf')), axis=0)
    maximum = tl.where(has_values, maximum, 0.0)
    numerator = tl.exp(tl.where(active, values - maximum, -float('inf')))
    denominator = tl.sum(numerator, axis=0)
    result = numerator / tl.where(has_values, denominator, 1.0)
    tl.store(out + row * cols + offsets, result, mask=in_bounds)


def row_softmax(x, valid):
    if (x.ndim != 2 or x.shape[1] == 0 or valid.shape != x.shape
            or x.dtype != torch.float32 or valid.dtype != torch.bool
            or not x.is_cuda or valid.device != x.device
            or min(x.stride()) <= 0 or min(valid.stride()) <= 0):
        raise ValueError("Expected float32 CUDA matrix and bool mask with positive strides")
    if x.shape[1] > 4096:
        raise ValueError("Teaching kernel supports at most 4096 columns")
    if not torch.isfinite(x).all().item():
        raise ValueError("Logits must be finite, including masked positions")
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    # ponytail: one program per row up to 4096 columns; use chunked reduction for wider rows.
    if x.shape[0]:
        with torch.cuda.device(x.device):
            softmax_kernel[(x.shape[0],)](x, valid, out, x.shape[1], *x.stride(), *valid.stride(),
                                        BLOCK_SIZE=triton.next_power_of_2(x.shape[1]), num_warps=4)
    return out


def demo():
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU with working torch and triton is required")
    torch.manual_seed(7)
    x, y = torch.randn(1003, device='cuda'), torch.randn(1003, device='cuda')
    torch.testing.assert_close(vector_add(x, y).double(), x.double() + y.double(), rtol=1e-5, atol=1e-6)
    for cols in (1, 7, 781, 4096):
        # Slice both axes so the contract is tested with non-contiguous inputs.
        logits = (torch.randn(8, cols * 2, device='cuda') * 10 + 1000)[::2, ::2]
        mask = (torch.rand(8, cols * 2, device='cuda') > 0.3)[::2, ::2]
        mask[0] = False
        mask[1, 0] = True
        result = row_softmax(logits, mask)
        reference = torch.softmax(logits.double().masked_fill(~mask, -torch.inf), dim=1)
        reference[0] = 0
        # Other randomly all-masked rows also follow the explicit zero-row policy.
        reference[~mask.any(dim=1)] = 0
        torch.testing.assert_close(result.double(), reference, rtol=1e-4, atol=1e-6)
        assert torch.isfinite(result).all() and (result[~mask] == 0).all()
    torch.cuda.synchronize()
    print("Triton checks passed: tail vector, strided masked rows, all-masked policy")


if __name__ == '__main__':
    demo()
