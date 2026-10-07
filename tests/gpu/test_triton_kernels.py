import pytest
import torch

from llms_from_scratch.gpu import triton_kernels as tk

needs_triton = pytest.mark.skipif(
    not (tk.HAS_TRITON and torch.cuda.is_available()), reason="需要 Triton 与 CUDA GPU"
)


def test_add_ref_masks_tail():
    x, y = torch.randn(1000), torch.randn(1000)
    assert torch.equal(tk.add_ref(x, y, block=128), x + y)


def test_softmax_ref_non_power_of_two():
    x = torch.randn(5, 37) * 4
    torch.testing.assert_close(tk.softmax_ref(x), torch.softmax(x, -1))


def test_program_to_tile_is_a_permutation():
    for num_m, num_n, group in [(9, 9, 3), (7, 5, 4), (4, 6, 1)]:
        tiles = {tk.program_to_tile(p, num_m, num_n, group) for p in range(num_m * num_n)}
        assert tiles == {(i, j) for i in range(num_m) for j in range(num_n)}


def test_grouped_order_reduces_footprint():
    # Triton 官方教程的例子：9×9 个输出块，同时运行 9 个 program
    assert tk.panels_touched(9, 9, group_m=1, wave=9) == 1 + 9
    assert tk.panels_touched(9, 9, group_m=3, wave=9) == 3 + 3


def test_matmul_ref():
    a, b = torch.randn(40, 24), torch.randn(24, 33)
    torch.testing.assert_close(tk.matmul_ref(a, b, 16, 16, 8, group_m=2), a @ b)


@needs_triton
def test_triton_kernels_on_gpu():
    x = torch.randn(10_000, device="cuda")
    y = torch.randn(10_000, device="cuda")
    torch.testing.assert_close(tk.add(x, y), x + y)
    s = torch.randn(64, 781, device="cuda")
    torch.testing.assert_close(tk.softmax(s), torch.softmax(s, -1))
    a = torch.randn(512, 256, device="cuda", dtype=torch.float16)
    b = torch.randn(256, 384, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(tk.matmul(a, b), a @ b, atol=1e-2, rtol=1e-2)
