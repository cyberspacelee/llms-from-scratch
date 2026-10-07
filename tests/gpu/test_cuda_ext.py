import re

import pytest
import torch

from llms_from_scratch.gpu.cuda_ext import EXTENSIONS, KERNEL_DIR, function_names, load

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA GPU 与 nvcc")


def test_every_declared_function_is_defined_in_source():
    """不编译也能检查：声明给 load_inline 的函数在 .cu 中确有同名定义。"""
    for name in EXTENSIONS:
        source = (KERNEL_DIR / f"{name}.cu").read_text()
        for fn in function_names(name):
            assert re.search(rf"torch::Tensor {fn}\(", source), (name, fn)


def test_kernel_regions_are_balanced():
    for path in KERNEL_DIR.glob("*.cu"):
        text = path.read_text()
        assert text.count("// region ") == text.count("// endregion"), path


@needs_cuda
def test_vector_add():
    ext = load("vector_add")
    a, b = torch.randn(100_003, device="cuda"), torch.randn(100_003, device="cuda")
    torch.testing.assert_close(ext.vector_add(a, b), a + b)


@needs_cuda
@pytest.mark.parametrize("variant", [0, 1, 2])
def test_transpose(variant):
    x = torch.randn(130, 67, device="cuda")
    assert torch.equal(load("transpose").transpose(x, variant), x.T)


@needs_cuda
def test_reduce_and_softmax():
    ext = load("reduce_softmax")
    x = torch.randn(1_000_000, device="cuda", dtype=torch.float64).float()
    torch.testing.assert_close(ext.sum_all(x)[0], x.double().sum().float(), rtol=1e-4, atol=1e-2)
    s = torch.randn(33, 1000, device="cuda") * 5
    torch.testing.assert_close(ext.softmax_rows(s), torch.softmax(s, -1))


@needs_cuda
@pytest.mark.parametrize("variant", [0, 1, 2, 3, 4])
def test_gemm(variant):
    m, k, n = (128, 64, 96) if variant == 4 else (130, 68, 100)
    a, b = torch.randn(m, k, device="cuda"), torch.randn(k, n, device="cuda")
    tol = 5e-2 if variant == 4 else 1e-3
    torch.testing.assert_close(load("gemm").gemm(a, b, variant), a @ b, atol=tol, rtol=tol)
