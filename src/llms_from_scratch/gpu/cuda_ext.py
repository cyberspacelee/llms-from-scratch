"""用 ``torch.utils.cpp_extension.load_inline`` 在运行时编译 ``kernels/*.cu`` 并从 Python 调用。

第一次调用会调用 nvcc 编译（数十秒），产物缓存在 ``~/.cache/torch_extensions``；
之后同样的源码直接加载。需要 CUDA 工具链与 GPU；没有时 :func:`load` 抛出 RuntimeError。
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

import torch

KERNEL_DIR = Path(__file__).parent / "kernels"

# 每个 .cu 文件导出的 C++ 函数签名：load_inline 据此生成 pybind11 绑定
EXTENSIONS: dict[str, list[str]] = {
    "vector_add": ["torch::Tensor vector_add(torch::Tensor a, torch::Tensor b);"],
    "transpose": ["torch::Tensor transpose(torch::Tensor x, int64_t variant);"],
    "reduce_softmax": [
        "torch::Tensor sum_all(torch::Tensor x);",
        "torch::Tensor softmax_rows(torch::Tensor x);",
    ],
    "gemm": ["torch::Tensor gemm(torch::Tensor a, torch::Tensor b, int64_t variant);"],
}


# region load
@cache
def load(name: str, verbose: bool = False):
    """编译并加载 ``kernels/<name>.cu``，返回一个 Python 模块，其属性就是导出的函数。"""
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA GPU 与 nvcc 才能编译 kernel")
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name=f"lfs_{name}",
        cpp_sources=EXTENSIONS[name],  # C++ 侧只需声明；pybind11 绑定代码自动生成
        cuda_sources=[(KERNEL_DIR / f"{name}.cu").read_text()],
        functions=function_names(name),
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=verbose,
    )


# endregion load


def function_names(name: str) -> list[str]:
    """不编译，只解析某个扩展导出的函数名（供测试与文档使用）。"""
    return [d.split("(")[0].split()[-1] for d in EXTENSIONS[name]]
