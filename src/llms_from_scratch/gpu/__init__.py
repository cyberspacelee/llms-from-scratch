"""第八部分 GPU 与 CUDA 编程的参考实现。

CUDA 源码在 ``kernels/*.cu``（经 :mod:`cuda_ext` 编译调用），Triton kernel 在
:mod:`triton_kernels` 与 :mod:`flash_attention_triton`；其余模块是在 CPU 上
按 block / warp / thread 顺序执行同样算法的模拟器与参考实现。
"""
