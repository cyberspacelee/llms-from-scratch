// 供 torch.utils.cpp_extension.load_inline 编译的 vector add：输入输出都是 torch::Tensor。
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

// region kernel
__global__ void vector_add_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                  float* __restrict__ c, int64_t n) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
// endregion kernel

// region launcher
torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32,
                "inputs must be float32");
    TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
    auto a_c = a.contiguous(), b_c = b.contiguous();
    auto c = torch::empty_like(a_c);
    const int64_t n = a_c.numel();
    if (n == 0) return c;
    const int threads = 256;
    const int blocks = static_cast<int>((n + threads - 1) / threads);
    // 在 PyTorch 当前的 stream 上启动，与其他 PyTorch 算子保持正确的先后顺序
    auto stream = at::cuda::getCurrentCUDAStream();
    vector_add_kernel<<<blocks, threads, 0, stream>>>(a_c.data_ptr<float>(), b_c.data_ptr<float>(),
                                                      c.data_ptr<float>(), n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();  // 相当于 cudaGetLastError + 抛出 C++ 异常
    return c;
}
// endregion launcher
