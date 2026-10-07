// 归约与 softmax：warp shuffle、block 归约、两阶段网格求和，以及一个 block 处理一行的在线 softmax。
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cfloat>

// region warp_reduce
__device__ __forceinline__ float warp_reduce_sum(float v) {
    // 每轮 lane i 加上 lane i+offset 的值；五轮后 lane 0 持有 32 个值的和
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_down_sync(0xffffffff, v, offset);
    return v;
}

__device__ __forceinline__ float block_reduce_sum(float v) {
    __shared__ float warp_sums[32];  // 每个 warp 一个槽，最多 1024 线程 = 32 个 warp
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_reduce_sum(v);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    const int n_warps = (blockDim.x + 31) / 32;
    v = threadIdx.x < n_warps ? warp_sums[lane] : 0.0f;
    if (warp == 0) v = warp_reduce_sum(v);
    return v;  // 只有 threadIdx.x == 0 的结果有意义
}
// endregion warp_reduce

// region grid_sum
__global__ void sum_kernel(const float* __restrict__ x, float* __restrict__ partial, int64_t n) {
    float v = 0.0f;
    // 网格跨步循环：线程数固定，任意长度的输入都能处理
    for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        v += x[i];
    }
    v = block_reduce_sum(v);
    if (threadIdx.x == 0) partial[blockIdx.x] = v;
}

torch::Tensor sum_all(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat32);
    auto in = x.contiguous();
    const int threads = 256, blocks = 1024;
    auto partial = torch::empty({blocks}, in.options());
    auto out = torch::empty({1}, in.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    // 第一阶段：1024 个 block 各得一个部分和；第二阶段：1 个 block 归约部分和
    sum_kernel<<<blocks, threads, 0, stream>>>(in.data_ptr<float>(), partial.data_ptr<float>(), in.numel());
    sum_kernel<<<1, threads, 0, stream>>>(partial.data_ptr<float>(), out.data_ptr<float>(), blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
// endregion grid_sum

// region softmax
struct MaxSum {
    float m, l;  // 运行最大值与相对它的指数和
};

__device__ __forceinline__ MaxSum merge(MaxSum a, MaxSum b) {
    const float m = fmaxf(a.m, b.m);
    if (m == -INFINITY) return {m, 0.0f};
    return {m, a.l * __expf(a.m - m) + b.l * __expf(b.m - m)};
}

__device__ __forceinline__ MaxSum warp_reduce_maxsum(MaxSum s) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        MaxSum other{__shfl_down_sync(0xffffffff, s.m, offset),
                     __shfl_down_sync(0xffffffff, s.l, offset)};
        s = merge(s, other);
    }
    return s;
}

// 一个 block 处理一行：第一遍在线求 (m, l)，block 内合并，第二遍写 exp(x - m) / l。
__global__ void softmax_rows_kernel(const float* __restrict__ x, float* __restrict__ y, int cols) {
    const float* row = x + static_cast<int64_t>(blockIdx.x) * cols;
    float* out = y + static_cast<int64_t>(blockIdx.x) * cols;
    MaxSum s{-INFINITY, 0.0f};
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s = merge(s, MaxSum{row[c], 1.0f});
    __shared__ MaxSum warp_states[32];
    __shared__ MaxSum total;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    s = warp_reduce_maxsum(s);
    if (lane == 0) warp_states[warp] = s;
    __syncthreads();
    if (warp == 0) {
        const int n_warps = (blockDim.x + 31) / 32;
        s = lane < n_warps ? warp_states[lane] : MaxSum{-INFINITY, 0.0f};
        s = warp_reduce_maxsum(s);
        if (lane == 0) total = s;
    }
    __syncthreads();
    const float m = total.m, inv_l = 1.0f / total.l;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] = __expf(row[c] - m) * inv_l;
}

torch::Tensor softmax_rows(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == torch::kFloat32);
    auto in = x.contiguous();
    auto out = torch::empty_like(in);
    const int rows = in.size(0), cols = in.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    softmax_rows_kernel<<<rows, 256, 0, stream>>>(in.data_ptr<float>(), out.data_ptr<float>(), cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
// endregion softmax
