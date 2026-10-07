// 矩阵转置三个版本：朴素 → 共享内存分块 → 共享内存 + padding（无 bank conflict）。
// 输入 [rows, cols] 行主序 float32，输出 [cols, rows]。block = (32, 8)，每个线程处理 4 个元素。
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;

// region naive
__global__ void transpose_naive(const float* in, float* out, int rows, int cols) {
    int x = blockIdx.x * TILE + threadIdx.x;  // 列
    int y = blockIdx.y * TILE + threadIdx.y;  // 行
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if (x < cols && y + j < rows) {
            // 读：同一 warp 的 32 个线程读同一行的连续 32 个 float → 合并
            // 写：同一 warp 的 32 个线程写 32 个不同的行 → 每个线程一个扇区
            out[x * rows + (y + j)] = in[(y + j) * cols + x];
        }
    }
}
// endregion naive

// region shared
template <int PAD>
__global__ void transpose_shared(const float* in, float* out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + PAD];  // PAD = 1 时每行错开一个 bank
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if (x < cols && y + j < rows) tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * cols + x];
    }
    __syncthreads();  // 整个 tile 写完后才能按列读
    // 交换 block 坐标：输出的第 (blockIdx.x) 个行块
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    for (int j = 0; j < TILE; j += BLOCK_ROWS) {
        if (x < rows && y + j < cols) {
            // 按列读共享内存：tile[threadIdx.x][...]，PAD=0 时 32 个线程落在同一个 bank
            out[(y + j) * rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}
// endregion shared

torch::Tensor transpose(torch::Tensor x, int64_t variant) {
    TORCH_CHECK(x.is_cuda() && x.dim() == 2 && x.scalar_type() == torch::kFloat32);
    auto in = x.contiguous();
    const int rows = in.size(0), cols = in.size(1);
    auto out = torch::empty({cols, rows}, in.options());
    dim3 block(TILE, BLOCK_ROWS);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* src = in.data_ptr<float>();
    float* dst = out.data_ptr<float>();
    if (variant == 0) transpose_naive<<<grid, block, 0, stream>>>(src, dst, rows, cols);
    else if (variant == 1) transpose_shared<0><<<grid, block, 0, stream>>>(src, dst, rows, cols);
    else transpose_shared<1><<<grid, block, 0, stream>>>(src, dst, rows, cols);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
