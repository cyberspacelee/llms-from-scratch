// SGEMM 的逐步优化：C[M,N] = A[M,K] @ B[K,N]，全部行主序 float32。
// 0 朴素 → 1 共享内存分块 → 2 二维寄存器分块 → 3 向量化加载；另附一个 WMMA（Tensor Core）版本。
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <torch/extension.h>

// region naive
__global__ void gemm_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    // threadIdx.x 沿列方向：同一 warp 的线程读 B 的同一行连续元素（合并），A 的同一元素（广播）
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}
// endregion naive

// region smem
template <int TILE>
__global__ void gemm_smem(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += TILE) {
        // 每个线程搬一个 A 元素和一个 B 元素；越界补 0
        As[ty][tx] = (row < M && k0 + tx < K) ? A[row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(k0 + ty) * N + col] : 0.0f;
        __syncthreads();  // 等整块搬完
        for (int k = 0; k < TILE; ++k) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();  // 等所有人用完，下一轮才能覆盖
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
// endregion smem

// region register_tiled
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_register_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    __shared__ float As[BK][BM];  // 转置存放，便于按 m 方向连续读出 TM 个值
    __shared__ float Bs[BK][BN];
    const int tid = threadIdx.x;
    const int tr = tid / (BN / TN), tc = tid % (BN / TN);  // 线程在 block 内负责的 thread tile
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN] = {};
    float ra[TM], rb[TN];
    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK; i += THREADS) {
            const int r = i / BK, c = i % BK;
            As[c][r] = (row0 + r < M && k0 + c < K) ? A[(row0 + r) * K + k0 + c] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += THREADS) {
            const int r = i / BN, c = i % BN;
            Bs[r][c] = (k0 + r < K && col0 + c < N) ? B[(k0 + r) * N + col0 + c] : 0.0f;
        }
        __syncthreads();
        for (int p = 0; p < BK; ++p) {
            for (int i = 0; i < TM; ++i) ra[i] = As[p][tr * TM + i];  // TM 次共享内存读
            for (int j = 0; j < TN; ++j) rb[j] = Bs[p][tc * TN + j];  // TN 次共享内存读
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];  // TM*TN 次 FMA
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            const int r = row0 + tr * TM + i, c = col0 + tc * TN + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}
// endregion register_tiled

// region vectorized
// 固定 BM=BN=64, BK=8, TM=TN=4，256 个线程；要求 K、N 是 4 的倍数（float4 对齐）。
__global__ void gemm_vectorized(const float* A, const float* B, float* C, int M, int N, int K) {
    constexpr int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
    __shared__ __align__(16) float As[BK][BM];
    __shared__ __align__(16) float Bs[BK][BN];
    const int tid = threadIdx.x;
    const int tr = tid / (BN / TN), tc = tid % (BN / TN);
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN] = {};
    for (int k0 = 0; k0 < K; k0 += BK) {
        if (tid < BM * BK / 4) {  // A 块 64×8 = 128 个 float4
            const int r = tid / 2, c = (tid % 2) * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (row0 + r < M && k0 + c < K)
                v = *reinterpret_cast<const float4*>(&A[(row0 + r) * K + k0 + c]);  // 一条 128 位加载
            As[c + 0][r] = v.x; As[c + 1][r] = v.y; As[c + 2][r] = v.z; As[c + 3][r] = v.w;
        } else {  // B 块 8×64 = 128 个 float4
            const int t = tid - BM * BK / 4, r = t / (BN / 4), c = (t % (BN / 4)) * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (k0 + r < K && col0 + c < N)
                v = *reinterpret_cast<const float4*>(&B[(k0 + r) * N + col0 + c]);
            *reinterpret_cast<float4*>(&Bs[r][c]) = v;
        }
        __syncthreads();
        for (int p = 0; p < BK; ++p) {
            const float4 a = *reinterpret_cast<const float4*>(&As[p][tr * TM]);  // 共享内存 128 位读
            const float4 b = *reinterpret_cast<const float4*>(&Bs[p][tc * TN]);
            const float ra[4] = {a.x, a.y, a.z, a.w}, rb[4] = {b.x, b.y, b.z, b.w};
            for (int i = 0; i < TM; ++i)
                for (int j = 0; j < TN; ++j) acc[i][j] += ra[i] * rb[j];
        }
        __syncthreads();
    }
    for (int i = 0; i < TM; ++i)
        for (int j = 0; j < TN; ++j) {
            const int r = row0 + tr * TM + i, c = col0 + tc * TN + j;
            if (r < M && c < N) C[r * N + c] = acc[i][j];
        }
}
// endregion vectorized

// region wmma
// 每个 warp 用 Tensor Core 计算一个 16×16 输出块；要求 M、N、K 都是 16 的倍数。
__global__ void gemm_wmma(const half* A, const half* B, float* C, int M, int N, int K) {
    using namespace nvcuda;
    const int tile_m = blockIdx.y, tile_n = blockIdx.x;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int k = 0; k < K; k += 16) {
        // 片段在 warp 的 32 个线程寄存器中的分布是不透明的，由硬件决定
        wmma::load_matrix_sync(a, A + tile_m * 16 * K + k, K);
        wmma::load_matrix_sync(b, B + k * N + tile_n * 16, N);
        wmma::mma_sync(acc, a, b, acc);  // acc += a @ b，FP16 输入、FP32 累加
    }
    wmma::store_matrix_sync(C + tile_m * 16 * N + tile_n * 16, acc, N, wmma::mem_row_major);
}
// endregion wmma

// region gemm_launcher
torch::Tensor gemm(torch::Tensor a, torch::Tensor b, int64_t variant) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.dim() == 2 && b.dim() == 2);
    TORCH_CHECK(a.size(1) == b.size(0), "inner dimensions must match");
    auto stream = at::cuda::getCurrentCUDAStream();
    const int M = a.size(0), K = a.size(1), N = b.size(1);
    if (variant == 4) {  // Tensor Core：FP16 输入
        TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0, "WMMA needs multiples of 16");
        auto ah = a.to(torch::kHalf).contiguous(), bh = b.to(torch::kHalf).contiguous();
        auto c = torch::empty({M, N}, a.options().dtype(torch::kFloat32));
        gemm_wmma<<<dim3(N / 16, M / 16), 32, 0, stream>>>(
            reinterpret_cast<const half*>(ah.data_ptr<at::Half>()),
            reinterpret_cast<const half*>(bh.data_ptr<at::Half>()), c.data_ptr<float>(), M, N, K);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return c;
    }
    auto A = a.to(torch::kFloat32).contiguous(), B = b.to(torch::kFloat32).contiguous();
    auto c = torch::empty({M, N}, A.options());
    const float *pa = A.data_ptr<float>(), *pb = B.data_ptr<float>();
    float* pc = c.data_ptr<float>();
    if (variant == 0) {
        dim3 block(32, 8);
        gemm_naive<<<dim3((N + 31) / 32, (M + 7) / 8), block, 0, stream>>>(pa, pb, pc, M, N, K);
    } else if (variant == 1) {
        gemm_smem<32><<<dim3((N + 31) / 32, (M + 31) / 32), dim3(32, 32), 0, stream>>>(pa, pb, pc, M, N, K);
    } else if (variant == 2) {
        gemm_register_tiled<64, 64, 8, 4, 4>
            <<<dim3((N + 63) / 64, (M + 63) / 64), 256, 0, stream>>>(pa, pb, pc, M, N, K);
    } else {
        TORCH_CHECK(K % 4 == 0 && N % 4 == 0, "vectorized kernel needs K, N multiples of 4");
        gemm_vectorized<<<dim3((N + 63) / 64, (M + 63) / 64), 256, 0, stream>>>(pa, pb, pc, M, N, K);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return c;
}
// endregion gemm_launcher
