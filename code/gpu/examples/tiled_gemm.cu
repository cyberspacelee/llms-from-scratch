// Build: nvcc -O2 code/gpu/examples/tiled_gemm.cu -o /tmp/tiled_gemm
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr int TILE = 16;

// Contract: contiguous row-major float32; distinct A, B, C allocations.
__global__ void tiled_gemm(const float* a, const float* b, float* c, int m, int k, int n) {
    __shared__ float at[TILE][TILE], bt[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * TILE + ty, col = blockIdx.x * TILE + tx;
    float sum = 0.f;
    for (int start = 0; start < k; start += TILE) {
        at[ty][tx] = row < m && start + tx < k ? a[row * k + start + tx] : 0.f;
        bt[ty][tx] = start + ty < k && col < n ? b[(start + ty) * n + col] : 0.f;
        __syncthreads();
        for (int q = 0; q < TILE; ++q) sum += at[ty][q] * bt[q][tx];
        // Protect readers before the next iteration overwrites shared memory.
        __syncthreads();
    }
    if (row < m && col < n) c[row * n + col] = sum;
}

void check_shape(int m, int k, int n) {
    std::vector<float> a(m * k), b(k * n), c(m * n);
    for (int i = 0; i < m * k; ++i) a[i] = (i % 11 - 5) * 0.125f;
    for (int i = 0; i < k * n; ++i) b[i] = (i % 7 - 3) * 0.2f;
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, c.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
    tiled_gemm<<<dim3((n + TILE - 1) / TILE, (m + TILE - 1) / TILE), dim3(TILE, TILE)>>>(da, db, dc, m, k, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(c.data(), dc, c.size() * sizeof(float), cudaMemcpyDeviceToHost));
    double largest = 0;
    for (int row = 0; row < m; ++row) for (int col = 0; col < n; ++col) {
        double expected = 0;
        for (int q = 0; q < k; ++q) expected += double(a[row * k + q]) * double(b[q * n + col]);
        double error = std::abs(double(c[row * n + col]) - expected);
        largest = std::fmax(largest, error);
        if (!std::isfinite(c[row * n + col]) || error > 1e-4 + 1e-4 * std::abs(expected)) {
            std::fprintf(stderr, "GEMM mismatch at (%d,%d)\n", row, col);
            std::exit(EXIT_FAILURE);
        }
    }
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
    std::printf("%dx%d @ %dx%d passed; max absolute error %.3g\n", m, k, k, n, largest);
}

int main() {
    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (properties.maxThreadsPerBlock < TILE * TILE || properties.maxThreadsDim[0] < TILE
            || properties.maxThreadsDim[1] < TILE || properties.sharedMemPerBlock < 2 * TILE * TILE * sizeof(float)) {
        std::fprintf(stderr, "Device does not support this teaching tile configuration\n");
        return EXIT_FAILURE;
    }
    check_shape(5, 7, 6);
    check_shape(17, 19, 13);
    check_shape(32, 32, 32);
    return 0;
}
