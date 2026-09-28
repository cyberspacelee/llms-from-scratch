// nvcc -O2 -std=c++17 code/gpu/examples/vector_add.cu -o /tmp/vector_add
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call) do { cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        std::exit(EXIT_FAILURE); \
    } } while (0)

__global__ void add(const float* a, const float* b, float* c, size_t n) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) { std::fprintf(stderr, "A CUDA-capable GPU is required.\n"); return 1; }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const int threads = 64;
    if (threads > prop.maxThreadsPerBlock) {
        std::fprintf(stderr, "Device does not support 64 threads per block.\n");
        return 1;
    }
    std::printf("device=%s SMs=%d warpSize=%d threads=%d\n", prop.name,
                prop.multiProcessorCount, prop.warpSize, threads);
    for (size_t n : {size_t(0), size_t(1), size_t(100), size_t(130), size_t(100003)}) {
        if (n == 0) { std::puts("n=0: no launch"); continue; }
        std::vector<float> a(n), b(n), c(n);
        for (size_t i = 0; i < n; ++i) {
            a[i] = float(int(i % 97) - 48) / 16.0f;
            b[i] = float(int(i % 31) - 15) / 8.0f;
        }
        const size_t bytes = n * sizeof(float);
        float *da = nullptr, *db = nullptr, *dc = nullptr;
        CUDA_CHECK(cudaMalloc(&da, bytes));
        CUDA_CHECK(cudaMalloc(&db, bytes));
        CUDA_CHECK(cudaMalloc(&dc, bytes));
        CUDA_CHECK(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
        const int blocks = int(1 + (n - 1) / threads);
        add<<<blocks, threads>>>(da, db, dc, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) {
            if (!std::isfinite(c[i]) || std::fabs(c[i] - (a[i] + b[i])) > 1e-6f) {
                std::fprintf(stderr, "Mismatch at %zu\n", i);
                CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
                return 1;
            }
        }
        CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
        std::printf("n=%zu blocks=%d: passed\n", n, blocks);
    }
}
