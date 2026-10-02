// Build: nvcc -O2 examples/gpu/examples/tiled_gemm.cu -o /tmp/tiled_gemm
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cstring>

#define CUDA_CHECK(call) do { \
    cudaError_t status = (call); \
    if (status != cudaSuccess) { \
        std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

// Contract: contiguous row-major float32; distinct A, B, C allocations.
// region tiled_gemm
template<int TILE>
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
// endregion tiled_gemm

template<int TILE>
void check_shape(int m, int k, int n, bool benchmark = false) {
    std::vector<float> a(m * k), b(k * n), c(m * n);
    for (int i = 0; i < m * k; ++i) a[i] = (i % 11 - 5) * 0.125f;
    for (int i = 0; i < k * n; ++i) b[i] = (i % 7 - 3) * 0.2f;
    float *da = nullptr, *db = nullptr, *dc = nullptr;
    CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, b.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, c.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, a.data(), a.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), b.size() * sizeof(float), cudaMemcpyHostToDevice));
    tiled_gemm<TILE><<<dim3((n + TILE - 1) / TILE, (m + TILE - 1) / TILE), dim3(TILE, TILE)>>>(da, db, dc, m, k, n);
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
    // region benchmark
    if (benchmark) {
        constexpr int WARMUP = 10, REPEATS = 31;
        const dim3 grid((n + TILE - 1) / TILE, (m + TILE - 1) / TILE), block(TILE, TILE);
        for (int i = 0; i < WARMUP; ++i) tiled_gemm<TILE><<<grid, block>>>(da, db, dc, m, k, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop));
        std::vector<float> times;
        for (int i = 0; i < REPEATS; ++i) {
            CUDA_CHECK(cudaEventRecord(start));
            tiled_gemm<TILE><<<grid, block>>>(da, db, dc, m, k, n);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float milliseconds = 0;
            CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
            times.push_back(milliseconds);
        }
        std::sort(times.begin(), times.end());
        std::printf("tile=%d shape=%dx%dx%d dtype=float32 warmup=%d repeats=%d event_ms min=%.6f median=%.6f max=%.6f\n",
                    TILE, m, k, n, WARMUP, REPEATS, times.front(), times[REPEATS/2], times.back());
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    }
    // endregion benchmark
    CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
    std::printf("tile=%d %dx%d @ %dx%d passed; max absolute error %.3g\n", TILE, m, k, k, n, largest);
}

template<int TILE>
void check_device(const cudaDeviceProp& properties) {
    if (properties.maxThreadsPerBlock < TILE * TILE || properties.maxThreadsDim[0] < TILE
            || properties.maxThreadsDim[1] < TILE || properties.sharedMemPerBlock < 2 * TILE * TILE * sizeof(float)) {
        std::fprintf(stderr, "Device does not support tile=%d teaching configuration\n", TILE);
        std::exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv) {
    const bool benchmark = argc == 2 && std::strcmp(argv[1], "--benchmark") == 0;
    if (argc > 2 || (argc == 2 && !benchmark)) {
        std::fprintf(stderr, "Usage: tiled_gemm [--benchmark]\n");
        return EXIT_FAILURE;
    }
    int device = 0, runtime_version = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    std::printf("GPU=%s capability=%d.%d SMs=%d CUDA_runtime=%d\n",
                properties.name, properties.major, properties.minor, properties.multiProcessorCount, runtime_version);
    check_device<16>(properties); check_device<32>(properties);
    check_shape<16>(5, 7, 6); check_shape<32>(5, 7, 6);
    check_shape<16>(17, 19, 13); check_shape<32>(17, 19, 13);
    check_shape<16>(32, 32, 32); check_shape<32>(32, 32, 32);
    if (benchmark) {
        // Identical deterministic inputs; only the tile size changes. No transfer inside event scope.
        check_shape<16>(256, 256, 256, true);
        check_shape<32>(256, 256, 256, true);
    }
    return 0;
}
