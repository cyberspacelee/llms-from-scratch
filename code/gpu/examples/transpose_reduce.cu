// nvcc -O2 transpose_reduce.cu -o transpose_reduce && ./transpose_reduce
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

struct Buffer {
    float* ptr = nullptr;
    explicit Buffer(size_t count) { check(cudaMalloc(&ptr, count * sizeof(float))); }
    ~Buffer() { if (ptr) cudaFree(ptr); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

__global__ void transpose(const float* input, float* output, int rows, int cols) {
    __shared__ float tile[32][33];
    const int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * 32 + tx, y = blockIdx.y * 32 + ty;
    for (int j = 0; j < 32; j += 8)
        tile[ty + j][tx] = x < cols && y + j < rows ? input[(y + j) * cols + x] : 0.0f;
    __syncthreads();
    x = blockIdx.y * 32 + tx;
    y = blockIdx.x * 32 + ty;
    for (int j = 0; j < 32; j += 8)
        if (x < rows && y + j < cols) output[(y + j) * rows + x] = tile[tx][ty + j];
}

__global__ void reduce(const float* input, float* output, int count) {
    __shared__ float values[256];
    const int t = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + t; i < count; i += blockDim.x * gridDim.x)
        sum += input[i];
    values[t] = sum;
    __syncthreads();
    for (int step = 128; step > 0; step /= 2) {
        if (t < step) values[t] += values[t + step];
        __syncthreads();
    }
    if (t == 0) output[blockIdx.x] = values[0];
}

int main() {
    try {
        cudaDeviceProp device{};
        check(cudaGetDeviceProperties(&device, 0));
        if (device.maxThreadsPerBlock < 256 || device.maxThreadsDim[0] < 32 ||
            device.maxThreadsDim[1] < 8 || device.sharedMemPerBlock < 32 * 33 * sizeof(float))
            throw std::runtime_error("Device cannot support this teaching launch");
        std::printf("Device: %s, SMs: %d\n", device.name, device.multiProcessorCount);
        const int rows = 35, cols = 67, count = rows * cols;
        std::vector<float> input(count), actual(count);
        for (int i = 0; i < count; ++i) input[i] = float(i % 17 - 8) / 8;
        Buffer source(count), transposed(count);
        check(cudaMemcpy(source.ptr, input.data(), count * sizeof(float), cudaMemcpyHostToDevice));
        transpose<<<dim3((cols + 31) / 32, (rows + 31) / 32), dim3(32, 8)>>>(source.ptr, transposed.ptr, rows, cols);
        check(cudaGetLastError());
        check(cudaDeviceSynchronize());
        check(cudaMemcpy(actual.data(), transposed.ptr, count * sizeof(float), cudaMemcpyDeviceToHost));
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < cols; ++x)
                if (actual[x * rows + y] != input[y * cols + x]) throw std::runtime_error("Transpose mismatch");
        // An empty input reduces to zero without a zero-block launch.
        for (int n : {0, 1, 257, count}) {
            const int blocks = (n + 255) / 256;
            Buffer partial(blocks ? blocks : 1), result(1);
            if (n == 0) check(cudaMemset(result.ptr, 0, sizeof(float)));
            else {
                reduce<<<blocks, 256>>>(source.ptr, partial.ptr, n);
                check(cudaGetLastError());
                reduce<<<1, 256>>>(partial.ptr, result.ptr, blocks);
                check(cudaGetLastError());
            }
            check(cudaDeviceSynchronize());
            float actual_sum;
            check(cudaMemcpy(&actual_sum, result.ptr, sizeof(float), cudaMemcpyDeviceToHost));
            double expected = 0;
            for (int i = 0; i < n; ++i) expected += input[i];
            if (!std::isfinite(actual_sum) || std::abs(actual_sum - expected) > 1e-5 * (1 + std::abs(expected)))
                throw std::runtime_error("Reduction mismatch");
        }
        std::puts("transpose and two-stage reduction OK");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
