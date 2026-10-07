// 独立的 CUDA 程序：主机/设备内存、拷贝、启动、同步与错误检查。
// 编译运行：nvcc -O2 -o vector_add vector_add_standalone.cu && ./vector_add
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

// region check
// 每个 CUDA runtime 调用都返回 cudaError_t；不检查的错误会在之后某个无关的调用里才冒出来。
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n",                   \
                         cudaGetErrorString(err_), __FILE__, __LINE__);        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
// endregion check

// region kernel
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // 本线程负责的全局下标
    if (i < n) {                                    // 最后一个 block 可能有多余线程
        c[i] = a[i] + b[i];
    }
}
// endregion kernel

int main() {
    // region host
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);
    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = std::sin(i);
        h_b[i] = std::cos(i);
    }

    float *d_a, *d_b, *d_c;  // 设备指针：主机代码不能直接解引用
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;  // 向上取整
    vector_add<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());       // 启动配置错误（如 block 过大）在这里报告
    CUDA_CHECK(cudaDeviceSynchronize());  // kernel 执行中的错误（如越界访问）在这里报告

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));
    // endregion host

    double max_err = 0.0;
    for (int i = 0; i < n; ++i) max_err = std::fmax(max_err, std::fabs(h_c[i] - (h_a[i] + h_b[i])));
    std::printf("n=%d blocks=%d threads=%d max_err=%g\n", n, blocks, threads, max_err);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    return max_err == 0.0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
