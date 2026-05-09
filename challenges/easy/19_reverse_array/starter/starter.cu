#include <cuda_runtime.h>

// 2
__global__ void reverse_array(float* input, int N) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t exchange_idx = N - 1 - idx;
    size_t middle = N / 2;
    if (idx >= middle)
        return;
    // TODO 应该是存在原地切换的才可以的
    size_t tmp = input[exchange_idx];
    input[exchange_idx] = input[idx];
    input[idx] = tmp;
}

// input is device pointer
extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    cudaDeviceSynchronize();
}
