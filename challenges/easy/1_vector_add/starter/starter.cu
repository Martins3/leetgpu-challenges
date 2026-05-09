#include <cuda_runtime.h>

__global__ void vector_add(const float* A, const float* B, float* C, size_t N) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* A, const float* B, float* C, size_t N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = static_cast<int>((N + threadsPerBlock - 1) / threadsPerBlock);

    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
}
