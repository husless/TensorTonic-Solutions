#include <cuda_runtime.h>

__global__ void matrix_add_kernel(const float* __restrict__ A, const float*  __restrict__ B, float* __restrict__ C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = idx; i < N; i += stride) {
        C[i] = A[i]+B[i];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    int threads = 256;
    int total = N*M;

    int blocks = (total + threads - 1) / threads;
    matrix_add_kernel<<<blocks, threads>>>(A, B, C, total);
    cudaDeviceSynchronize();
}
