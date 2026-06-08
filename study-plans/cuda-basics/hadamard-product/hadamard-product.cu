#include <cuda_runtime.h>

__global__ void hadamard_kernel(const float* A, const float* B, float* C, int M, int N) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i<M && j<N) {
        int idx = j + i * N;
        C[idx] = A[idx] * B[idx];
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    hadamard_kernel<<<blocks, threads>>>(A, B, C, M, N);
    cudaDeviceSynchronize();
}
