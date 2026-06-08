#include <cuda_runtime.h>

constexpr int TILE = 32;

__global__ void matrix_transpose_kernel(const float* A, float* B, int M, int N) {
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    if (i<M && j<N) {
       B[i + j*M] = A[j + i*N];
    }
}

extern "C" void solve(const float* A, float* B, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_transpose_kernel<<<blocks, threads>>>(A, B, M, N);
    cudaDeviceSynchronize();
}
