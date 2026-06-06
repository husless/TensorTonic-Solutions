#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 32;


__global__ void outer_product_kernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ C,
                                     int M, int N)
{
    __shared__ float shared_a[BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE];
    
    // Naive version
    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

    // Thread 0 of each row dimension loads into shared A
    if (threadIdx.x == 0 && i < M) {
        shared_a[threadIdx.y] = a[i];
    }

    // Thread 0 of each col dimension loads into shared B
    if (threadIdx.y == 0 && j < N) {
        shared_b[threadIdx.x] = b[j];
    }

    // Synchronize to ensure all shared memory slots are filled
    __syncthreads();

    if (i<M && j<N) {
        C[j+i*N] = shared_a[threadIdx.y] * shared_b[threadIdx.x];
    }
}

extern "C" void solve(const float* a, const float* b, float* C, int M, int N) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE-1) / BLOCK_SIZE, (M + BLOCK_SIZE-1) / BLOCK_SIZE);
    outer_product_kernel<<<blocks, threads>>>(a, b, C, M, N);
    cudaDeviceSynchronize();
}
