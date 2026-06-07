#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 32;


__global__ void outer_product_kernel(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ C,
                                     int M, int N)
{
    // Coalesced Shared Memory
    __shared__ float shared_a[BLOCK_SIZE];
    __shared__ float shared_b[BLOCK_SIZE];

    // Linear thread index within the 2D block (0 to 1023 for a 32x32 block)
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y;

    // 1. Fully Coalesced Load for Vector A Segment
    // Threads read contiguous global memory addresses and store them into shared memory
    for (int i = tid; i < BLOCK_SIZE; i += block_size) {
        int g_row = blockIdx.y * BLOCK_SIZE + i;
        if (g_row < M) {
            shared_a[i] = a[g_row];
        } else {
            shared_a[i] = 0.0f; // Padding for out-of-bound requests
        }
    }

    // 2. Fully Coalesced Load for Vector B Segment
    for (int i = tid; i < BLOCK_SIZE; i += block_size) {
        int g_col = blockIdx.x * BLOCK_SIZE + i;
        if (g_col < N) {
            shared_b[i] = b[g_col];
        } else {
            shared_b[i] = 0.0f;
        }
    }

    // Synchronize to guarantee shared memory is fully loaded by all cooperative threads
    __syncthreads();

    int i = threadIdx.y + blockIdx.y * blockDim.y;
    int j = threadIdx.x + blockIdx.x * blockDim.x;

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
