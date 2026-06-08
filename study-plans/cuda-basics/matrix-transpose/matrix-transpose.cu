#include <cuda_runtime.h>

constexpr int TILE = 32;

__global__ void matrix_transpose_kernel(const float* A, float* B, int M, int N) {
    // 1. Allocate shared memory tile with a +1 padding column to prevent bank conflicts
    __shared__ float tile[TILE][TILE + 1];

    // 2. Calculate initial coordinates for reading from matrix A
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // 3. Coalesced read from global memory into shared memory tile
    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    }

    // 4. Synchronize all threads within the block before shifting layouts
    __syncthreads();

    // 5. Re-map coordinates to output a transposed grid layout
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;

    // 6. Coalesced write from shared memory tile out to global memory B
    if (x < M && y < N) {
        B[y * M + x] = tile[threadIdx.x][threadIdx.y];
    }
}




extern "C" void solve(const float* A, float* B, int M, int N) {
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matrix_transpose_kernel<<<blocks, threads>>>(A, B, M, N);
    cudaDeviceSynchronize();
}
