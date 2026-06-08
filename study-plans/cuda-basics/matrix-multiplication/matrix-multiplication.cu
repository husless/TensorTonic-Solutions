#include <cuda_runtime.h>

constexpr int TILE = 32;

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Pad the width by +1 to eliminate shared memory bank conflicts
    __shared__ float s_A[TILE][TILE+1];
    __shared__ float s_B[TILE][TILE+1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = ty + blockIdx.y * TILE;
    int col = tx + blockIdx.x * TILE;

    float sum = 0.0f;

    for (int tileIdx = 0; tileIdx < (K + TILE - 1) / TILE; ++tileIdx) {
        
        // Collaborative Load for Matrix A
        // Threads load sequentially along the row (K dimension)
        int a_col = tileIdx * TILE + tx;
        if (row < M && a_col < K) {
            s_A[ty][tx] = A[row * K + a_col];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Collaborative Load for Matrix B
        // Threads load sequentially along the row (N dimension)
        int b_row = tileIdx * TILE + ty;
        if (b_row < K && col < N) {
            s_B[ty][tx] = B[b_row * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute phase: Padded width prevents bank conflicts during iteration
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        // Wait until all threads finish computing before loading the next tile
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1)/TILE, (M + TILE - 1)/TILE);
    matmul_tiled_kernel<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
