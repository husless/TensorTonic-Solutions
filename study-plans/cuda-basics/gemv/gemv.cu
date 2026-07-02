#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;

__global__ void gemv_kernel(const float* A, const float* x, float* y, int M, int N) {
    // Allocate shared memory for a tile of the vector x
    __shared__ float x_shared[BLOCK_SIZE];

    // Each thread handles one row of the matrix
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    float sum = 0.0f;

    // Loop over the matrix columns in tile steps
    for (int m = 0; m < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        // 1. Collaboratively load a tile of vector x into shared memory
        int x_idx = m * BLOCK_SIZE + tx;
        if (x_idx < N) {
            x_shared[tx] = x[x_idx];
        } else {
            x_shared[tx] = 0.0f; // Pad with 0 if out of bounds
        }

        // Synchronize to make sure the tile is completely loaded
        __syncthreads();

        // 2. Compute the dot product for the current tile
        if (row < M) {
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                int col_idx = m * BLOCK_SIZE + k;
                if (col_idx < N) {
                    sum += A[row * N + col_idx] * x_shared[k];
                }
            }
        }

        // Synchronize before loading the next tile of x
        __syncthreads();
    }
    // 3. Write the final accumulated result to global memory
    if (row < M) {
        y[row] = sum;
    }
}

extern "C" void solve(const float* A, const float* x, float* y, int M, int N) {
    dim3 threads(256);
    dim3 blocks((M + 255) / 256);
    gemv_kernel<<<blocks, threads>>>(A, x, y, M, N);
    cudaDeviceSynchronize();
}
