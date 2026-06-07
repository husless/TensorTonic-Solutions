#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float value) {
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }
    return value;
}

__global__ void dot_kernel(const float* A, const float* B, float* result, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // 1. Accumulate grid-stride loop results directly into a local register
    float local_sum = 0.0f;
    for(int i = idx; i < N; i += blockDim.x * gridDim.x) {
        local_sum += A[i] * B[i];
    }

    // 2. Warp-level reduction using shuffle instructions (no shared memory)
    float warp_sum = warp_reduce_sum(local_sum);

    // 3. Shared memory for block-level reduction (only 8 elements used per block)
    __shared__ float buffer[WARP_SIZE];
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    if (lane == 0) {
        buffer[warp_id] = warp_sum;
    }
    __syncthreads();

    // 4. Final reduction of warp totals by the first warp
    if (warp_id == 0) {
        // Read from shared memory only if the warp actually wrote data
        int num_warps = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
        float block_sum = (lane < num_warps) ? buffer[lane] : 0.0f;
        
        // Reduce the first warp
        block_sum = warp_reduce_sum(block_sum);

        // 5. Single atomic addition per block
        if (lane == 0) {
            atomicAdd(result, block_sum);
        }
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    cudaMemsetAsync(result, 0, sizeof(float));

    int threads = BLOCK_SIZE;
    int prospective_blocks = (N + threads - 1) / threads;
    int blocks = min(prospective_blocks, 320); // 320 blocks saturates most modern GPUs
    
    dot_kernel<<<blocks, threads>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
