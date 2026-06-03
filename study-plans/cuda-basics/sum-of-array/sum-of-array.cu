#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;


__global__ void sum_kernel(const float* input, float* result, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

     // Accumulate grid-stride loop results directly into a local register
    float local_sum = 0.0f;
    for(int i = idx; i < N; i += blockDim.x * gridDim.x) {
        local_sum += input[i];
    }
    // 2. Intra-warp reduction using warp shuffle
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    // 3. Shared memory for communication between warp leaders
    __shared__ float warp_buffer[WARP_SIZE]; // Maximum 32 warps per block
    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (lane == 0) {
        warp_buffer[warp_id] = local_sum;
    }

    __syncthreads();

    // 4. Final reduction of warp totals by the first warp
    if (warp_id == 0) {
        // Read from shared memory only if the warp exists
        float final_sum = (tid < (blockDim.x / WARP_SIZE)) ? warp_buffer[lane] : 0.0f;
        
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            final_sum += __shfl_down_sync(0xFFFFFFFF, final_sum, offset);
        }

        // 5. Single atomic add per block to global memory
        if (tid == 0) {
            atomicAdd(result, final_sum);
        }
    }
}

extern "C" void solve(const float* input, float* result, int N) {
    int threads = BLOCK_SIZE;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(result, 0, sizeof(float));
    sum_kernel<<<blocks, threads>>>(input, result, N);
    cudaDeviceSynchronize();
}
