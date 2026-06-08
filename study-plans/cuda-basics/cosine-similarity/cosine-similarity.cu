#include <cuda_runtime.h>
#include <math.h>

constexpr int WARP_SIZE = 32;


// Warp-level reduction using shuffle down
__device__ __inline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction combining shared memory and warp shuffle
__device__ float block_reduce_sum(float val) {
    __shared__ float shared[WARP_SIZE]; // Shared mem for the 32 warps
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val); // First level warp reduction

    if (lane == 0) { shared[wid] = val; } // Write warp sum to shared memory

    __syncthreads(); // Wait for all partial sums

    // Read from shared memory only if that warp existed
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (lane < num_warps) ? shared[lane] : 0.0f;

    if (wid == 0) { val = warp_reduce_sum(val); }// Final reduce within first warp

    __syncthreads();

    return val;
}

__global__ void cosine_partials_kernel(const float* A, const float* B, float* scratch, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 1. Grid-stride loop accumulation in registers
    float local_dot = 0.0f;
    float local_norm_a = 0.0f;
    float local_norm_b = 0.0f;
    for(int i = idx; i < N; i += blockDim.x * gridDim.x) {
        float a = A[i];
        float b = B[i];
        local_dot += a*b;
        local_norm_a += a*a;
        local_norm_b += b*b;
    }

    float dot = block_reduce_sum(local_dot);
    float norm_a = block_reduce_sum(local_norm_a);
    float norm_b = block_reduce_sum(local_norm_b);
    if (threadIdx.x == 0) {
        atomicAdd(scratch, dot);
        atomicAdd(scratch+1, norm_a);
        atomicAdd(scratch+2, norm_b);
    }
}

__global__ void cosine_finalize_kernel(const float* scratch, float* result) {
    if (threadIdx.x==0 && blockIdx.x==0) {
        *result = scratch[0] * rsqrtf(scratch[1]) * rsqrtf(scratch[2]);
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    float* scratch;
    cudaMalloc(&scratch, 3 * sizeof(float));
    cudaMemset(scratch, 0, 3 * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    cosine_partials_kernel<<<blocks, threads>>>(A, B, scratch, N);
    cosine_finalize_kernel<<<1, 1>>>(scratch, result);

    cudaDeviceSynchronize();
    cudaFree(scratch);
}