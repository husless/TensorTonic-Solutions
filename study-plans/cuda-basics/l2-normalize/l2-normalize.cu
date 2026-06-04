#include <cuda_runtime.h>


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

    if (lane == 0) shared[wid] = val; // Write warp sum to shared memory

    __syncthreads(); // Wait for all partial sums

    // Read from shared memory only if that warp existed
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (lane < num_warps) ? shared[lane] : 0.0f;

    if (wid == 0) val = warp_reduce_sum(val); // Final reduce within first warp

    return val;
}


__global__ void reduce_sq_sum(const float* input, float* sumv, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 1. Grid-stride loop accumulation in registers
    float local_sum = 0.0f;
    for(int i = idx; i < N; i += blockDim.x * gridDim.x) {
        local_sum += input[i]*input[i];
    }

    float sum = block_reduce_sum(local_sum);
    if (threadIdx.x == 0) {
        atomicAdd(sumv, sum);
    }
}

__global__ void divide_by_sqrt(const float* input, float* output, const float* sumv, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float norm_inv = rsqrtf(*sumv);
    // Prevent division by zero if the sub-array sums to 0
    //if (norm == 0.0f) norm = 1.0f; 

    for(int i=threadIdx.x + blockIdx.x*blockDim.x; i<N; i += blockDim.x * gridDim.x) {
        output[i] = input[i] * norm_inv;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    float* d_sum;
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    reduce_sq_sum<<<1, 256>>>(input, d_sum, N);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    divide_by_sqrt<<<blocks, threads>>>(input, output, d_sum, N);

    cudaDeviceSynchronize();
    cudaFree(d_sum);
}
