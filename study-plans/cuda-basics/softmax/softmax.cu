#include <cuda_runtime.h>
#include <math_constants.h>
#include <math.h>

#define WARP_SIZE 32

// Block reduction for Maximum
__device__ __forceinline__ float block_reduce_max(float value, float* shared_mem) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // 1. Warp reduction
    unsigned int mask = __ballot_sync(0xffffffff, threadIdx.x < blockDim.x);
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(mask, value, offset));
    }

    // 2. Write to shared memory
    if (lane == 0) shared_mem[wid] = value;
    __syncthreads();

    // 3. Final warp reduction
    if (wid == 0) {
        value = (threadIdx.x < num_warps) ? shared_mem[lane] : -CUDART_INF_F;
        unsigned int stage3_mask = __ballot_sync(0xffffffff, threadIdx.x < num_warps);
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            value = fmaxf(value, __shfl_down_sync(stage3_mask, value, offset));
        }
    }

    // 4. Broadcast
    if (threadIdx.x == 0) shared_mem[0] = value;
    __syncthreads();

    return shared_mem[0];
}

// Block reduction for Sum
__device__ __forceinline__ float block_reduce_sum(float value, float* shared_mem) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // 1. Warp reduction
    unsigned int mask = __ballot_sync(0xffffffff, threadIdx.x < blockDim.x);
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(mask, value, offset);
    }

    // 2. Write to shared memory
    if (lane == 0) shared_mem[wid] = value;
    __syncthreads();

    // 3. Final warp reduction
    if (wid == 0) {
        value = (threadIdx.x < num_warps) ? shared_mem[lane] : 0.0f;
        unsigned int stage3_mask = __ballot_sync(0xffffffff, threadIdx.x < num_warps);
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            value += __shfl_down_sync(stage3_mask, value, offset);
        }
    }

    // 4. Broadcast
    if (threadIdx.x == 0) shared_mem[0] = value;
    __syncthreads();

    return shared_mem[0];
}


// Global variables for block-to-block communication
__device__ float g_global_max = -INFINITY;;
__device__ float g_global_sum = 0.0f;

// Helper to perform atomic max on float variables
__device__ __forceinline__ void atomicMaxFloat(float* addr, float val) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_int, assumed,
                        __float_as_int(fmaxf(__int_as_float(assumed), val)));
    } while (assumed != old);
}


__global__ void softmax_phase1_max(const float* __restrict__ input, int size) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    float local_max = -CUDART_INF_F;
    for (int i = tid; i < size; i += stride) {
        local_max = fmaxf(local_max, input[i]);
    }

    local_max = block_reduce_max(local_max, shared_mem);

    // The first thread of each block updates the global maximum
    if (threadIdx.x == 0) {
        atomicMaxFloat(&g_global_max, local_max);
    }
}


__global__ void softmax_phase2_sum(const float* __restrict__ input, float* __restrict__ output, int size) {
    extern __shared__ float shared_mem[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Read the finalized global max computed in Phase 1
    float max_val = g_global_max; 
    float local_sum = 0.0f;

    for (int i = tid; i < size; i += stride) {
        float val = __expf(input[i] - max_val);
        output[i] = val; // Store partial exp value
        local_sum += val;
    }

    local_sum = block_reduce_sum(local_sum, shared_mem);

    // The first thread of each block updates the global sum
    if (threadIdx.x == 0) {
        atomicAdd(&g_global_sum, local_sum);
    }
}


__global__ void softmax_phase3_divide(float* __restrict__ output, int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Read the finalized global sum computed in Phase 2
    float inv_sum = 1.0f / (g_global_sum + 1e-6f);

    for (int i = tid; i < size; i += stride) {
        output[i] *= inv_sum;
    }
}





extern "C" void solve(const float* d_input, float* d_output, int size) {
    int threads_per_block = 256;
    // Dynamic grid sizing based on array size
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid > 1024) blocks_per_grid = 1024; // Cap grid size to avoid atomic saturation

    int num_warps = (threads_per_block + 31) / 32;
    size_t shared_mem_size = num_warps * sizeof(float);

    // Reset global device variables before starting
    float h_init_max =  -INFINITY;;
    float h_init_sum = 0.0f;
    cudaMemcpyToSymbol(g_global_max, &h_init_max, sizeof(float));
    cudaMemcpyToSymbol(g_global_sum, &h_init_sum, sizeof(float));

    // Execute the 3-step pipeline
    softmax_phase1_max<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_input, size);
    
    // Implicit or explicit synchronization between steps is required 
    // so Phase 2 reads the completely finished Phase 1 maximum.
    softmax_phase2_sum<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(d_input, d_output, size);
    
    softmax_phase3_divide<<<blocks_per_grid, threads_per_block>>>(d_output, size);
    
    cudaDeviceSynchronize();
     
}