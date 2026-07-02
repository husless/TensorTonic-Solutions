#include <cuda_runtime.h>
#include <float.h>

constexpr int WARP_SIZE = 32;
constexpr int FULL_MASK = 0xffffffff;

// Device helper function to reduce a single warp using shuffle instructions
__device__ __inline__ void warp_reduce_argmax(float &val, int &idx) {
    // Unroll loop at compile time since WARP_SIZE is a constexpr expression
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float next_val = __shfl_down_sync(FULL_MASK, val, offset);
        int next_idx = __shfl_down_sync(FULL_MASK, idx, offset);
        
        if (next_val > val) {
            val = next_val;
            idx = next_idx;
        } else if (next_val == val) {
            // Guard against uninitialized lanes (-1) ruining a valid tie
            if (idx == -1 || (next_idx != -1 && next_idx < idx)) {
                idx = next_idx;
            }
        }
    }
}

__global__ void argmax_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    // Shared memory allocated using explicit constexpr bounds
    __shared__ float s_vals[WARP_SIZE];
    __shared__ int s_idxs[WARP_SIZE];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    float local_val = -FLT_MAX;
    int local_idx = -1;
    if (gid < N) {
        local_val = input[gid];
        local_idx = gid;
    }
    
    // Step 1: Reduce within each independent warp
    warp_reduce_argmax(local_val, local_idx);

    // Step 2: Write the winner of each warp to shared memory
    if (lane == 0) {
        s_vals[warp_id] = local_val;
        s_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    // Step 3: The first warp reduces the shared memory warp-winners
    if (warp_id == 0) {
        local_val = (lane < (blockDim.x / WARP_SIZE)) ? s_vals[lane] : -FLT_MAX;
        local_idx = (lane < (blockDim.x / WARP_SIZE)) ? s_idxs[lane] : -1;

        warp_reduce_argmax(local_val, local_idx);

        // Step 4: Thread 0 writes the final block winner to global memory
        if (lane == 0) {
            block_vals[blockIdx.x] = local_val;
            block_idxs[blockIdx.x] = local_idx;
        }
    }
}

__global__ void argmax_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    __shared__ float s_vals[WARP_SIZE];
    __shared__ int s_idxs[WARP_SIZE];

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (tid < WARP_SIZE) {
        s_vals[tid] = -FLT_MAX;
        s_idxs[tid] = -1;
    }
    __syncthreads();

    float local_val = -FLT_MAX;
    int local_idx = -1;

    // Grid-stride loop inside the single block
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float val = block_vals[i];
        int idx = block_idxs[i];
        if (val > local_val) {
            local_val = val;
            local_idx = idx;
        } else if (val == local_val) {
            if (local_idx == -1 || (idx != -1 && idx < local_idx)) {
                local_idx = idx;
            }
        }
    }

    // Step 1: Warp reduction
    warp_reduce_argmax(local_val, local_idx);

    // Step 2: Store warp winners in shared memory
    if (lane == 0) {
        s_vals[warp_id] = local_val;
        s_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    // Step 3: Final warp reduction across warp winners
    if (warp_id == 0) {
        int active_warps = (blockDim.x / WARP_SIZE);
        
        local_val = (lane < active_warps) ? s_vals[lane] : -FLT_MAX;
        local_idx = (lane < active_warps) ? s_idxs[lane] : -1;

        warp_reduce_argmax(local_val, local_idx);

        if (lane == 0) {
            *result = local_idx;
        }
    }
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    argmax_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmax_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}
