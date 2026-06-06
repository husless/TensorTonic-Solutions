#include <cuda_runtime.h>
#include <cfloat>
#include <limits>

constexpr int WARP_SIZE = 32;
constexpr int BLOCK_SIZE = 256;


// Inline device function for warp-level reduction
__device__ __forceinline__ float warp_reduce_min(float val) {
    // Each thread gets the value from its neighbor 'offset' positions away
    // 0xffffffff is the mask indicating all threads in the warp are active
    val = fminf(val, __shfl_down_sync(0xffffffff, val, 16));
    val = fminf(val, __shfl_down_sync(0xffffffff, val, 8));
    val = fminf(val, __shfl_down_sync(0xffffffff, val, 4));
    val = fminf(val, __shfl_down_sync(0xffffffff, val, 2));
    val = fminf(val, __shfl_down_sync(0xffffffff, val, 1));
    return val; // Lane 0 holds the final max for this warp
}


// Internal device kernel (Warp Shuffle Reduction)
__global__ void min_warp_shuffle_kernel(const float* __restrict__ d_in, float* __restrict__ d_out, int n, float largest_val) {
    __shared__ float shared[WARP_SIZE]; // Accommodates up to 1024 threads (32 warps)

    unsigned int tid = threadIdx.x;
    unsigned int lane = tid % WARP_SIZE;
    unsigned int warp_id = tid / WARP_SIZE;

    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float local_min = largest_val;

    for (int i=gid; i < n; i += blockDim.x * gridDim.x) {
        local_min = fminf(local_min, d_in[i]);
    }

    // Warp-level reduction
    float shuffle_val = warp_reduce_min(local_min);

    if (lane == 0) {
        shared[warp_id] = shuffle_val;
    }
    __syncthreads();

    // Final block-level reduction inside the first warp
    if (warp_id == 0) {
        unsigned int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        float block_min = (lane < num_warps) ? shared[lane] : largest_val;
        
        block_min = warp_reduce_min(block_min);

        if (lane == 0) {
            d_out[blockIdx.x] = block_min;
        }
    }
}

 void find_min(const float* d_input, float* result, int N) {
    if (N <= 0) return;

    // Base case: Only 1 element
    if (N == 1) {
        cudaMemcpy(result, d_input, sizeof(float), cudaMemcpyDeviceToHost);
        return;
    }

    int current_size = N;
    int first_grid_size = (current_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int second_grid_size = (first_grid_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
    float* d_buf1 = nullptr;
    float* d_buf2 = nullptr;
        
    cudaMalloc(&d_buf1, first_grid_size * sizeof(float));
    if (first_grid_size > 1) {
        cudaMalloc(&d_buf2, second_grid_size * sizeof(float));
    }
        
    const float* d_current_input = d_input;
    float* d_current_output = d_buf1;
    float largest_value = std::numeric_limits<float>::max();
        
    while (current_size > 1) {
        int blocks_per_grid = (current_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
        // Launch kernel asynchronously on the designated stream
        min_warp_shuffle_kernel<<<blocks_per_grid, BLOCK_SIZE>>>(
            d_current_input, 
            d_current_output, 
            current_size,
            largest_value
        );

        // Swap working pointers for ping-pong structure (No allocations inside loop)
        d_current_input = d_current_output;
        d_current_output = (d_current_output == d_buf1) ? d_buf2 : d_buf1;
        current_size = blocks_per_grid;
    }

    // Clean up the unused secondary buffer safely
    cudaMemcpy(result, d_current_input, sizeof(float), cudaMemcpyDeviceToHost);

    // Ensure ALL dynamically allocated internal buffers are cleaned up
    if (d_buf1 != nullptr) cudaFree(d_buf1);
    if (d_buf2 != nullptr) cudaFree(d_buf2);
}

extern "C" void solve(const float* input, float* result, int N) {
    find_min(input, result, N);
}
