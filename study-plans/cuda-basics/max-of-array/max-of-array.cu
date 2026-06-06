#include <cuda_runtime.h>
#include <cfloat>
#include <limits>

constexpr int WARP_SIZE = 32;


// Inline device function for warp-level reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    // Each thread gets the value from its neighbor 'offset' positions away
    // 0xffffffff is the mask indicating all threads in the warp are active
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 8));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 4));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 2));
    val = fmaxf(val, __shfl_down_sync(0xffffffff, val, 1));
    return val; // Lane 0 holds the final max for this warp
}


// Internal device kernel (Warp Shuffle Reduction)
__global__ void max_warp_shuffle_kernel(const float* __restrict__ d_in, float* __restrict__ d_out, int n, float lowest_val) {
    __shared__ float warp_maxes[WARP_SIZE]; // Accommodates up to 1024 threads (32 warps)

    unsigned int tid = threadIdx.x;
    unsigned int lane = tid % WARP_SIZE;
    unsigned int warp_id = tid / WARP_SIZE;

    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float local_max = lowest_val;

    for (int i=gid; i < n; i += blockDim.x * gridDim.x) {
        local_max = fmaxf(local_max, d_in[i]);
    }

    // Warp-level reduction
    float shuffle_val = warp_reduce_max(local_max);

    if (lane == 0) {
        warp_maxes[warp_id] = shuffle_val;
    }
    __syncthreads();

    // Final block-level reduction inside the first warp
    if (warp_id == 0) {
        unsigned int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
        float block_max = (lane < num_warps) ? warp_maxes[lane] : lowest_val;
        
        block_max = warp_reduce_max(block_max);

        if (lane == 0) {
            d_out[blockIdx.x] = block_max;
        }
    }
}

// Production C++ Wrapper Class
template <int BLOCK_SIZE=256>
class GPUReducer {
private:
    const int threads_per_block = BLOCK_SIZE;
    const int elements_per_block = threads_per_block; // 2048 elements

public:

    /**
     * Executes the multi-pass maximum reduction engine.
     * @param d_input Pointer to the original input array already resident on the GPU.
     * @param n Number of elements in the array.
     * @param stream Optional CUDA stream for concurrent execution.
     * @param Pointer to the device memory containing the single maximum value element.
     */
    void find_max(const float* d_input, float *result, int n) {

        if (n <= 0) return;

        // Base case: Only 1 element
        if (n == 1) {
            cudaMemcpy(result, d_input, sizeof(float), cudaMemcpyDeviceToHost);
            return;
        }

        int current_size = n;
        int first_grid_size = (current_size + threads_per_block - 1) / threads_per_block;
        int second_grid_size = (first_grid_size + threads_per_block - 1) / threads_per_block;
        
        float* d_buf1 = nullptr;
        float* d_buf2 = nullptr;
        
        cudaMalloc(&d_buf1, first_grid_size * sizeof(float));
        if (first_grid_size > 1) {
            cudaMalloc(&d_buf2, second_grid_size * sizeof(float));
        }
        
        const float* d_current_input = d_input;
        float* d_current_output = d_buf1;
        float lowest_value = std::numeric_limits<float>::lowest();
        
        while (current_size > 1) {
            int blocks_per_grid = (current_size + threads_per_block - 1) / threads_per_block;
            
            // Launch kernel asynchronously on the designated stream
            max_warp_shuffle_kernel<<<blocks_per_grid, threads_per_block>>>(
                d_current_input, 
                d_current_output, 
                current_size,
                lowest_value
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
};


extern "C" void solve(const float* input, float* result, int N) {
    if (N<=0) { return; }
    
    GPUReducer<256> reducer;
    reducer.find_max(input, result, N);
}
