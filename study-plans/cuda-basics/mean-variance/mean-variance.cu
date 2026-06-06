#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// Structure to track Welford's running state
struct WelfordData {
    double count; // Using double to handle large counts easily in math
    double mean;
    double m2;
};

// Inline device function to combine two Welford states (Chan's formula)
__device__ __inline__ WelfordData combine_welford(WelfordData a, WelfordData b) {
    if (a.count == 0.0) return b;
    if (b.count == 0.0) return a;

    WelfordData out;
    out.count = a.count + b.count;
    double delta = b.mean - a.mean;
    out.mean = a.mean + delta * (b.count / out.count);
    out.m2 = a.m2 + b.m2 + (delta * delta) * (a.count * b.count / out.count);
    return out;
}

// Inline device function for warp-level Welford reduction
__device__ __inline__ WelfordData warp_reduce_welford(WelfordData val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        WelfordData remote;
        remote.count = __shfl_down_sync(0xffffffff, val.count, offset);
        remote.mean  = __shfl_down_sync(0xffffffff, val.mean, offset);
        remote.m2    = __shfl_down_sync(0xffffffff, val.m2, offset);
        
        val = combine_welford(val, remote);
    }
    return val;
}

// Kernel 1: Numerically stable block reduction using Welford + Shuffle
__global__ void welford_reduction_kernel(const float* __restrict__ d_in, 
                                         double* __restrict__ d_block_count,
                                         double* __restrict__ d_block_mean, 
                                         double* __restrict__ d_block_m2, 
                                         size_t n) {
    // Shared buffers for warp leaders
    __shared__ double s_count[BLOCK_SIZE / WARP_SIZE];
    __shared__ double s_mean[BLOCK_SIZE / WARP_SIZE];
    __shared__ double s_m2[BLOCK_SIZE / WARP_SIZE];

    size_t tid = threadIdx.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t grid_size = blockDim.x * gridDim.x;

    WelfordData local = {0.0, 0.0, 0.0};

    // Grid-stride loop: Update local thread state using standard Welford
    while (i < n) {
        double val = (double)d_in[i];
        local.count += 1.0;
        double delta = val - local.mean;
        local.mean += delta / local.count;
        local.m2 += delta * (val - local.mean);
        i += grid_size;
    }

    // Step 1: Reduce within each warp
    local = warp_reduce_welford(local);

    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    // Step 2: Write warp leaders to shared memory
    if (lane == 0) {
        s_count[warp_id] = local.count;
        s_mean[warp_id]  = local.mean;
        s_m2[warp_id]    = local.m2;
    }
    __syncthreads();

    // Step 3: Warp 0 aggregates the shared warp-level results
    if (warp_id == 0) {
        bool isValidWarp = (tid < blockDim.x / WARP_SIZE);
        local.count = isValidWarp ? s_count[lane] : 0.0;
        local.mean  = isValidWarp ? s_mean[lane]  : 0.0;
        local.m2    = isValidWarp ? s_m2[lane]    : 0.0;

        local = warp_reduce_welford(local);

        // Step 4: Block leader writes to global intermediate buffers
        if (tid == 0) {
            d_block_count[blockIdx.x] = local.count;
            d_block_mean[blockIdx.x]  = local.mean;
            d_block_m2[blockIdx.x]    = local.m2;
        }
    }
}

// Kernel 2: Tiny single-thread finalization kernel
__global__ void welford_finalize_kernel(const double* __restrict__ d_block_count,
                                        const double* __restrict__ d_block_mean, 
                                        const double* __restrict__ d_block_m2, 
                                        float* __restrict__ d_mean_out, 
                                        float* __restrict__ d_var_out, 
                                        size_t num_blocks) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        WelfordData total = {0.0, 0.0, 0.0};

        // Combine the results of all blocks sequentially
        for (size_t i = 0; i < num_blocks; ++i) {
            WelfordData block_data;
            block_data.count = d_block_count[i];
            block_data.mean  = d_block_mean[i];
            block_data.m2    = d_block_m2[i];

            total = combine_welford(total, block_data);
        }

        // Population Variance = M2 / N
        double variance = (total.count > 0.0) ? (total.m2 / total.count) : 0.0;

        // Write scalar results to 1-element device buffers
        *d_mean_out = (float)total.mean;
        *d_var_out  = (float)variance;
    }
}


void compute_mean_variance_welford(const float* d_in, size_t n, float* d_mean_out, float* d_var_out) {
    if (n == 0) return;

    int threads_per_block = BLOCK_SIZE;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    if (blocks_per_grid > 1024) blocks_per_grid = 1024; 

    // Allocate intermediate block accumulators
    double *d_block_count, *d_block_mean, *d_block_m2;
    cudaMalloc(&d_block_count, blocks_per_grid * sizeof(double));
    cudaMalloc(&d_block_mean, blocks_per_grid * sizeof(double));
    cudaMalloc(&d_block_m2, blocks_per_grid * sizeof(double));

    // Pass 1: Block-level stable reductions
    welford_reduction_kernel<<<blocks_per_grid, threads_per_block>>>(
        d_in, d_block_count, d_block_mean, d_block_m2, n
    );

    // Pass 2: Final single-thread reduction sweep
    welford_finalize_kernel<<<1, 1>>>(
        d_block_count, d_block_mean, d_block_m2, d_mean_out, d_var_out, blocks_per_grid
    );

    // Clean up
    cudaFree(d_block_count);
    cudaFree(d_block_mean);
    cudaFree(d_block_m2);
}


extern "C" void solve(const float* input, float* mean_out, float* var_out, int N) {
    compute_mean_variance_welford(input, N, mean_out, var_out);
}