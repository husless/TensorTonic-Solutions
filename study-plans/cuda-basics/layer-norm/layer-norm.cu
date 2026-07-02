#include <cuda_runtime.h>
#include <math.h>

// Compile-time constant for GPU warp architecture
constexpr int WARP_SIZE = 32;
constexpr unsigned int FULL_WARP_MASK = 0xffffffff;

// Cooperatively reduce a float2 (sum and sum_sq) across the block
__device__ float2 block_reduce_sum(float2 val) {
    // Shared memory allocated for the maximum possible number of warps in a block (1024 / 32 = 32)
    static __shared__ float2 shared[WARP_SIZE]; 
    
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    // Warp-level reduction
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(FULL_WARP_MASK, val.x, offset);
        val.y += __shfl_down_sync(FULL_WARP_MASK, val.y, offset);
    }

    if (lane == 0) shared[wid] = val;

    __syncthreads(); // Wait for all warp reductions to finish
    
    // Read from shared memory only if that warp lane is valid for active warps
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : make_float2(0.0f, 0.0f);

    if (wid == 0) {
        // Final warp reduction
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            val.x += __shfl_down_sync(FULL_WARP_MASK, val.x, offset);
            val.y += __shfl_down_sync(FULL_WARP_MASK, val.y, offset);
        }
    }

    return val;
}

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float epsilon) {
    // One block maps to one row
    int row = blockIdx.x; 
    if (row >= M) return;

    // Pointer offsets to the start of this block's row
    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    // Step 1: Compute Mean
    float sum = 0.0f;
    float sum_sq = 0.0f;
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float val = row_input[col];
        sum += val;
        sum_sq += val * val;
    }
    
    // Cooperatively sum all threads
    float2 total = block_reduce_sum(make_float2(sum, sum_sq));

    // Shared variables to broadcast statistical scalars to all threads
    __shared__ float s_mean;
    __shared__ float s_rsqrt_var;
    if (threadIdx.x == 0) {
        float mean = total.x / N;
        // Numerical formulation: var = E[X^2] - (E[X])^2
        float variance = (total.y / N) - (mean * mean);
        
        // Clamp variance at 0.0f to guarantee numerical stability against minor floating-point precision drifts
        if (variance < 0.0f) variance = 0.0f; 

        s_mean = mean;
        s_rsqrt_var = rsqrtf(variance + epsilon);
    }
    __syncthreads();

    float mean = s_mean;
    float rsqrt_var = s_rsqrt_var;

    // Step 2: Read memory the second (and final) time to write out normalized values
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float g = gamma[col];
        float b = beta[col];
        row_output[col] = ((row_input[col] - mean) * rsqrt_var) * g + b;
    }
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, M, N, eps);
    cudaDeviceSynchronize();
}
