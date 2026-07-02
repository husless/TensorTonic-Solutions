#include <cuda_runtime.h>
#include <math.h>


constexpr int WARP_SIZE = 32;


// Warp reduction for block-wide sum
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block reduction using shared memory for warp results
__device__ inline float block_reduce_sum(float val) {
    static __shared__ float shared[WARP_SIZE]; // Max 32 warps for 1024 threads
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }

    __syncthreads();

    // Read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0.0f;

    if (wid == 0) {
        val = warp_reduce_sum(val);
    }

    return val;
}



__global__ void rms_norm_kernel(const float* input, const float* gamma, float* output, int M, int N, float epsilon) {
    // Each block handles exactly one row
    int row = blockIdx.x;
    if (row >= M) return;

    // Point to the start of this row
    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    float sum_sq = 0.0f;

    // Step 1: Accumulate sum of squares for this row (coarse-grained loop if N > blockDim.x)
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        float val = row_input[col];
        sum_sq += val * val;
    }

    // Step 2: Reduce across the block
    sum_sq = block_reduce_sum(sum_sq);

    // Step 3: Broadcast RMS reciprocal to all threads in the block
    __shared__ float rrms;
    if (threadIdx.x == 0) {
        float mean_sq = sum_sq / N;
        rrms = rsqrtf(mean_sq + epsilon); // Reciprocal RMS
    }
    __syncthreads();

    // Step 4: Normalize and apply learnable scale gamma
    for (int col = threadIdx.x; col < N; col += blockDim.x) {
        row_output[col] = row_input[col] * rrms * gamma[col];
    }
}

extern "C" void solve(const float* input, const float* gamma, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    rms_norm_kernel<<<blocks, threads>>>(input, gamma, output, M, N, eps);
    cudaDeviceSynchronize();
}