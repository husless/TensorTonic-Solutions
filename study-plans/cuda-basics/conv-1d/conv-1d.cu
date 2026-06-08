#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;


__global__ void conv1d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int N,
    int kN
) {
    // each block cooperatively loads its output span
    // plus a trailing kN-1 halo into __shared__ memory once,
    // then every thread reads its window from shared memory
    extern __shared__ float buffer_s[];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int outN = N - kN + 1;

    if (idx < N) {
        buffer_s[tid] = input[idx];
    }else {
        buffer_s[tid] = 0.0f;
    }
    // halo loading
    if (tid < kN-1) {
        int right_s_idx = tid + blockDim.x; // Position right after main block data
        int right_g_idx = idx + blockDim.x; // Correct global index for halo element
        
        if (right_g_idx < N) {
            buffer_s[right_s_idx] = input[right_g_idx];
        } else {
            buffer_s[right_s_idx] = 0.0f;
        }
    }

    __syncthreads();

    if (idx < outN) {
        float s = 0.0f;
        for(int a = 0; a<kN; ++a) {
            s += kernel[a] * buffer_s[tid + a];
        }
        output[idx] = s;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int N, int kN) {
    int outN = N - kN + 1;
    if (outN <= 0) { 
        return; // Guard against invalid dimensions
    }

    int threads = BLOCK_SIZE;
    dim3 blocks((outN + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int shared_mem_size = (kN - 1 + BLOCK_SIZE) * sizeof(float);
    conv1d_kernel<<<blocks, threads, shared_mem_size>>>(input, kernel, output, N, kN);
    cudaDeviceSynchronize();
}
