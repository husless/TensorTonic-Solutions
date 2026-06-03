#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 256;

__global__ void dot_kernel(const float* A, const float* B, float* result, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

     // Accumulate grid-stride loop results directly into a local register
    float local_sum = 0.0f;
    for(int i = idx; i < N; i += blockDim.x * gridDim.x) {
        local_sum += A[i] * B[i];
    }

    __shared__ float buffer[BLOCK_SIZE];
    buffer[tid] = local_sum;

    for(int s=BLOCK_SIZE/2; s>0; s >>= 1) {
        __syncthreads();
        if(tid<s) {
            buffer[tid] += buffer[tid+s];
        }
    }

    if (tid==0) {
        atomicAdd(result, buffer[0]);
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threads = BLOCK_SIZE;
    int blocks = (N + threads - 1) / threads;
    cudaMemset(result, 0, sizeof(float));
    dot_kernel<<<blocks, threads>>>(A, B, result, N);
    cudaDeviceSynchronize();
}
