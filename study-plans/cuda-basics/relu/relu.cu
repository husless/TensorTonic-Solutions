#include <cuda_runtime.h>

__global__ void relu_kernel(const float* input, float* output, int N) {
    for (int i=threadIdx.x + blockIdx.x * blockDim.x; i<N; i += blockDim.x*gridDim.x) {
        output[i] = max(input[i], 0.0f);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}