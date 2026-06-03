#include <cuda_runtime.h>

__device__  __forceinline__  float swish(float x) {
    return x / (1.0f + expf(-x));
}

__global__ void swish_kernel(const float* input, float* output, int N) {
    // 1. Calculate global thread stride in elements, processing 4 elements per step
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    int stride = gridDim.x * blockDim.x * 4;

    // 2. Cast input/output pointers to float4 for vectorized memory access
    const float4* input4 = reinterpret_cast<const float4*>(input);
    float4* output4 = reinterpret_cast<float4*>(output);

    // 3. Process the bulk of data in chunks of 4 elements (vectorized loop)
    int n_minus_3 = N - 3; 
    for (int i = idx; i < n_minus_3; i += stride) {
        float4 in_val = input4[i / 4];
        float4 out_val;

        // Perform computation on each vector component
        out_val.x = swish(in_val.x);
        out_val.y = swish(in_val.y);
        out_val.z = swish(in_val.z);
        out_val.w = swish(in_val.w);

        output4[i / 4] = out_val;
    }

    // 4. Cleanup loop to handle unaligned or remaining edge elements safely
    int remainder_start = N - (N % 4);
    int thread_cleanup_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int cleanup_stride = gridDim.x * blockDim.x;

    for (int i = remainder_start + thread_cleanup_idx; i < N; i += cleanup_stride) {
        output[i] = swish(input[i]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    swish_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
