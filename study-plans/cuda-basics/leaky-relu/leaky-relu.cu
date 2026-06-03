#include <cuda_runtime.h>


__global__ void leaky_relu_kernel(const float* __restrict__ input, float* __restrict__ output, float alpha, int N) {
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
        out_val.x = fmaxf(in_val.x, alpha * in_val.x);
        out_val.y = fmaxf(in_val.y, alpha * in_val.y);
        out_val.z = fmaxf(in_val.z, alpha * in_val.z);
        out_val.w = fmaxf(in_val.w, alpha * in_val.w);

        output4[i / 4] = out_val;
    }

    // 4. Cleanup loop to handle unaligned or remaining edge elements safely
    int remainder_start = N - (N % 4);
    int thread_cleanup_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int cleanup_stride = gridDim.x * blockDim.x;

    for (int i = remainder_start + thread_cleanup_idx; i < N; i += cleanup_stride) {
        float x = input[i];
        output[i] = fmaxf(x, alpha * x);
    }
}


extern "C" void solve(const float* input, float* output, float alpha, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    leaky_relu_kernel<<<blocks, threads>>>(input, output, alpha, N);
    cudaDeviceSynchronize();
}