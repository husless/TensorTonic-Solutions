#include <cuda_runtime.h>


#define M_SQRT1_2 0.707106781186547524401f


__device__ __forceinline__ float gelu_scalar(float x) {
    // Computes exact GELU: 0.5 * x * (1.0 + erf(x / sqrt(2)))
    return 0.5f * x * (1.0f + erff(x * M_SQRT1_2));
}


__global__ void gelu_kernel(const float* input, float* output, int N) {
    const float4* input_v = reinterpret_cast<const float4*>(input);
    float4* output_v = reinterpret_cast<float4*>(output);

    const int VN = N/4;

    int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (size_t idx = start_idx; idx < VN; idx += stride) {
        float4 in_val = input_v[idx];
        float4 out_val;

        out_val.x = gelu_scalar(in_val.x);
        out_val.y = gelu_scalar(in_val.y);
        out_val.z = gelu_scalar(in_val.z);
        out_val.w = gelu_scalar(in_val.w);

        output_v[idx] = out_val;
    }

    int tail_start = VN * 4;
    for (int idx = tail_start + start_idx; idx < N; idx += stride) {
        output[idx] = gelu_scalar(input[idx]);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    dim3 blocks((N + 255) / 256);
    gelu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}
