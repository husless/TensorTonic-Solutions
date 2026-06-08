#include <cuda_runtime.h>

__global__ void dropout_kernel(const float* input, const float* mask, float* output, float scale, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    const float4 * A4 = reinterpret_cast<const float4*>(input);
    const float4 * mask4 = reinterpret_cast<const float4*>(mask);
    float4 * out4 = reinterpret_cast<float4*>(output);

    int VN = N / 4;
    for(int i=idx; i<VN; i += stride) {
        float4 a = A4[i];
        float4 m = mask4[i];
        float4 o;
        o.x = a.x * m.x * scale;
        o.y = a.y * m.y * scale;
        o.z = a.z * m.z * scale;
        o.w = a.w * m.w * scale;
        out4[i] = o;
    }

    for(int i=VN*4 + idx; i<N; i += stride) {
        output[i] = input[i]*mask[i]*scale;
    }

}

extern "C" void solve(const float* input, const float* mask, float* output, float p, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float scale = 1.0 / (1.0 - p);
    dropout_kernel<<<blocks, threads>>>(input, mask, output, scale, N);
    cudaDeviceSynchronize();
}