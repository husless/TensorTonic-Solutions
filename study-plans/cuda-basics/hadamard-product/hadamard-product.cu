#include <cuda_runtime.h>

__global__ void hadamard_kernel_vectorized(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    // Cast pointers to float4 for vectorized memory instructions
    const float4* A_4 = reinterpret_cast<const float4*>(A);
    const float4* B_4 = reinterpret_cast<const float4*>(B);
    float4* C_4 = reinterpret_cast<float4*>(C);

    // Process the bulk of the data 4 elements at a time
    int vectorized_size = size / 4;

    for (int i = idx; i < vectorized_size; i += stride) {
        float4 a = A_4[i];
        float4 b = B_4[i];
        float4 c;

        c.x = a.x * b.x;
        c.y = a.y * b.y;
        c.z = a.z * b.z;
        c.w = a.w * b.w;

        C_4[i] = c;
    }

    // Handle remaining elements if 'size' is not a multiple of 4
    int remainder_start = vectorized_size * 4;
    int remainder_idx = remainder_start + idx;

    for (int i = remainder_idx; i < size; i += stride) {
        C[i] = A[i] * B[i];
    }
}


extern "C" void solve(const float* A, const float* B, float* C, int M, int N) {
    int threads = 256;
    int size = M*N;
    
    int blocks = (size + threads - 1) / threads;
    hadamard_kernel_vectorized<<<blocks, threads>>>(A, B, C, size);
    cudaDeviceSynchronize();
}
