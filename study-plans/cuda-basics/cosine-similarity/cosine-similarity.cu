#include <cuda_runtime.h>
#include <math.h>

constexpr int WARP_SIZE = 32;

// Overload operator+= for float3 to allow easy vector math
__device__ __inline__ void operator+=(float3 &a, const float3 &b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}


// Warp-level reduction using shuffle down
__device__ __inline__ float3 warp_reduce_sum_f3(float3 val) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val.x += __shfl_down_sync(0xffffffff, val.x, offset);
        val.y += __shfl_down_sync(0xffffffff, val.y, offset);
        val.z += __shfl_down_sync(0xffffffff, val.z, offset);
    }
    return val;
}

// Block-level reduction combining shared memory and warp shuffle
__device__ float3 block_reduce_sum_f3(float3 val) {
    __shared__ float3 shared[WARP_SIZE]; // Shared mem for the 32 warps
    int lane = threadIdx.x % WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    val = warp_reduce_sum_f3(val); // First level warp reduction

    if (lane == 0) { shared[wid] = val; } // Write warp sum to shared memory

    __syncthreads(); // Wait for all partial sums

    // Read from shared memory only if that warp existed
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (lane < num_warps) ? shared[lane] : make_float3(0.0f, 0.0f, 0.0f);

    if (wid == 0) { val = warp_reduce_sum_f3(val); }// Final reduce within first warp

    __syncthreads();

    return val;
}

__global__ void cosine_partials_kernel(const float* A, const float* B, float* scratch, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    const float4 * A4 = reinterpret_cast<const float4*>(A);
    const float4 * B4 = reinterpret_cast<const float4*>(B);

    // 1. Grid-stride loop accumulation in registers
    // float3 (dot, norm_a, norm_b)
    float3 local_val = make_float3(0.0f, 0.0f, 0.0f);

    int VN = N / 4;
    for(int i = idx; i < VN; i += stride) {
        float4 a4 = A4[i];
        float4 b4 = B4[i];
        
        local_val += make_float3(a4.x*b4.x, a4.x*a4.x, b4.x*b4.x);
        local_val += make_float3(a4.y*b4.y, a4.y*a4.y, b4.y*b4.y);
        local_val += make_float3(a4.z*b4.z, a4.z*a4.z, b4.z*b4.z);
        local_val += make_float3(a4.w*b4.w, a4.w*a4.w, b4.w*b4.w);
    }

    for(int i=VN*4 + idx; i<N; i += stride) {
        float a = A[i];
        float b = B[i];
        local_val += make_float3(a*b, a*a, b*b);
    }

    float3 value = block_reduce_sum_f3(local_val);
    if (threadIdx.x == 0) {
        atomicAdd(scratch, value.x);
        atomicAdd(scratch+1, value.y);
        atomicAdd(scratch+2, value.z);
    }
}

__global__ void cosine_finalize_kernel(const float* scratch, float* result) {
    if (threadIdx.x==0 && blockIdx.x==0) {
        *result = scratch[0] * rsqrtf(scratch[1]) * rsqrtf(scratch[2]);
    }
}

extern "C" void solve(const float* A, const float* B, float* result, int N) {
    float* scratch;
    cudaMalloc(&scratch, 3 * sizeof(float));
    cudaMemset(scratch, 0, 3 * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    if (blocks > 1024) blocks = 1024;

    cosine_partials_kernel<<<blocks, threads>>>(A, B, scratch, N);
    cosine_finalize_kernel<<<1, 1>>>(scratch, result);

    cudaDeviceSynchronize();
    cudaFree(scratch);
}