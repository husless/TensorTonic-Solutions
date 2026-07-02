#include <cuda_runtime.h>
#include <float.h>

constexpr int WARP_SIZE = 32;
constexpr int FULL_MASK = 0xffffffff;

struct GreaterThan {
    __device__ __inline__ bool operator()(float a, float b) const { return a > b; }
    __device__ __inline__ float init_val() const { return -FLT_MAX; }
};

struct LessThan {
    __device__ __inline__ bool operator()(float a, float b) const { return a < b; }
    __device__ __inline__ float init_val() const { return FLT_MAX; }
};

template <typename Comp>
__device__ __inline__ void warp_reduce_idx(float &val, int &idx, Comp comp) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        float next_val = __shfl_down_sync(FULL_MASK, val, offset);
        int next_idx = __shfl_down_sync(FULL_MASK, idx, offset);
        
        if (comp(next_val, val)) {
            val = next_val;
            idx = next_idx;
        } else if (next_val == val) {
            // Keep the lower index on ties. Guard against padding (-1).
            if (idx == -1 || (next_idx != -1 && next_idx < idx)) {
                idx = next_idx;
            }
        }
    }
}

template <typename Comp>
__global__ void argmin_kernel(const float* input, float* block_vals, int* block_idxs, int N, Comp comp) {
    __shared__ float s_vals[WARP_SIZE];
    __shared__ int s_idxs[WARP_SIZE];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    float local_val = comp.init_val();
    int local_idx = -1;
    if (gid < N) {
        local_val = input[gid];
        local_idx = gid;
    }

    warp_reduce_idx(local_val, local_idx, comp);

    if (lane == 0) {
        s_vals[warp_id] = local_val;
        s_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int active_warps = (blockDim.x / WARP_SIZE);
        
        local_val = (lane < active_warps) ? s_vals[lane] : comp.init_val();
        local_idx = (lane < active_warps) ? s_idxs[lane] : -1;

        warp_reduce_idx(local_val, local_idx, comp);

        if (lane == 0) {
            block_vals[blockIdx.x] = local_val;
            block_idxs[blockIdx.x] = local_idx;
        }
    }
}

template <typename Comp>
__global__ void argmin_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks, Comp comp) {
    __shared__ float s_vals[WARP_SIZE];
    __shared__ int s_idxs[WARP_SIZE];

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    if (tid < WARP_SIZE) {
        s_vals[tid] = comp.init_val();
        s_idxs[tid] = -1;
    }
    __syncthreads();

    float local_val = comp.init_val();
    int local_idx = -1;

    for (int i = tid; i < num_blocks; i += blockDim.x) {
        float val = block_vals[i];
        int idx = block_idxs[i];
        
        if (comp(val, local_val)) {
            local_val = val;
            local_idx = idx;
        } else if (val == local_val) {
            if (local_idx == -1 || (idx != -1 && idx < local_idx)) {
                local_idx = idx;
            }
        }
    }

    warp_reduce_idx(local_val, local_idx, comp);

    if (lane == 0) {
        s_vals[warp_id] = local_val;
        s_idxs[warp_id] = local_idx;
    }
    __syncthreads();

    if (warp_id == 0) {
        int active_warps = (blockDim.x / WARP_SIZE);
        
        local_val = (lane < active_warps) ? s_vals[lane] : comp.init_val();
        local_idx = (lane < active_warps) ? s_idxs[lane] : -1;

        warp_reduce_idx(local_val, local_idx, comp);

        if (lane == 0) {
            *result = local_idx;
        }
    }
}


extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    argmin_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N, LessThan{});
    argmin_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks, LessThan{});
    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}
