#pragma once

#include "kernel_types.cuh"
#include "ring_queue.cuh"
#include <cub/cub.cuh>
#include <cfloat>

namespace coraanns::detail {

__global__ void row_norms_fused_kernel(const float *__restrict__ A, int A_rows, int A_cols,
                                       const float *__restrict__ B, int B_rows, int B_cols,
                                       float *__restrict__ A_norms, float *__restrict__ B_norms) {
    extern __shared__ float sdata[];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int nWarps = (blockDim.x + 31) / 32;
    const int total_rows = A_rows + B_rows;

    for (int g = blockIdx.x; g < total_rows; g += gridDim.x) {
        const float *row_ptr;
        int cols;
        float *out_ptr;
        if (g < A_rows) {
            row_ptr = A + (size_t)g * A_cols;
            cols = A_cols;
            out_ptr = A_norms + g;
        } else {
            const int br = g - A_rows;
            row_ptr = B + (size_t)br * B_cols;
            cols = B_cols;
            out_ptr = B_norms + br;
        }
        float sum = 0.0f;
        for (int c = tid; c < cols; c += blockDim.x) {
            sum += row_ptr[c] * row_ptr[c];
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, off);
        if (lane == 0)
            sdata[warp_id] = sum;
        __syncthreads();
        if (warp_id == 0) {
            float warp_sum = (tid < nWarps) ? sdata[tid] : 0.0f;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                warp_sum += __shfl_down_sync(0xffffffff, warp_sum, off);
            if (tid == 0)
                *out_ptr = warp_sum;
        }
        __syncthreads();
    }
}

__global__ void add_norms_kernel(float *distances, const float *query_norms,
                                 const float *centroid_norms, int Q, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Q * K) {
        int q = idx / K;
        int k = idx % K;
        distances[idx] = distances[idx] + query_norms[q] + centroid_norms[k];
    }
}

template <typename T> __device__ __forceinline__ T warpReduceSum(T val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename T>
__global__ void computeQueryFactorsWarpKernel(const T *d_query, float *d_G_k1xSumq,
                                              float *d_G_kbxSumq, size_t num_queries, size_t D,
                                              size_t ex_bits) {
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int query_idx = blockIdx.x * (blockDim.x / 32) + warp_id;

    if (query_idx >= num_queries)
        return;
    float c_1 = -static_cast<float>((1 << 1) - 1) / 2.0f;
    float c_b = -static_cast<float>((1 << (ex_bits + 1)) - 1) / 2.0f;
    const T *query = d_query + query_idx * D;
    T sum = 0;
    for (int i = lane_id; i < D; i += 32)
        sum += query[i];
    sum = warpReduceSum(sum);
    if (lane_id == 0) {
        d_G_k1xSumq[query_idx] = sum * c_1;
        d_G_kbxSumq[query_idx] = sum * c_b;
    }
}

template <typename T>
void computeQueryFactors(const T *d_query, float *d_G_k1xSumq, float *d_G_kbxSumq,
                         size_t num_queries, size_t D, size_t ex_bits, cudaStream_t stream) {
    const int threads_per_block = 256;
    const int blocks = (num_queries + (threads_per_block / 32) - 1) / (threads_per_block / 32);
    computeQueryFactorsWarpKernel<T><<<blocks, threads_per_block, 0, stream>>>(
        d_query, d_G_k1xSumq, d_G_kbxSumq, num_queries, D, ex_bits);
}

__global__ void flatten_pairs_kernel(const int *d_probe_idx, int num_queries, int nprobe,
                                     ClusterQueryPair *d_pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries * nprobe) {
        d_pairs[idx].cluster_idx = d_probe_idx[idx];
        d_pairs[idx].query_idx = idx / nprobe;
        d_pairs[idx].probe_idx = idx % nprobe;
    }
}

// Symmetric signed quantization: the high bit has negative weight. Reverse
// ballot bits to match the short codes' dimension-0-at-bit-31 layout.
__global__ void quantize_query_bitwise_kernel(const float *__restrict__ d_query, int num_queries,
                                              int D, uint32_t *__restrict__ d_q_bits,
                                              float *__restrict__ d_q_scale,
                                              float *__restrict__ d_q_min) {
    constexpr int Bq = RABITQ_BQ;
    const int q = blockIdx.x;
    const int tid = threadIdx.x;
    if (q >= num_queries)
        return;

    const float *q_ptr = d_query + (size_t)q * D;
    const int num_chunks = D / 32;

    float local_amax = 0.0f;
    for (int i = tid; i < D; i += blockDim.x) {
        local_amax = fmaxf(local_amax, fabsf(q_ptr[i]));
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_amax = fmaxf(local_amax, __shfl_down_sync(0xffffffff, local_amax, off));
    }
    __shared__ float s_amax_arr[32];
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int nWarps = blockDim.x >> 5;
    if (lane == 0)
        s_amax_arr[warp] = local_amax;
    __syncthreads();
    if (warp == 0) {
        float v = (tid < nWarps) ? s_amax_arr[tid] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
        }
        if (tid == 0)
            s_amax_arr[0] = v;
    }
    __syncthreads();
    const float vmax_abs = s_amax_arr[0];

    const float max_int_val = float((1 << (Bq - 1)) - 1);
    const float width = fmaxf(vmax_abs / max_int_val, 1e-30f);
    const float inv_width = 1.0f / width;

    if (tid == 0) {
        d_q_scale[q] = width;
        d_q_min[q] = 0.0f;
    }

    uint32_t *q_bits_q = d_q_bits + (size_t)q * Bq * num_chunks;
    const int qmin_int = -(1 << (Bq - 1));
    const int qmax_int = (1 << (Bq - 1)) - 1;

    for (int wchunk = warp; wchunk < num_chunks; wchunk += nWarps) {
        int dim_idx = wchunk * 32 + lane;
        float qv = q_ptr[dim_idx];

        int qint = __float2int_rn(qv * inv_width);
        qint = max(qmin_int, min(qmax_int, qint));

        unsigned u_bits = ((unsigned)qint) & ((1u << Bq) - 1u);

#pragma unroll
        for (int j = 0; j < Bq; ++j) {
            unsigned mask = __ballot_sync(0xffffffff, (u_bits >> j) & 1);
            unsigned mask_be = __brev(mask);
            if (lane == 0) {
                q_bits_q[j * num_chunks + wchunk] = mask_be;
            }
        }
    }
}

static inline void quantize_query_bitwise(const float *d_query, int num_queries, int D,
                                          uint32_t *d_q_bits, float *d_q_scale, float *d_q_min,
                                          cudaStream_t stream) {
    const int block_size = 128;
    quantize_query_bitwise_kernel<<<num_queries, block_size, 0, stream>>>(
        d_query, num_queries, D, d_q_bits, d_q_scale, d_q_min);
}

__global__ void precompute_lut_fp16_kernel(const float *d_query, half *d_lut, int D) {

    int q_idx = blockIdx.y;
    int tid = threadIdx.x;
    int num_nibbles = D / 4;

    for (int i = tid; i < num_nibbles * 16; i += blockDim.x) {
        int nibble_idx = i / 16;
        int pattern = i % 16;

        float sum = 0.0f;
        if (pattern & 0x8)
            sum += d_query[q_idx * D + nibble_idx * 4 + 0];
        if (pattern & 0x4)
            sum += d_query[q_idx * D + nibble_idx * 4 + 1];
        if (pattern & 0x2)
            sum += d_query[q_idx * D + nibble_idx * 4 + 2];
        if (pattern & 0x1)
            sum += d_query[q_idx * D + nibble_idx * 4 + 3];

        d_lut[q_idx * (num_nibbles * 16) + i] = __float2half(sum);
    }
}

} // namespace coraanns::detail
