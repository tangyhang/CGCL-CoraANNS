#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cfloat>
#include <cub/cub.cuh>
#include <omp.h>
#include <chrono>
#include <cuda_fp16.h>
#include <type_traits>
#include <cstdlib>
#include <stdexcept>

#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/matrix/select_k.cuh>
#include <rmm/cuda_stream_view.hpp>
#include <optional>
#include <climits>

#include "coraanns/search/hybrid.cuh"
#include "ring_queue.cuh"

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err = call;                                                                    \
        if (err != cudaSuccess) {                                                                  \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":"    \
                      << __LINE__ << std::endl;                                                    \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    } while (0)

struct PoolOffsetOp {
    int pool_size;
    __host__ __device__ __forceinline__ int operator()(int idx) const {
        return idx * pool_size;
    }
};

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

__global__ void select_topk_kernel(float *d_centroid_dists, int candidate_pool_size, int topk,
                                   float *d_final_vals, int *d_final_idx) {
    int query_id = blockIdx.x;
    int tid = threadIdx.x;
    float *row_dists = d_centroid_dists + query_id * candidate_pool_size;
    float *out_vals = d_final_vals + query_id * topk;
    int *out_idx = d_final_idx + query_id * topk;
    __shared__ float s_vals[256];
    __shared__ int s_idx[256];

    for (int k = 0; k < topk; k++) {
        float local_min = FLT_MAX;
        int local_idx = -1;
        for (int i = tid; i < candidate_pool_size; i += blockDim.x) {
            if (row_dists[i] < local_min) {
                local_min = row_dists[i];
                local_idx = i;
            }
        }
        s_vals[tid] = local_min;
        s_idx[tid] = local_idx;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride && s_vals[tid + stride] < s_vals[tid]) {
                s_vals[tid] = s_vals[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
            __syncthreads();
        }
        if (tid == 0) {
            out_vals[k] = s_vals[0];
            out_idx[k] = s_idx[0];
            if (s_idx[0] != -1) {
                row_dists[s_idx[0]] = FLT_MAX;
            }
        }
        __syncthreads();
    }
}

__global__ void init_local_indices_kernel(int *indices, int pool_size, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements)
        indices[idx] = idx % pool_size;
}

__global__ void gather_nprobe_kernel(int num_queries, int nprobe, int num_centroids,
                                     const float *d_sorted_dists, const int *d_sorted_indices,
                                     float *out_probe_vals, int *out_probe_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries * nprobe) {
        int src_idx = (idx / nprobe) * num_centroids + (idx % nprobe);
        out_probe_vals[idx] = d_sorted_dists[src_idx];
        out_probe_idx[idx] = d_sorted_indices[src_idx];
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

__global__ void computeInnerProducts1BitBitwiseProducerPersistent(
    const ComputeBitwiseKernelParams params, const GlobalAsyncBuffer d_global_buffer,
    QueueIndex *d_vram_write_count, const float *d_dynamic_thresholds, int *d_global_pair_counter) {
    QueueIndex queue_write_limit = d_global_buffer.capacity;
    constexpr int Bq = RABITQ_BQ;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int num_chunks = params.D / 32;

    while (true) {
        __shared__ int s_block_id;
        if (tid == 0)
            s_block_id = atomicAdd(d_global_pair_counter, 1);
        __syncthreads();

        int block_id = s_block_id;
        if (block_id >= params.num_pairs)
            break;

        ClusterQueryPair pair = params.d_sorted_pairs[block_id];
        int cluster_idx = pair.cluster_idx;
        int query_idx = pair.query_idx;
        int probe_idx = pair.probe_idx;

        size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
        size_t cluster_start_index = params.d_cluster_meta[cluster_idx].start_index;

        long long t_start = 0, t_compute = 0, t_sort = 0, t_write = 0;
        if (threadIdx.x == 0)
            t_start = clock64();

        extern __shared__ char smem[];
        uint32_t *s_qbits = reinterpret_cast<uint32_t *>(smem);
        float *s_compact_dist = reinterpret_cast<float *>(s_qbits + Bq * num_chunks);
        float *s_compact_ip = s_compact_dist + params.max_candidates_per_pair;
        int *s_compact_idx = reinterpret_cast<int *>(s_compact_ip + params.max_candidates_per_pair);

        const uint32_t *q_bits_gmem = params.d_q_bits + (size_t)query_idx * Bq * num_chunks;
        const int total_qbits_words = Bq * num_chunks;
        for (int i = tid; i < total_qbits_words; i += num_threads) {
            s_qbits[i] = q_bits_gmem[i];
        }

        __shared__ float s_current_thresh;
        __shared__ float s_q_scale;
        __shared__ float s_q_min;
        if (tid == 0) {
            s_current_thresh = d_dynamic_thresholds[query_idx];
            s_q_scale = params.d_q_scale[query_idx];
            s_q_min = params.d_q_min[query_idx];
        }
        __syncthreads();

        const float q_g_add =
            params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
        const float q_k1xsumq = params.d_G_k1xSumq[query_idx];
        const float q_g_error = sqrtf(q_g_add);
        const float delta_q = s_q_scale;
        const float qmin_q = s_q_min;
        int actual_cands = min((int)num_vectors_in_cluster, params.max_candidates_per_pair);
        const size_t short_code_length = num_chunks;

        const float cutoff = (s_current_thresh == FLT_MAX) ? FLT_MAX : s_current_thresh;

        __shared__ int s_compact_count;
        if (tid == 0)
            s_compact_count = 0;
        __syncthreads();

        for (int c = tid; c < actual_cands; c += num_threads) {

            int acc_pos[RABITQ_BQ - 1];
#pragma unroll
            for (int j = 0; j < Bq - 1; ++j)
                acc_pos[j] = 0;
            int acc_msb = 0;
            int popc_xb = 0;

#pragma unroll 1
            for (int w = 0; w < (int)short_code_length; ++w) {
                uint32_t xw = params.d_short_data[cluster_start_index * short_code_length +
                                                  w * num_vectors_in_cluster + c];
                popc_xb += __popc(xw);

#pragma unroll
                for (int j = 0; j < Bq - 1; ++j) {
                    uint32_t qj = s_qbits[j * num_chunks + w];
                    acc_pos[j] += __popc(xw & qj);
                }
                uint32_t q_msb = s_qbits[(Bq - 1) * num_chunks + w];
                acc_msb += __popc(xw & q_msb);
            }

            int ip_int = -(acc_msb << (Bq - 1));
#pragma unroll
            for (int j = 0; j < Bq - 1; ++j) {
                ip_int += (acc_pos[j] << j);
            }

            float exact_ip = delta_q * (float)ip_int + qmin_q * (float)popc_xb;

            float3 factors =
                reinterpret_cast<const float3 *>(params.d_short_factors)[cluster_start_index + c];
            float low_dist =
                factors.x + q_g_add + factors.y * (exact_ip + q_k1xsumq) - factors.z * q_g_error;

            if (low_dist < cutoff) {
                int pos = atomicAdd(&s_compact_count, 1);
                if (pos < params.max_candidates_per_pair) {
                    s_compact_dist[pos] = low_dist;
                    s_compact_ip[pos] = exact_ip;
                    s_compact_idx[pos] = c;
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0)
            t_compute = clock64();

        const int num_sort = params.topk * 2;

        __shared__ bool s_needs_sort;
        if (tid == 0)
            s_needs_sort = (s_compact_count > num_sort);
        __syncthreads();

        __shared__ int s_block_write_count;
        if (tid == 0)
            s_block_write_count = 0;
        __syncthreads();

        __shared__ uint16_t s_out_query[20];
        __shared__ uint16_t s_out_probe[20];
        __shared__ float s_out_exact_ip[20];
        __shared__ float s_out_low_dist[20];
        __shared__ uint32_t s_out_gid[20];

        if (s_needs_sort) {
            constexpr int BLOCK_THREADS = 256;
            constexpr int ITEMS_PER_THREAD = 4;
            typedef cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int> BlockRadixSort;
            __shared__ typename BlockRadixSort::TempStorage temp_storage;

            float thread_keys[ITEMS_PER_THREAD];
            int thread_vals[ITEMS_PER_THREAD];

            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                int idx = tid * ITEMS_PER_THREAD + i;
                if (idx < s_compact_count) {
                    thread_keys[i] = s_compact_dist[idx];
                    thread_vals[i] = idx;
                } else {
                    thread_keys[i] = FLT_MAX;
                    thread_vals[i] = -1;
                }
            }
            __syncthreads();

            BlockRadixSort(temp_storage).Sort(thread_keys, thread_vals);
            __syncthreads();

            for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
                int sorted_rank = tid * ITEMS_PER_THREAD + i;
                if (sorted_rank < num_sort && thread_vals[i] != -1 && thread_keys[i] < cutoff) {
                    int slot = atomicAdd(&s_block_write_count, 1);
                    if (slot < num_sort) {
                        int compact_idx = thread_vals[i];
                        int original_c = s_compact_idx[compact_idx];
                        s_out_query[slot] = query_idx;
                        s_out_probe[slot] = probe_idx;
                        s_out_exact_ip[slot] = s_compact_ip[compact_idx];
                        s_out_low_dist[slot] = thread_keys[i];
                        s_out_gid[slot] = cluster_start_index + original_c;
                    }
                }
            }
        } else {
            for (int i = tid; i < s_compact_count; i += blockDim.x) {
                int slot = atomicAdd(&s_block_write_count, 1);
                if (slot < num_sort) {
                    int original_c = s_compact_idx[i];
                    s_out_query[slot] = query_idx;
                    s_out_probe[slot] = probe_idx;
                    s_out_exact_ip[slot] = s_compact_ip[i];
                    s_out_low_dist[slot] = s_compact_dist[i];
                    s_out_gid[slot] = cluster_start_index + original_c;
                }
            }
        }
        __syncthreads();

        if (threadIdx.x == 0)
            t_sort = clock64();

        __shared__ QueueIndex s_block_write_base;
        int actual_write_count = min((int)s_block_write_count, num_sort);
        if (tid == 0 && actual_write_count > 0) {
            s_block_write_base = atomicAdd(d_vram_write_count, QueueIndex(actual_write_count));
        }
        __syncthreads();

        for (int i = tid; i < actual_write_count; i += blockDim.x) {
            QueueIndex global_slot = s_block_write_base + i;
            {
                uint32_t word0 = ((uint32_t)s_out_probe[i] << 16) | (uint32_t)s_out_query[i];
                uint32_t word1 = s_out_gid[i];
                uint32_t word2 = __float_as_uint(s_out_exact_ip[i]);
                uint32_t word3 = __float_as_uint(s_out_low_dist[i]);

                uint4 payload = make_uint4(word0, word1, word2, word3);
                coraanns::queue::device::publish_task(d_global_buffer, global_slot, payload,
                                                      queue_write_limit);
            }
        }

        if (threadIdx.x == 0) {
            t_write = clock64();
            if (block_id < 4 || block_id == params.num_pairs - 1 ||
                block_id == params.num_pairs / 2 || block_id == params.num_pairs / 4 ||
                block_id == (params.num_pairs * 3) / 4) {
                printf(
                    "[GPU Block %d] 1-Bit Bitwise IP: %lld cycles, Sort/Bypass: %lld cycles, Global Write: %lld cycles, cands=%d threads=%d\n",
                    block_id, t_compute - t_start, t_sort - t_compute, t_write - t_sort,
                    actual_cands, num_threads);
            }
        }
    }
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

namespace {
#ifndef RABITQ_VANGUARD_CAPACITY
#define RABITQ_VANGUARD_CAPACITY 512
#endif
static_assert(RABITQ_VANGUARD_CAPACITY == 512 || RABITQ_VANGUARD_CAPACITY == 1024,
              "Vanguard capacity must be 512 or 1024");
constexpr int VANGUARD_BLOCK_THREADS = 256;
constexpr int VANGUARD_ITEMS_PER_THREAD = 2 * RABITQ_VANGUARD_CAPACITY / VANGUARD_BLOCK_THREADS;
constexpr int VANGUARD_TILE_SIZE = VANGUARD_BLOCK_THREADS * VANGUARD_ITEMS_PER_THREAD;

constexpr int VANGUARD_MAX_KEEP = VANGUARD_TILE_SIZE / 2;

using VanguardBlockRadixSort =
    cub::BlockRadixSort<float, VANGUARD_BLOCK_THREADS, VANGUARD_ITEMS_PER_THREAD, int>;
using VanguardRadixTempStorage = VanguardBlockRadixSort::TempStorage;

constexpr int VANGUARD_RECOVERY_ITEMS_PER_THREAD = VANGUARD_MAX_KEEP / VANGUARD_BLOCK_THREADS;
using VanguardRecoveryRadixSort =
    cub::BlockRadixSort<int, VANGUARD_BLOCK_THREADS, VANGUARD_RECOVERY_ITEMS_PER_THREAD>;
using VanguardRecoveryRadixTempStorage = VanguardRecoveryRadixSort::TempStorage;

__host__ __device__ constexpr size_t align_up_size(size_t value, size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

size_t vanguard_dynamic_smem_bytes(int padded_dim) {
    const size_t lut_bytes = (size_t)(padded_dim / 4) * 16 * sizeof(half);
    const size_t cub_offset = align_up_size(lut_bytes, alignof(VanguardRadixTempStorage));
    const size_t best_offset =
        align_up_size(cub_offset + sizeof(VanguardRadixTempStorage), alignof(float));
    return best_offset + (size_t)VANGUARD_MAX_KEEP * sizeof(float) +
           (size_t)VANGUARD_MAX_KEEP * sizeof(int);
}

size_t vanguard_recovery_dynamic_smem_bytes(int padded_dim) {
    const size_t lut_bytes = (size_t)(padded_dim / 4) * 16 * sizeof(half);
    const size_t cub_offset = align_up_size(lut_bytes, alignof(VanguardRecoveryRadixTempStorage));
    const size_t seed_offset =
        align_up_size(cub_offset + sizeof(VanguardRecoveryRadixTempStorage), alignof(int));
    return seed_offset + (size_t)VANGUARD_MAX_KEEP * sizeof(int);
}
} // namespace

__device__ __forceinline__ float
vanguard_lut_inner_product(const uint32_t *short_data, const half *lut, size_t cluster_start,
                           size_t cluster_size, size_t short_code_length, int cluster_local_idx) {
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    const uint32_t *p = short_data + cluster_start * short_code_length + cluster_local_idx;

#pragma unroll 4
    for (int w = 0; w < (int)short_code_length; ++w) {
        const uint32_t chunk = *p;
        p += cluster_size;
        const int base_nibble = w * 8;

        acc0 += __half2float(lut[(base_nibble + 0) * 16 + ((chunk >> 28) & 0xF)]);
        acc1 += __half2float(lut[(base_nibble + 1) * 16 + ((chunk >> 24) & 0xF)]);
        acc0 += __half2float(lut[(base_nibble + 2) * 16 + ((chunk >> 20) & 0xF)]);
        acc1 += __half2float(lut[(base_nibble + 3) * 16 + ((chunk >> 16) & 0xF)]);
        acc0 += __half2float(lut[(base_nibble + 4) * 16 + ((chunk >> 12) & 0xF)]);
        acc1 += __half2float(lut[(base_nibble + 5) * 16 + ((chunk >> 8) & 0xF)]);
        acc0 += __half2float(lut[(base_nibble + 6) * 16 + ((chunk >> 4) & 0xF)]);
        acc1 += __half2float(lut[(base_nibble + 7) * 16 + ((chunk >> 0) & 0xF)]);
    }
    return acc0 + acc1;
}

__global__ void computeInnerProducts1BitLutProducerVanguard(
    const ComputeBitwiseKernelParams params, const GlobalAsyncBuffer d_global_buffer,
    QueueIndex *d_vram_write_count, const float *d_dynamic_thresholds, int *d_global_pair_counter,
    int *d_seed_indices, int *d_seed_counts) {
    QueueIndex queue_write_limit = d_global_buffer.capacity;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ __align__(256) char smem[];

    const size_t lut_bytes = (params.D / 4) * 16 * sizeof(half);
    const size_t cub_offset = align_up_size(lut_bytes, alignof(VanguardRadixTempStorage));
    const size_t best_offset =
        align_up_size(cub_offset + sizeof(VanguardRadixTempStorage), alignof(float));

    half *s_LUT = reinterpret_cast<half *>(smem);
    VanguardRadixTempStorage &radix_temp =
        *reinterpret_cast<VanguardRadixTempStorage *>(smem + cub_offset);
    float *s_best_dist = reinterpret_cast<float *>(smem + best_offset);
    int *s_best_idx = reinterpret_cast<int *>(s_best_dist + VANGUARD_MAX_KEEP);

    while (true) {

        __shared__ int s_block_id;
        if (tid == 0)
            s_block_id = atomicAdd(d_global_pair_counter, 1);
        __syncthreads();

        int block_id = s_block_id;
        if (block_id >= params.num_pairs)
            return;

        ClusterQueryPair pair = params.d_sorted_pairs[block_id];
        int cluster_idx = pair.cluster_idx;
        int query_idx = pair.query_idx;
        int probe_idx = pair.probe_idx;

        size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
        size_t cluster_start_index = params.d_cluster_meta[cluster_idx].start_index;

        int lut_elements = (params.D / 4) * 16;
        const half *q_global_lut = params.d_global_luts + query_idx * lut_elements;
        int num_vec_loads = lut_elements / 8;
        float4 *s_LUT_vec = (float4 *)s_LUT;
        const float4 *q_global_lut_vec = (const float4 *)q_global_lut;

        for (int i = tid; i < num_vec_loads; i += num_threads) {
            s_LUT_vec[i] = q_global_lut_vec[i];
        }
        __syncthreads();

        float q_g_add = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
        float q_k1xsumq = params.d_G_k1xSumq[query_idx];
        float q_g_error = sqrtf(q_g_add);

        __shared__ float s_current_thresh;
        if (tid == 0) {
            float thresh = d_dynamic_thresholds[query_idx];
            s_current_thresh = (thresh == FLT_MAX) ? FLT_MAX : thresh;
        }
        __syncthreads();
        float cutoff = s_current_thresh;
        size_t short_code_length = params.D / 32;

        const int num_keep = max(params.topk, 20);
        for (int i = tid; i < VANGUARD_MAX_KEEP; i += num_threads) {
            s_best_dist[i] = FLT_MAX;
            s_best_idx[i] = -1;
        }
        __syncthreads();

        for (size_t tile_base = 0; tile_base < num_vectors_in_cluster;
             tile_base += VANGUARD_TILE_SIZE) {
            float thread_keys[VANGUARD_ITEMS_PER_THREAD];
            int thread_vals[VANGUARD_ITEMS_PER_THREAD];

#pragma unroll
            for (int item = 0; item < VANGUARD_ITEMS_PER_THREAD; ++item) {
                const size_t c = tile_base + tid + item * VANGUARD_BLOCK_THREADS;
                if (c < num_vectors_in_cluster) {
                    const float exact_ip = vanguard_lut_inner_product(
                        params.d_short_data, s_LUT, cluster_start_index, num_vectors_in_cluster,
                        short_code_length, (int)c);
                    const float3 factors = reinterpret_cast<const float3 *>(
                        params.d_short_factors)[cluster_start_index + c];
                    const float low_dist = factors.x + q_g_add +
                                           factors.y * (exact_ip + q_k1xsumq) -
                                           factors.z * q_g_error;

                    thread_keys[item] = (low_dist < cutoff) ? low_dist : FLT_MAX;
                    thread_vals[item] = (low_dist < cutoff) ? (int)c : -1;
                } else {
                    thread_keys[item] = FLT_MAX;
                    thread_vals[item] = -1;
                }
            }

            VanguardBlockRadixSort(radix_temp).Sort(thread_keys, thread_vals);
            __syncthreads();

#pragma unroll
            for (int item = 0; item < VANGUARD_ITEMS_PER_THREAD; ++item) {
                const int rank = tid * VANGUARD_ITEMS_PER_THREAD + item;
                if (rank >= num_keep && rank < 2 * num_keep) {
                    thread_keys[item] = s_best_dist[rank - num_keep];
                    thread_vals[item] = s_best_idx[rank - num_keep];
                } else if (rank >= 2 * num_keep) {
                    thread_keys[item] = FLT_MAX;
                    thread_vals[item] = -1;
                }
            }

            VanguardBlockRadixSort(radix_temp).Sort(thread_keys, thread_vals);
            __syncthreads();

#pragma unroll
            for (int item = 0; item < VANGUARD_ITEMS_PER_THREAD; ++item) {
                const int rank = tid * VANGUARD_ITEMS_PER_THREAD + item;
                if (rank < num_keep) {
                    s_best_dist[rank] = thread_keys[item];
                    s_best_idx[rank] = thread_vals[item];
                }
            }
            __syncthreads();
        }

        __shared__ int s_output_count;
        __shared__ QueueIndex s_block_write_base;
        if (tid == 0) {
            int valid = 0;
            while (valid < num_keep && s_best_idx[valid] >= 0)
                ++valid;
            s_output_count = valid;
            d_seed_counts[block_id] = valid;
            s_block_write_base = (valid > 0) ? atomicAdd(d_vram_write_count, QueueIndex(valid)) : 0;
        }
        __syncthreads();

        for (int rank = tid; rank < s_output_count; rank += num_threads) {
            const int original_c = s_best_idx[rank];
            d_seed_indices[(size_t)block_id * VANGUARD_MAX_KEEP + rank] = original_c;
            const float exact_ip =
                vanguard_lut_inner_product(params.d_short_data, s_LUT, cluster_start_index,
                                           num_vectors_in_cluster, short_code_length, original_c);
            const float low_dist = s_best_dist[rank];
            const QueueIndex global_slot = s_block_write_base + rank;

            {
                const uint32_t word0 = ((uint32_t)probe_idx << 16) | (uint32_t)query_idx;
                const uint32_t word1 = (uint32_t)(cluster_start_index + original_c);
                const uint32_t word2 = __float_as_uint(exact_ip);
                const uint32_t word3 = __float_as_uint(low_dist);

                const uint4 payload = make_uint4(word0, word1, word2, word3);
                coraanns::queue::device::publish_task(d_global_buffer, global_slot, payload,
                                                      queue_write_limit);
            }
        }
        __syncthreads();
    }
}

__global__ void computeInnerProducts1BitLutProducerVanguardRecovery(
    const ComputeBitwiseKernelParams params, const GlobalAsyncBuffer d_global_buffer,
    QueueIndex *d_vram_write_count, const float *d_dynamic_thresholds, int *d_global_pair_counter,
    const int *d_seed_indices, const int *d_seed_counts) {
    QueueIndex queue_write_limit = d_global_buffer.capacity;
    const int tid = threadIdx.x;
    const int lut_elements = (params.D / 4) * 16;
    const int num_lut_loads = lut_elements / 8;
    const size_t short_code_length = params.D / 32;

    extern __shared__ __align__(256) char smem[];
    const size_t lut_bytes = (size_t)lut_elements * sizeof(half);
    const size_t cub_offset = align_up_size(lut_bytes, alignof(VanguardRecoveryRadixTempStorage));
    const size_t seed_offset =
        align_up_size(cub_offset + sizeof(VanguardRecoveryRadixTempStorage), alignof(int));

    half *s_lut = reinterpret_cast<half *>(smem);
    auto &sort_temp = *reinterpret_cast<VanguardRecoveryRadixTempStorage *>(smem + cub_offset);
    int *s_seed_indices = reinterpret_cast<int *>(smem + seed_offset);

    while (true) {
        __shared__ int s_pair_id;
        if (tid == 0)
            s_pair_id = atomicAdd(d_global_pair_counter, 1);
        __syncthreads();

        const int pair_id = s_pair_id;
        if (pair_id >= params.num_pairs)
            return;

        const ClusterQueryPair pair = params.d_sorted_pairs[pair_id];
        const int cluster_idx = pair.cluster_idx;
        const int query_idx = pair.query_idx;
        const int probe_idx = pair.probe_idx;
        const size_t cluster_size = params.d_cluster_meta[cluster_idx].num;
        const size_t cluster_start = params.d_cluster_meta[cluster_idx].start_index;

        const half *global_lut = params.d_global_luts + (size_t)query_idx * lut_elements;
        auto *s_lut_vec = reinterpret_cast<float4 *>(s_lut);
        const auto *global_lut_vec = reinterpret_cast<const float4 *>(global_lut);
        for (int i = tid; i < num_lut_loads; i += blockDim.x) {
            s_lut_vec[i] = global_lut_vec[i];
        }

        const int seed_count = d_seed_counts[pair_id];
        int seed_keys[VANGUARD_RECOVERY_ITEMS_PER_THREAD];
#pragma unroll
        for (int item = 0; item < VANGUARD_RECOVERY_ITEMS_PER_THREAD; ++item) {
            const int rank = tid * VANGUARD_RECOVERY_ITEMS_PER_THREAD + item;
            seed_keys[item] = (rank < seed_count)
                                  ? d_seed_indices[(size_t)pair_id * VANGUARD_MAX_KEEP + rank]
                                  : INT_MAX;
        }
        __syncthreads();
        VanguardRecoveryRadixSort(sort_temp).Sort(seed_keys);
        __syncthreads();

#pragma unroll
        for (int item = 0; item < VANGUARD_RECOVERY_ITEMS_PER_THREAD; ++item) {
            const int rank = tid * VANGUARD_RECOVERY_ITEMS_PER_THREAD + item;
            if (rank < VANGUARD_MAX_KEEP) {
                s_seed_indices[rank] = seed_keys[item];
            }
        }
        __syncthreads();

        const float q_g_add =
            params.d_centroid_distances[(size_t)query_idx * params.num_centroids + cluster_idx];
        const float q_k1xsumq = params.d_G_k1xSumq[query_idx];
        const float q_g_error = sqrtf(q_g_add);
        const float cutoff = d_dynamic_thresholds[query_idx];

        for (int c = tid; c < (int)cluster_size; c += blockDim.x) {
            int lo = 0;
            int hi = seed_count - 1;
            bool is_seed = false;
            while (lo <= hi) {
                const int mid = lo + ((hi - lo) >> 1);
                const int seed_idx = s_seed_indices[mid];
                if (seed_idx < c) {
                    lo = mid + 1;
                } else if (seed_idx > c) {
                    hi = mid - 1;
                } else {
                    is_seed = true;
                    break;
                }
            }
            if (is_seed)
                continue;

            const float exact_ip = vanguard_lut_inner_product(
                params.d_short_data, s_lut, cluster_start, cluster_size, short_code_length, c);
            const float3 factors =
                reinterpret_cast<const float3 *>(params.d_short_factors)[cluster_start + c];
            const float low_dist =
                factors.x + q_g_add + factors.y * (exact_ip + q_k1xsumq) - factors.z * q_g_error;

            if (low_dist < cutoff) {
                const QueueIndex slot = atomicAdd(d_vram_write_count, 1ULL);
                {
                    coraanns::queue::device::publish_task(
                        d_global_buffer, slot,
                        make_uint4(((uint32_t)probe_idx << 16) | (uint32_t)query_idx,
                                   (uint32_t)(cluster_start + c), __float_as_uint(exact_ip),
                                   __float_as_uint(low_dist)),
                        queue_write_limit);
                }
            }
        }
        __syncthreads();
    }
}

__global__ void computeInnerProducts1BitLutProducerPersistent(
    const ComputeBitwiseKernelParams params, const GlobalAsyncBuffer d_global_buffer,
    QueueIndex *d_vram_write_count, const float *d_dynamic_thresholds, int *d_global_pair_counter) {
    QueueIndex queue_write_limit = d_global_buffer.capacity;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem[];
    half *s_LUT = (half *)smem;

    while (true) {

        __shared__ int s_block_id;
        if (tid == 0) {
            s_block_id = atomicAdd(d_global_pair_counter, 1);
        }
        __syncthreads();

        int block_id = s_block_id;
        if (block_id >= params.num_pairs)
            return;

        ClusterQueryPair pair = params.d_sorted_pairs[block_id];

        int cluster_idx = pair.cluster_idx;
        int query_idx = pair.query_idx;
        int probe_idx = pair.probe_idx;

        const uint32_t num_vec = params.d_cluster_meta[cluster_idx].num;
        const uint32_t cstart = params.d_cluster_meta[cluster_idx].start_index;
        const uint32_t scl = params.D / 32;
        const uint32_t cbase = cstart * scl;

        int lut_elements = (params.D / 4) * 16;
        const half *q_global_lut = params.d_global_luts + query_idx * lut_elements;

        int num_vec_loads = lut_elements / 8;
        float4 *s_LUT_vec = (float4 *)s_LUT;
        const float4 *q_global_lut_vec = (const float4 *)q_global_lut;

        for (int i = tid; i < num_vec_loads; i += num_threads) {
            s_LUT_vec[i] = q_global_lut_vec[i];
        }
        __syncthreads();

        float q_g_add = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
        float q_k1xsumq = params.d_G_k1xSumq[query_idx];
        float q_g_error = sqrtf(q_g_add);

        for (int c = tid; c < num_vec; c += num_threads) {

            const uint32_t *p = params.d_short_data + cbase + c;
            float acc0 = 0.f, acc1 = 0.f;
#pragma unroll(4)
            for (uint32_t w = 0; w < scl; w++) {
                uint32_t chunk = __ldg(p);
                p += num_vec;
                const int bn = w * 8;
                acc0 += __half2float(s_LUT[(bn + 0) * 16 + ((chunk >> 28) & 0xF)]);
                acc1 += __half2float(s_LUT[(bn + 1) * 16 + ((chunk >> 24) & 0xF)]);
                acc0 += __half2float(s_LUT[(bn + 2) * 16 + ((chunk >> 20) & 0xF)]);
                acc1 += __half2float(s_LUT[(bn + 3) * 16 + ((chunk >> 16) & 0xF)]);
                acc0 += __half2float(s_LUT[(bn + 4) * 16 + ((chunk >> 12) & 0xF)]);
                acc1 += __half2float(s_LUT[(bn + 5) * 16 + ((chunk >> 8) & 0xF)]);
                acc0 += __half2float(s_LUT[(bn + 6) * 16 + ((chunk >> 4) & 0xF)]);
                acc1 += __half2float(s_LUT[(bn + 7) * 16 + ((chunk >> 0) & 0xF)]);
            }
            float exact_ip = acc0 + acc1;

            float3 factors = reinterpret_cast<const float3 *>(params.d_short_factors)[cstart + c];
            float low_dist =
                factors.x + q_g_add + factors.y * (exact_ip + q_k1xsumq) - factors.z * q_g_error;

            float cutoff = d_dynamic_thresholds[query_idx];
            if (low_dist < cutoff) {
                const QueueIndex global_slot = atomicAdd(d_vram_write_count, 1ULL);

                {
                    uint32_t word0 = ((uint32_t)probe_idx << 16) | (uint32_t)query_idx;
                    uint32_t word1 = cstart + c;
                    uint32_t word2 = __float_as_uint(exact_ip);
                    uint32_t word3 = __float_as_uint(low_dist);
                    uint4 payload = make_uint4(word0, word1, word2, word3);
                    coraanns::queue::device::publish_task(d_global_buffer, global_slot, payload,
                                                          queue_write_limit);
                }
            }
        }

        __syncthreads();
    }
}

#define CHUNK_MAJOR_MAX_VEC_PER_THREAD 6

__global__ void computeInnerProducts1BitLutChunkMajorStaged(const ComputeBitwiseKernelParams params,
                                                            const GlobalAsyncBuffer d_global_buffer,
                                                            QueueIndex *d_vram_write_count,
                                                            const float *d_dynamic_thresholds,
                                                            int *d_global_pair_counter) {
    QueueIndex queue_write_limit = d_global_buffer.capacity;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem[];
    half *s_LUT = (half *)smem;
    const int lut_elements = (params.D / 4) * 16;
    uint32_t *s_chunk = (uint32_t *)(s_LUT + lut_elements);

    const size_t short_code_length = params.D / 32;

    while (true) {

        __shared__ int s_block_id;
        if (tid == 0)
            s_block_id = atomicAdd(d_global_pair_counter, 1);
        __syncthreads();

        int block_id = s_block_id;
        if (block_id >= params.num_pairs)
            return;

        long long t_start = 0, t_stage = 0, t_end = 0;
        const bool do_print =
            (tid == 0 && (block_id < 4 || block_id == params.num_pairs - 1 ||
                          block_id == params.num_pairs / 2 || block_id == params.num_pairs / 4 ||
                          block_id == (params.num_pairs * 3) / 4));
        if (do_print)
            t_start = clock64();

        ClusterQueryPair pair = params.d_sorted_pairs[block_id];
        int cluster_idx = pair.cluster_idx;
        int query_idx = pair.query_idx;
        int probe_idx = pair.probe_idx;

        size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
        size_t cluster_start_index = params.d_cluster_meta[cluster_idx].start_index;

        const half *q_global_lut = params.d_global_luts + (size_t)query_idx * lut_elements;
        int num_lut_loads = lut_elements / 8;
        float4 *s_LUT_vec = (float4 *)s_LUT;
        const float4 *q_global_lut_vec = (const float4 *)q_global_lut;
        for (int i = tid; i < num_lut_loads; i += num_threads)
            s_LUT_vec[i] = q_global_lut_vec[i];
        __syncthreads();

        float q_g_add =
            params.d_centroid_distances[(size_t)query_idx * params.num_centroids + cluster_idx];
        float q_k1xsumq = params.d_G_k1xSumq[query_idx];
        float q_g_error = sqrtf(q_g_add);
        float cutoff = d_dynamic_thresholds[query_idx];

        float partial_ip[CHUNK_MAJOR_MAX_VEC_PER_THREAD];
#pragma unroll
        for (int i = 0; i < CHUNK_MAJOR_MAX_VEC_PER_THREAD; i++)
            partial_ip[i] = 0.0f;

        for (int w = 0; w < (int)short_code_length; w++) {

            for (int c = tid; c < (int)num_vectors_in_cluster; c += num_threads) {
                s_chunk[c] = __ldg(&params.d_short_data[cluster_start_index * short_code_length +
                                                        (size_t)w * num_vectors_in_cluster + c]);
            }
            __syncthreads();

            int base_nibble = w * 8;
            int vidx = 0;
            for (int c = tid; c < (int)num_vectors_in_cluster; c += num_threads, vidx++) {
                uint32_t chunk = s_chunk[c];
                partial_ip[vidx] +=
                    __half2float(s_LUT[(base_nibble + 0) * 16 + ((chunk >> 28) & 0xF)]) +
                    __half2float(s_LUT[(base_nibble + 1) * 16 + ((chunk >> 24) & 0xF)]) +
                    __half2float(s_LUT[(base_nibble + 2) * 16 + ((chunk >> 20) & 0xF)]) +
                    __half2float(s_LUT[(base_nibble + 3) * 16 + ((chunk >> 16) & 0xF)]) +
                    __half2float(s_LUT[(base_nibble + 4) * 16 + ((chunk >> 12) & 0xF)]) +
                    __half2float(s_LUT[(base_nibble + 5) * 16 + ((chunk >> 8) & 0xF)]) +
                    __half2float(s_LUT[(base_nibble + 6) * 16 + ((chunk >> 4) & 0xF)]) +
                    __half2float(s_LUT[(base_nibble + 7) * 16 + ((chunk >> 0) & 0xF)]);
            }
            __syncthreads();
        }

        if (do_print)
            t_stage = clock64();

        {
            int vidx = 0;
            for (int c = tid; c < (int)num_vectors_in_cluster; c += num_threads, vidx++) {
                float exact_ip = partial_ip[vidx];
                float3 factors = reinterpret_cast<const float3 *>(
                    params.d_short_factors)[cluster_start_index + c];
                float low_dist = factors.x + q_g_add + factors.y * (exact_ip + q_k1xsumq) -
                                 factors.z * q_g_error;

                if (low_dist < cutoff) {
                    const QueueIndex global_slot = atomicAdd(d_vram_write_count, 1ULL);

                    {
                        uint32_t word0 = ((uint32_t)probe_idx << 16) | (uint32_t)query_idx;
                        uint32_t word1 = (uint32_t)(cluster_start_index + c);
                        uint32_t word2 = __float_as_uint(exact_ip);
                        uint32_t word3 = __float_as_uint(low_dist);
                        coraanns::queue::device::publish_task(
                            d_global_buffer, global_slot, make_uint4(word0, word1, word2, word3),
                            queue_write_limit);
                    }
                }
            }
        }
    }
}

__global__ void computeInnerProducts1BitLutVecTile(const ComputeBitwiseKernelParams params,
                                                   const GlobalAsyncBuffer d_global_buffer,
                                                   QueueIndex *d_vram_write_count,
                                                   const float *d_dynamic_thresholds,
                                                   int *d_global_pair_counter) {
    QueueIndex queue_write_limit = d_global_buffer.capacity;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem[];
    half *s_LUT = (half *)smem;
    const int lut_elements = (params.D / 4) * 16;
    uint32_t *s_data = (uint32_t *)(s_LUT + lut_elements);

    const size_t short_code_length = params.D / 32;

    while (true) {

        __shared__ int s_block_id;
        if (tid == 0)
            s_block_id = atomicAdd(d_global_pair_counter, 1);
        __syncthreads();

        int block_id = s_block_id;
        if (block_id >= params.num_pairs)
            return;

        long long t_start = 0, t_lut = 0, t_end = 0;
        const bool do_print =
            (tid == 0 && (block_id < 4 || block_id == params.num_pairs - 1 ||
                          block_id == params.num_pairs / 2 || block_id == params.num_pairs / 4 ||
                          block_id == (params.num_pairs * 3) / 4));
        if (do_print)
            t_start = clock64();

        ClusterQueryPair pair = params.d_sorted_pairs[block_id];
        int cluster_idx = pair.cluster_idx;
        int query_idx = pair.query_idx;
        int probe_idx = pair.probe_idx;

        size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
        size_t cluster_start_index = params.d_cluster_meta[cluster_idx].start_index;

        const half *q_global_lut = params.d_global_luts + (size_t)query_idx * lut_elements;
        int num_lut_loads = lut_elements / 8;
        float4 *s_LUT_vec = (float4 *)s_LUT;
        const float4 *q_global_lut_vec = (const float4 *)q_global_lut;
        for (int i = tid; i < num_lut_loads; i += num_threads)
            s_LUT_vec[i] = q_global_lut_vec[i];
        __syncthreads();

        if (do_print)
            t_lut = clock64();

        float q_g_add =
            params.d_centroid_distances[(size_t)query_idx * params.num_centroids + cluster_idx];
        float q_k1xsumq = params.d_G_k1xSumq[query_idx];
        float q_g_error = sqrtf(q_g_add);
        float cutoff = d_dynamic_thresholds[query_idx];

        for (int c_base = 0; c_base < (int)num_vectors_in_cluster; c_base += num_threads) {
            int c = c_base + tid;
            bool in_range = (c < (int)num_vectors_in_cluster);

            if (in_range) {
#pragma unroll 4
                for (int w = 0; w < (int)short_code_length; w++) {
                    s_data[w * num_threads + tid] =
                        __ldg(&params.d_short_data[cluster_start_index * short_code_length +
                                                   (size_t)w * num_vectors_in_cluster + c]);
                }
            }
            __syncthreads();

            if (in_range) {
                float ip0 = 0.0f, ip1 = 0.0f, ip2 = 0.0f, ip3 = 0.0f;

#pragma unroll 4
                for (int w = 0; w < (int)short_code_length; w++) {
                    uint32_t chunk = s_data[w * num_threads + tid];
                    int base_nibble = w * 8;

                    ip0 += __half2float(s_LUT[(base_nibble + 0) * 16 + ((chunk >> 28) & 0xF)]);
                    ip1 += __half2float(s_LUT[(base_nibble + 1) * 16 + ((chunk >> 24) & 0xF)]);
                    ip2 += __half2float(s_LUT[(base_nibble + 2) * 16 + ((chunk >> 20) & 0xF)]);
                    ip3 += __half2float(s_LUT[(base_nibble + 3) * 16 + ((chunk >> 16) & 0xF)]);

                    ip0 += __half2float(s_LUT[(base_nibble + 4) * 16 + ((chunk >> 12) & 0xF)]);
                    ip1 += __half2float(s_LUT[(base_nibble + 5) * 16 + ((chunk >> 8) & 0xF)]);
                    ip2 += __half2float(s_LUT[(base_nibble + 6) * 16 + ((chunk >> 4) & 0xF)]);
                    ip3 += __half2float(s_LUT[(base_nibble + 7) * 16 + ((chunk >> 0) & 0xF)]);
                }

                float exact_ip = ip0 + ip1 + ip2 + ip3;
                float3 factors = reinterpret_cast<const float3 *>(
                    params.d_short_factors)[cluster_start_index + c];
                float low_dist = factors.x + q_g_add + factors.y * (exact_ip + q_k1xsumq) -
                                 factors.z * q_g_error;

                if (low_dist < cutoff) {
                    const QueueIndex global_slot = atomicAdd(d_vram_write_count, 1ULL);

                    {
                        uint32_t word0 = ((uint32_t)probe_idx << 16) | (uint32_t)query_idx;
                        uint32_t word1 = (uint32_t)(cluster_start_index + c);
                        uint32_t word2 = __float_as_uint(exact_ip);
                        uint32_t word3 = __float_as_uint(low_dist);
                        coraanns::queue::device::publish_task(
                            d_global_buffer, global_slot, make_uint4(word0, word1, word2, word3),
                            queue_write_limit);
                    }
                }
            }
            __syncthreads();
        }

        if (do_print) {
            t_end = clock64();
            printf(
                "[GPU Block %d VecTile] LUT: %lld cycles, Stage+Compute+Filter: %lld cycles, num_vec=%zu\n",
                block_id, t_lut - t_start, t_end - t_lut, num_vectors_in_cluster);
        }
    }
}

__global__ void computeInnerProducts1BitLutProducer(const ComputeBitwiseKernelParams params,
                                                    const GlobalAsyncBuffer d_global_buffer,
                                                    QueueIndex *d_vram_write_count,
                                                    const float *d_dynamic_thresholds)

{
    QueueIndex queue_write_limit = d_global_buffer.capacity;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    int block_id = blockIdx.x;
    if (block_id >= params.num_pairs)
        return;

    extern __shared__ char smem[];
    half *s_LUT = (half *)smem;

    long long t_start = 0, t_lut = 0, t_write = 0;
    if (threadIdx.x == 0 &&
        (block_id < 4 || block_id == params.num_pairs - 1 || block_id == params.num_pairs / 2 ||
         block_id == params.num_pairs / 4 || block_id == (params.num_pairs * 3) / 4)) {
        t_start = clock64();
    }

    ClusterQueryPair pair = params.d_sorted_pairs[block_id];
    int cluster_idx = pair.cluster_idx;
    int query_idx = pair.query_idx;
    int probe_idx = pair.probe_idx;

    size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
    size_t cluster_start_index = params.d_cluster_meta[cluster_idx].start_index;

    int lut_elements = (params.D / 4) * 16;
    const half *q_global_lut = params.d_global_luts + query_idx * lut_elements;

    int num_vec_loads = lut_elements / 8;
    float4 *s_LUT_vec = (float4 *)s_LUT;
    const float4 *q_global_lut_vec = (const float4 *)q_global_lut;

    for (int i = tid; i < num_vec_loads; i += num_threads) {
        s_LUT_vec[i] = q_global_lut_vec[i];
    }
    __syncthreads();

    if (threadIdx.x == 0 &&
        (block_id < 4 || block_id == params.num_pairs - 1 || block_id == params.num_pairs / 2 ||
         block_id == params.num_pairs / 4 || block_id == (params.num_pairs * 3) / 4)) {
        t_lut = clock64();
    }

    float q_g_add = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
    float q_k1xsumq = params.d_G_k1xSumq[query_idx];
    float q_g_error = sqrtf(q_g_add);
    float cutoff = d_dynamic_thresholds[query_idx];
    size_t short_code_length = params.D / 32;

    for (int c = tid; c < num_vectors_in_cluster; c += num_threads) {
        float exact_ip0 = 0.0f;
        float exact_ip1 = 0.0f;
        float exact_ip2 = 0.0f;
        float exact_ip3 = 0.0f;

#pragma unroll(4)
        for (int w = 0; w < short_code_length; w++) {
            uint32_t chunk = __ldg(&params.d_short_data[cluster_start_index * short_code_length +
                                                        w * num_vectors_in_cluster + c]);

            int base_nibble = w * 8;

            exact_ip0 += __half2float(s_LUT[(base_nibble + 0) * 16 + ((chunk >> 28) & 0xF)]);
            exact_ip1 += __half2float(s_LUT[(base_nibble + 1) * 16 + ((chunk >> 24) & 0xF)]);
            exact_ip2 += __half2float(s_LUT[(base_nibble + 2) * 16 + ((chunk >> 20) & 0xF)]);
            exact_ip3 += __half2float(s_LUT[(base_nibble + 3) * 16 + ((chunk >> 16) & 0xF)]);

            exact_ip0 += __half2float(s_LUT[(base_nibble + 4) * 16 + ((chunk >> 12) & 0xF)]);
            exact_ip1 += __half2float(s_LUT[(base_nibble + 5) * 16 + ((chunk >> 8) & 0xF)]);
            exact_ip2 += __half2float(s_LUT[(base_nibble + 6) * 16 + ((chunk >> 4) & 0xF)]);
            exact_ip3 += __half2float(s_LUT[(base_nibble + 7) * 16 + ((chunk >> 0) & 0xF)]);
        }

        float exact_ip = exact_ip0 + exact_ip1 + exact_ip2 + exact_ip3;
        float3 factors =
            reinterpret_cast<const float3 *>(params.d_short_factors)[cluster_start_index + c];
        float low_dist =
            factors.x + q_g_add + factors.y * (exact_ip + q_k1xsumq) - factors.z * q_g_error;

        if (low_dist < cutoff) {
            const QueueIndex global_slot = atomicAdd(d_vram_write_count, 1ULL);

            {
                uint32_t word0 = ((uint32_t)probe_idx << 16) | (uint32_t)query_idx;
                uint32_t word1 = cluster_start_index + c;
                uint32_t word2 = __float_as_uint(exact_ip);
                uint32_t word3 = __float_as_uint(low_dist);
                uint4 payload = make_uint4(word0, word1, word2, word3);
                coraanns::queue::device::publish_task(d_global_buffer, global_slot, payload,
                                                      queue_write_limit);
            }
        }
    }

    if (threadIdx.x == 0 &&
        (block_id < 4 || block_id == params.num_pairs - 1 || block_id == params.num_pairs / 2 ||
         block_id == params.num_pairs / 4 || block_id == (params.num_pairs * 3) / 4)) {
        t_write = clock64();
    }
}

template <int Q>
__global__
__launch_bounds__(256, Q == 4 ? 3 : 1) void computeInnerProducts1BitLutProducerPersistentQBatch(
    const ComputeBitwiseKernelParams params, const GlobalAsyncBuffer d_global_buffer,
    QueueIndex *d_vram_write_count, const float *d_dynamic_thresholds, int *d_global_pair_counter) {
    QueueIndex queue_write_limit = d_global_buffer.capacity;
    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int lut_elements = (params.D / 4) * 16;
    const int num_lut_loads = lut_elements / 8;
    const size_t scl = params.D / 32;

    extern __shared__ char smem[];
    half *s_LUT = (half *)smem;
    float4 *s_LUT_vec = (float4 *)s_LUT;

    while (true) {

        __shared__ int s_base_id;
        if (tid == 0)
            s_base_id = atomicAdd(d_global_pair_counter, Q);
        __syncthreads();
        const int base_id = s_base_id;
        if (base_id >= params.num_pairs)
            return;

        int n = params.num_pairs - base_id;
        if (n > Q)
            n = Q;

        ClusterQueryPair pr[Q];
#pragma unroll
        for (int i = 0; i < Q; i++)
            pr[i] = (i < n) ? params.d_sorted_pairs[base_id + i] : pr[0];

        bool fast = (n == Q);
#pragma unroll
        for (int i = 1; i < Q; i++)
            if (pr[i].cluster_idx != pr[0].cluster_idx)
                fast = false;

        if (fast) {

            const int cluster_idx = pr[0].cluster_idx;
            const size_t num_vec = params.d_cluster_meta[cluster_idx].num;
            const size_t cstart = params.d_cluster_meta[cluster_idx].start_index;
            const size_t cbase = cstart * scl;

#pragma unroll
            for (int q = 0; q < Q; q++) {
                const float4 *g =
                    (const float4 *)(params.d_global_luts + (size_t)pr[q].query_idx * lut_elements);
                float4 *s = s_LUT_vec + (size_t)q * num_lut_loads;
                for (int i = tid; i < num_lut_loads; i += num_threads)
                    s[i] = g[i];
            }
            __syncthreads();

            float q_g_add[Q], q_k1xsumq[Q], q_g_error[Q], cutoff[Q];
#pragma unroll
            for (int q = 0; q < Q; q++) {
                q_g_add[q] =
                    params.d_centroid_distances[(size_t)pr[q].query_idx * params.num_centroids +
                                                cluster_idx];
                q_k1xsumq[q] = params.d_G_k1xSumq[pr[q].query_idx];
                q_g_error[q] = sqrtf(q_g_add[q]);
                cutoff[q] = d_dynamic_thresholds[pr[q].query_idx];
            }

            for (int c = tid; c < (int)num_vec; c += num_threads) {

                float acc0[Q], acc1[Q];
#pragma unroll
                for (int q = 0; q < Q; q++) {
                    acc0[q] = 0.f;
                    acc1[q] = 0.f;
                }

                const uint32_t *p = params.d_short_data + cbase + c;
#pragma unroll(2)
                for (int w = 0; w < (int)scl; w++) {
                    const uint32_t chunk = __ldg(p);
                    p += num_vec;
                    const int bn = w * 8;

                    const int v0 = (chunk >> 28) & 0xF, v1 = (chunk >> 24) & 0xF,
                              v2 = (chunk >> 20) & 0xF, v3 = (chunk >> 16) & 0xF;
                    const int v4 = (chunk >> 12) & 0xF, v5 = (chunk >> 8) & 0xF,
                              v6 = (chunk >> 4) & 0xF, v7 = (chunk)&0xF;
#pragma unroll
                    for (int q = 0; q < Q; q++) {
                        const half *L = s_LUT + (size_t)q * lut_elements;
                        acc0[q] += __half2float(L[(bn + 0) * 16 + v0]);
                        acc1[q] += __half2float(L[(bn + 1) * 16 + v1]);
                        acc0[q] += __half2float(L[(bn + 2) * 16 + v2]);
                        acc1[q] += __half2float(L[(bn + 3) * 16 + v3]);
                        acc0[q] += __half2float(L[(bn + 4) * 16 + v4]);
                        acc1[q] += __half2float(L[(bn + 5) * 16 + v5]);
                        acc0[q] += __half2float(L[(bn + 6) * 16 + v6]);
                        acc1[q] += __half2float(L[(bn + 7) * 16 + v7]);
                    }
                }

                const float3 f =
                    reinterpret_cast<const float3 *>(params.d_short_factors)[cstart + c];
#pragma unroll
                for (int q = 0; q < Q; q++) {
                    const float ip = acc0[q] + acc1[q];
                    const float low =
                        f.x + q_g_add[q] + f.y * (ip + q_k1xsumq[q]) - f.z * q_g_error[q];
                    if (low < cutoff[q]) {
                        const QueueIndex slot = atomicAdd(d_vram_write_count, 1ULL);
                        {
                            coraanns::queue::device::publish_task(
                                d_global_buffer, slot,
                                make_uint4(((uint32_t)pr[q].probe_idx << 16) |
                                               (uint32_t)pr[q].query_idx,
                                           (uint32_t)(cstart + c), __float_as_uint(ip),
                                           __float_as_uint(low)),
                                queue_write_limit);
                        }
                    }
                }
            }
            __syncthreads();
        } else {

            for (int i = 0; i < n; i++) {
                const ClusterQueryPair pp = pr[i];
                const int cluster_idx = pp.cluster_idx;
                const size_t num_vec = params.d_cluster_meta[cluster_idx].num;
                const size_t cstart = params.d_cluster_meta[cluster_idx].start_index;
                const size_t cbase = cstart * scl;

                const float4 *g =
                    (const float4 *)(params.d_global_luts + (size_t)pp.query_idx * lut_elements);
                for (int k = tid; k < num_lut_loads; k += num_threads)
                    s_LUT_vec[k] = g[k];
                __syncthreads();

                const float q_g_add =
                    params.d_centroid_distances[(size_t)pp.query_idx * params.num_centroids +
                                                cluster_idx];
                const float q_k1xsumq = params.d_G_k1xSumq[pp.query_idx];
                const float q_g_error = sqrtf(q_g_add);
                const float cutoff = d_dynamic_thresholds[pp.query_idx];

                for (int c = tid; c < (int)num_vec; c += num_threads) {
                    float a0 = 0.f, a1 = 0.f;
                    const uint32_t *p = params.d_short_data + cbase + c;
#pragma unroll(2)
                    for (int w = 0; w < (int)scl; w++) {
                        const uint32_t chunk = __ldg(p);
                        p += num_vec;
                        const int bn = w * 8;
                        a0 += __half2float(s_LUT[(bn + 0) * 16 + ((chunk >> 28) & 0xF)]);
                        a1 += __half2float(s_LUT[(bn + 1) * 16 + ((chunk >> 24) & 0xF)]);
                        a0 += __half2float(s_LUT[(bn + 2) * 16 + ((chunk >> 20) & 0xF)]);
                        a1 += __half2float(s_LUT[(bn + 3) * 16 + ((chunk >> 16) & 0xF)]);
                        a0 += __half2float(s_LUT[(bn + 4) * 16 + ((chunk >> 12) & 0xF)]);
                        a1 += __half2float(s_LUT[(bn + 5) * 16 + ((chunk >> 8) & 0xF)]);
                        a0 += __half2float(s_LUT[(bn + 6) * 16 + ((chunk >> 4) & 0xF)]);
                        a1 += __half2float(s_LUT[(bn + 7) * 16 + ((chunk)&0xF)]);
                    }
                    const float ip = a0 + a1;
                    const float3 f =
                        reinterpret_cast<const float3 *>(params.d_short_factors)[cstart + c];
                    const float low = f.x + q_g_add + f.y * (ip + q_k1xsumq) - f.z * q_g_error;
                    if (low < cutoff) {
                        const QueueIndex slot = atomicAdd(d_vram_write_count, 1ULL);
                        {
                            coraanns::queue::device::publish_task(
                                d_global_buffer, slot,
                                make_uint4(((uint32_t)pp.probe_idx << 16) | (uint32_t)pp.query_idx,
                                           (uint32_t)(cstart + c), __float_as_uint(ip),
                                           __float_as_uint(low)),
                                queue_write_limit);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }
}

IVF_RaBitQ_Raw::IVF_RaBitQ_Raw() {
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamCreate(&stream_d2h));
    CUDA_CHECK(cudaStreamCreate(&stream_factors));

    CUDA_CHECK(cudaEventCreate(&event_rot_done));
    CUDA_CHECK(cudaEventCreate(&event_probe_done));
    CUDA_CHECK(cudaEventCreate(&event_factors_done));

    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventCreate(&filter_start_event));
    CUDA_CHECK(cudaEventCreate(&filter_stop_event));
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);
}

IVF_RaBitQ_Raw::~IVF_RaBitQ_Raw() {
    cublasDestroy(cublas_handle);
    CUDA_CHECK(cudaStreamDestroy(stream));
    cudaStreamDestroy(stream_d2h);
    cudaStreamDestroy(stream_factors);
    cudaEventDestroy(event_rot_done);
    cudaEventDestroy(event_probe_done);
    cudaEventDestroy(event_factors_done);
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(filter_start_event);
    cudaEventDestroy(filter_stop_event);
    if (d_centroids)
        CUDA_CHECK(cudaFree(d_centroids));
    if (d_rotator)
        CUDA_CHECK(cudaFree(d_rotator));
    if (d_short_data)
        CUDA_CHECK(cudaFree(d_short_data));
    if (d_short_factors)
        CUDA_CHECK(cudaFree(d_short_factors));
    if (d_cluster_meta)
        CUDA_CHECK(cudaFree(d_cluster_meta));
    free_workspace();
}

void IVF_RaBitQ_Raw::load_from_raw_pointers(size_t num_vecs, size_t d, size_t p_d,
                                            size_t n_centroids, size_t ex_b,
                                            const float *h_centroids, const float *h_rotator,
                                            const uint32_t *h_short_data,
                                            const float *h_short_factors,
                                            const uint8_t *h_long_codes, const float *h_ex_factors,
                                            const uint32_t *h_ids, const size_t *h_cluster_sizes) {
    this->num_vectors = num_vecs;
    this->dim = d;
    this->padded_dim = p_d;
    this->num_centroids = n_centroids;
    this->ex_bits = ex_b;

    std::vector<GPUClusterMeta> h_cluster_meta(num_centroids);
    size_t current_offset = 0;
    for (size_t i = 0; i < num_centroids; ++i) {
        h_cluster_meta[i] = GPUClusterMeta(h_cluster_sizes[i], current_offset);
        current_offset += h_cluster_sizes[i];
    }
    CUDA_CHECK(cudaMallocAsync(&d_cluster_meta, num_centroids * sizeof(GPUClusterMeta), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_cluster_meta, h_cluster_meta.data(),
                               num_centroids * sizeof(GPUClusterMeta), cudaMemcpyHostToDevice,
                               stream));

    size_t centroids_bytes = num_centroids * padded_dim * sizeof(float);
    CUDA_CHECK(cudaMallocAsync(&d_centroids, centroids_bytes, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_centroids, h_centroids, centroids_bytes, cudaMemcpyHostToDevice, stream));

    size_t rotator_bytes = dim * padded_dim * sizeof(float);
    CUDA_CHECK(cudaMallocAsync(&d_rotator, rotator_bytes, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(d_rotator, h_rotator, rotator_bytes, cudaMemcpyHostToDevice, stream));

    size_t short_data_bytes = num_vectors * (padded_dim / 32) * sizeof(uint32_t);
    CUDA_CHECK(cudaMallocAsync(&d_short_data, short_data_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_short_data, h_short_data, short_data_bytes, cudaMemcpyHostToDevice,
                               stream));

    size_t short_factors_bytes = num_vectors * 3 * sizeof(float);
    CUDA_CHECK(cudaMallocAsync(&d_short_factors, short_factors_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_short_factors, h_short_factors, short_factors_bytes,
                               cudaMemcpyHostToDevice, stream));

    h_long_codes_ref = h_long_codes;
    h_ex_factors_ref = h_ex_factors;
    h_pids_ref = h_ids;
}

void IVF_RaBitQ_Raw::set_queue_capacity(size_t capacity) {
    if (capacity == 0 || (capacity & (capacity - 1)) != 0 ||
        capacity > SIZE_MAX / sizeof(GlobalTaskItem)) {
        throw std::invalid_argument("queue-capacity must be a positive power of two");
    }
    if (h_global_buffer) {
        throw std::logic_error("set queue capacity before prepare_workspace");
    }
    queue_capacity_ = capacity;
}

void IVF_RaBitQ_Raw::set_queue_spill_capacity(size_t capacity) {
    if (capacity > SIZE_MAX / sizeof(SpillTaskItem)) {
        throw std::invalid_argument("queue-spill-capacity is too large");
    }
    if (h_global_buffer) {
        throw std::logic_error("set spill capacity before prepare_workspace");
    }
    queue_spill_capacity_ = capacity;
}

void IVF_RaBitQ_Raw::allocate_candidate_queue() {
    h_global_buffer = new GlobalAsyncBuffer;
    h_global_buffer->capacity = queue_capacity_;
    h_global_buffer->mask = queue_capacity_ - 1;
    h_global_buffer->spill_capacity = queue_spill_capacity_;
    CUDA_CHECK(cudaHostAlloc(&h_global_buffer->items, queue_capacity_ * sizeof(GlobalTaskItem),
                             cudaHostAllocMapped));
    CUDA_CHECK(
        cudaHostAlloc(&h_queue_progress, sizeof(coraanns::queue::Progress), cudaHostAllocMapped));
    new (h_queue_progress) coraanns::queue::Progress{};
    coraanns::queue::Progress *d_progress = nullptr;
    CUDA_CHECK(cudaHostGetDevicePointer(&d_progress, h_queue_progress, 0));
    h_global_buffer->progress = d_progress;
    h_global_write_count = &h_queue_progress->produced;
    h_gpu_done = &h_queue_progress->done;
    if (queue_spill_capacity_ != 0) {
        CUDA_CHECK(cudaMallocAsync(&h_global_buffer->spill_items,
                                   queue_spill_capacity_ * sizeof(SpillTaskItem), stream));
    }
    CUDA_CHECK(cudaMallocAsync(&h_global_buffer->spill_count, sizeof(QueueIndex), stream));
    CUDA_CHECK(cudaMallocAsync(&h_global_buffer->spill_overflow, sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&h_global_buffer->cached_write_limit, sizeof(QueueIndex), stream));
    CUDA_CHECK(cudaMallocAsync(&h_global_buffer->refresh_lock, sizeof(int), stream));
    GlobalAsyncBuffer device_queue = *h_global_buffer;
    CUDA_CHECK(cudaHostGetDevicePointer(&device_queue.items, h_global_buffer->items, 0));
    d_global_buffer = device_queue;
}

void IVF_RaBitQ_Raw::release_candidate_queue() {
    if (h_global_buffer) {
        cudaFreeHost(h_global_buffer->items);
        if (h_global_buffer->spill_items)
            cudaFree(h_global_buffer->spill_items);
        cudaFree(h_global_buffer->spill_count);
        cudaFree(h_global_buffer->spill_overflow);
        cudaFree(h_global_buffer->cached_write_limit);
        cudaFree(h_global_buffer->refresh_lock);
        delete h_global_buffer;
        h_global_buffer = nullptr;
    }
    if (h_queue_progress)
        cudaFreeHost(h_queue_progress);
    d_global_buffer = {};
    h_queue_progress = nullptr;
    h_global_write_count = nullptr;
    h_gpu_done = nullptr;
}

void IVF_RaBitQ_Raw::prepare_workspace(size_t max_queries, size_t max_topk, size_t max_nprobe) {
    free_workspace();
    ws_max_queries = max_queries;
    ws_max_topk = max_topk;
    ws_max_nprobe = max_nprobe;

    cudaHostAlloc(&h_rotated_queries, max_queries * padded_dim * sizeof(float),
                  cudaHostAllocDefault);
    cudaHostAlloc(&h_G_kbxSumq, max_queries * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void **)&h_probe_vals, max_queries * max_nprobe * sizeof(float),
                  cudaHostAllocDefault);
    allocate_candidate_queue();

    cudaHostAlloc((void **)&h_dynamic_thresholds, max_queries * sizeof(float), cudaHostAllocMapped);
    cudaHostGetDevicePointer((void **)&d_dynamic_thresholds, h_dynamic_thresholds, 0);

    std::memset(h_global_buffer->items, 0xFF, queue_capacity_ * sizeof(GlobalTaskItem));
    *h_global_write_count = 0;
    *h_gpu_done = 0;
    last_written_count = 0;

#pragma omp parallel for
    for (int q = 0; q < max_queries; q++)
        h_dynamic_thresholds[q] = FLT_MAX;

    CUDA_CHECK(cudaMallocAsync(&ws_d_queries, max_queries * padded_dim * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_raw_queries, max_queries * dim * sizeof(float), stream));
    CUDA_CHECK(
        cudaMallocAsync(&ws_d_centroid_dists, max_queries * num_centroids * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_q_norms, max_queries * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_c_norms, num_centroids * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_probe_vals, max_queries * max_nprobe * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_probe_idx, max_queries * max_nprobe * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_sorted_pairs,
                               max_queries * max_nprobe * sizeof(ClusterQueryPair), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_G_k1xSumq, max_queries * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_G_kbxSumq, max_queries * sizeof(float), stream));

    const size_t max_vanguard_depth = (max_nprobe < vanguard_size_) ? max_nprobe : vanguard_size_;
    const size_t max_vanguard_pairs = max_queries * max_vanguard_depth;
    if (max_vanguard_pairs > 0) {
        CUDA_CHECK(cudaMallocAsync(&ws_d_vanguard_seed_indices,
                                   max_vanguard_pairs * VANGUARD_MAX_KEEP * sizeof(int), stream));
        CUDA_CHECK(
            cudaMallocAsync(&ws_d_vanguard_seed_counts, max_vanguard_pairs * sizeof(int), stream));
    }

    CUDA_CHECK(cudaMallocAsync(&d_global_pair_counter, sizeof(int), stream));

    CUDA_CHECK(cudaMallocAsync(&d_vram_write_count, sizeof(QueueIndex), stream));

    const int num_chunks = padded_dim / 32;
    CUDA_CHECK(cudaMallocAsync(&ws_d_q_bits,
                               max_queries * RABITQ_BQ * num_chunks * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_q_scale, max_queries * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_q_min, max_queries * sizeof(float), stream));

    size_t lut_elements_per_q = (padded_dim / 4) * 16;
    CUDA_CHECK(cudaMallocAsync(&ws_d_global_luts, max_queries * lut_elements_per_q * sizeof(half),
                               stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void IVF_RaBitQ_Raw::free_workspace() {
    if (ws_d_queries)
        cudaFree(ws_d_queries);
    if (ws_d_raw_queries)
        cudaFree(ws_d_raw_queries);
    if (ws_d_centroid_dists)
        cudaFree(ws_d_centroid_dists);
    if (ws_d_centroid_dists_copy)
        cudaFree(ws_d_centroid_dists_copy);
    if (ws_d_q_norms)
        cudaFree(ws_d_q_norms);
    if (ws_d_c_norms)
        cudaFree(ws_d_c_norms);
    if (ws_d_probe_vals)
        cudaFree(ws_d_probe_vals);
    if (ws_d_probe_idx)
        cudaFree(ws_d_probe_idx);
    if (ws_d_sorted_pairs)
        cudaFree(ws_d_sorted_pairs);
    if (ws_d_G_k1xSumq)
        cudaFree(ws_d_G_k1xSumq);
    if (ws_d_G_kbxSumq)
        cudaFree(ws_d_G_kbxSumq);
    if (ws_d_centroid_dists_sorted)
        cudaFree(ws_d_centroid_dists_sorted);
    if (ws_d_centroid_indices_temp)
        cudaFree(ws_d_centroid_indices_temp);
    if (ws_d_centroid_indices_sorted)
        cudaFree(ws_d_centroid_indices_sorted);
    if (ws_d_vanguard_seed_indices)
        cudaFree(ws_d_vanguard_seed_indices);
    if (ws_d_vanguard_seed_counts)
        cudaFree(ws_d_vanguard_seed_counts);
    if (ws_d_global_luts)
        cudaFree(ws_d_global_luts);
    if (d_vram_write_count)
        cudaFree(d_vram_write_count);
    if (d_global_pair_counter)
        cudaFree(d_global_pair_counter);

    if (ws_d_q_bits)
        cudaFree(ws_d_q_bits);
    if (ws_d_q_scale)
        cudaFree(ws_d_q_scale);
    if (ws_d_q_min)
        cudaFree(ws_d_q_min);

    if (h_rotated_queries)
        cudaFreeHost(h_rotated_queries);
    if (h_G_kbxSumq)
        cudaFreeHost(h_G_kbxSumq);
    release_candidate_queue();
    if (h_dynamic_thresholds)
        cudaFreeHost(h_dynamic_thresholds);
    if (h_probe_vals)
        cudaFreeHost(h_probe_vals);

    ws_d_queries = nullptr;
    ws_d_centroid_dists = nullptr;
    ws_d_centroid_dists_copy = nullptr;
    ws_d_vanguard_seed_indices = nullptr;
    ws_d_vanguard_seed_counts = nullptr;
    ws_d_q_bits = nullptr;
    ws_d_q_scale = nullptr;
    ws_d_q_min = nullptr;
    ws_d_raw_queries = nullptr;
    ws_d_q_norms = nullptr;
    ws_d_c_norms = nullptr;
    ws_d_probe_vals = nullptr;
    ws_d_probe_idx = nullptr;
    ws_d_sorted_pairs = nullptr;
    ws_d_G_k1xSumq = nullptr;
    ws_d_G_kbxSumq = nullptr;
    ws_d_centroid_dists_sorted = nullptr;
    ws_d_centroid_indices_temp = nullptr;
    ws_d_centroid_indices_sorted = nullptr;
    ws_d_global_luts = nullptr;
    d_vram_write_count = nullptr;
    d_global_pair_counter = nullptr;
    h_rotated_queries = nullptr;
    h_G_kbxSumq = nullptr;
    h_dynamic_thresholds = nullptr;
    d_dynamic_thresholds = nullptr;
    h_probe_vals = nullptr;
}

void IVF_RaBitQ_Raw::reset_buffer(const float *h_queries, size_t num_queries) {

    CUDA_CHECK(cudaMemcpyAsync(ws_d_raw_queries, h_queries, num_queries * dim * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    coraanns::queue::device::reset_queue<<<1, 1, 0, stream>>>(d_global_buffer, d_vram_write_count);
    CUDA_CHECK(cudaGetLastError());
    *h_queue_progress = coraanns::queue::Progress{};
    last_spilled_count_ = 0;
#pragma omp parallel for
    for (int q = 0; q < num_queries; q++)
        h_dynamic_thresholds[q] = FLT_MAX;
    *h_global_write_count = 0;
    *h_gpu_done = 0;
}

void IVF_RaBitQ_Raw::search(const float *h_queries, size_t num_queries, size_t topk, size_t nprobe,
                            float *h_final_dists, uint32_t *h_final_pids, int batch_q) {

    float *d_queries = ws_d_queries;
    float *d_raw_queries = ws_d_raw_queries;
    float *d_centroid_dists = ws_d_centroid_dists;
    float *d_q_norms = ws_d_q_norms;
    float *d_c_norms = ws_d_c_norms;
    float *d_probe_vals = ws_d_probe_vals;
    int *d_probe_idx = ws_d_probe_idx;
    ClusterQueryPair *d_sorted_pairs = ws_d_sorted_pairs;
    float *d_G_k1xSumq = ws_d_G_k1xSumq;
    float *d_G_kbxSumq = ws_d_G_kbxSumq;

    float ms = 0.0f;
    last_gpu_filter_ms_ = 0.0f;
#define TICK()                                                                                     \
    do {                                                                                           \
        if (profiling_enabled_)                                                                    \
            CUDA_CHECK(cudaEventRecord(start_event, stream));                                      \
    } while (0)
#define TOCK(step_name)                                                                            \
    do {                                                                                           \
        if (profiling_enabled_) {                                                                  \
            CUDA_CHECK(cudaEventRecord(stop_event, stream));                                       \
            CUDA_CHECK(cudaEventSynchronize(stop_event));                                          \
            CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));                        \
            std::cout << "[Profile] " << step_name << " time: " << ms << " ms\n";                  \
        }                                                                                          \
    } while (0)

    TICK();

    const float alpha_rot = 1.0f;
    const float beta_rot = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, padded_dim, num_queries, dim, &alpha_rot,
                d_rotator, padded_dim, d_raw_queries, dim, &beta_rot, d_queries, padded_dim);

    CUDA_CHECK(cudaEventRecord(event_rot_done, stream));

    CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, event_rot_done, 0));
    CUDA_CHECK(cudaMemcpyAsync(h_rotated_queries, d_queries,
                               num_queries * padded_dim * sizeof(float), cudaMemcpyDeviceToHost,
                               stream_d2h));

    CUDA_CHECK(cudaMemsetAsync(d_global_pair_counter, 0, sizeof(int), stream));

    CUDA_CHECK(cudaStreamWaitEvent(stream_factors, event_rot_done, 0));
    computeQueryFactors<float>(d_queries, d_G_k1xSumq, d_G_kbxSumq, num_queries, padded_dim,
                               ex_bits, stream_factors);

    if (ip_kernel_type_ == IPKernelType::BITWISE_DECOMP) {
        quantize_query_bitwise(d_queries, num_queries, padded_dim, ws_d_q_bits, ws_d_q_scale,
                               ws_d_q_min, stream_factors);
    }

    CUDA_CHECK(cudaEventRecord(event_factors_done, stream_factors));
    CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, event_factors_done, 0));
    CUDA_CHECK(cudaMemcpyAsync(h_G_kbxSumq, d_G_kbxSumq, num_queries * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_d2h));

    const float alpha = -2.0f;
    const float beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, num_centroids, num_queries, padded_dim,
                &alpha, d_centroids, padded_dim, d_queries, padded_dim, &beta, d_centroid_dists,
                num_centroids);

    int norm_block_size = 256;
    size_t norm_shared_mem = ((norm_block_size + 31) / 32) * sizeof(float);
    row_norms_fused_kernel<<<num_centroids + num_queries, norm_block_size, norm_shared_mem,
                             stream>>>(d_queries, num_queries, padded_dim, d_centroids,
                                       num_centroids, padded_dim, d_q_norms, d_c_norms);
    add_norms_kernel<<<(num_queries * num_centroids + 255) / 256, 256, 0, stream>>>(
        d_centroid_dists, d_q_norms, d_c_norms, num_queries, num_centroids);

    TOCK("0~2. Initial Matrix Math");

    TICK();

    // Keep probe selection on the main stream so event_probe_done also orders
    // its D2H transfer before CPU refinement reads h_probe_vals.
    raft::device_resources handle{rmm::cuda_stream_view{stream}};
    raft::matrix::select_k<float, int>(
        handle,
        raft::make_device_matrix_view<const float, int64_t>(d_centroid_dists, (int)num_queries,
                                                            (int)num_centroids),
        std::nullopt,
        raft::make_device_matrix_view<float, int64_t>(d_probe_vals, (int)num_queries, (int)nprobe),
        raft::make_device_matrix_view<int, int64_t>(d_probe_idx, (int)num_queries, (int)nprobe),
        /*select_min=*/true, /*sorted=*/true);

    CUDA_CHECK(cudaEventRecord(event_probe_done, stream));
    CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, event_probe_done, 0));
    CUDA_CHECK(cudaMemcpyAsync(h_probe_vals, d_probe_vals, num_queries * nprobe * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_d2h));

    TOCK("3. Select Top-nprobe Clusters");

    TICK();

    int total_probes = num_queries * nprobe;
    flatten_pairs_kernel<<<(total_probes + 255) / 256, 256, 0, stream>>>(d_probe_idx, num_queries,
                                                                         nprobe, d_sorted_pairs);
    thrust::sort(thrust::cuda::par.on(stream), d_sorted_pairs, d_sorted_pairs + total_probes,
                 PairSortCompare(vanguard_size_));

    dim3 grid_lut(1, num_queries);
    precompute_lut_fp16_kernel<<<grid_lut, 256, 0, stream>>>(d_queries, ws_d_global_luts,
                                                             padded_dim);

    CUDA_CHECK(cudaStreamWaitEvent(stream, event_factors_done, 0));

    TOCK("4. Flatten Pairs");

    TICK();
    ComputeBitwiseKernelParams bitwiseParams;
    bitwiseParams.d_sorted_pairs = d_sorted_pairs;
    bitwiseParams.d_cluster_meta = d_cluster_meta;
    bitwiseParams.d_query = d_queries;
    bitwiseParams.d_G_k1xSumq = d_G_k1xSumq;
    bitwiseParams.d_centroid_distances = d_centroid_dists;
    bitwiseParams.d_short_data = d_short_data;
    bitwiseParams.d_short_factors = d_short_factors;
    bitwiseParams.d_q_bits = ws_d_q_bits;
    bitwiseParams.d_q_scale = ws_d_q_scale;
    bitwiseParams.d_q_min = ws_d_q_min;
    bitwiseParams.d_global_luts = ws_d_global_luts;
    bitwiseParams.num_queries = num_queries;
    bitwiseParams.num_centroids = num_centroids;
    bitwiseParams.D = padded_dim;
    bitwiseParams.max_candidates_per_pair = this->max_cluster_size_;
    bitwiseParams.num_pairs = total_probes;
    bitwiseParams.topk = topk;

    if (gpu_filter_timing_enabled_) {
        CUDA_CHECK(cudaEventRecord(filter_start_event, stream));
    }

    const int num_chunks = padded_dim / 32;

    if (ip_kernel_type_ == IPKernelType::LUT_4BIT) {

        const int requested_vanguard_depth =
            static_cast<int>((nprobe < vanguard_size_) ? nprobe : vanguard_size_);
        const int vanguard_keep = ((int)topk > 20) ? (int)topk : 20;
        const bool vanguard_supported = vanguard_keep <= VANGUARD_MAX_KEEP;
        const int vanguard_depth = vanguard_supported ? requested_vanguard_depth : 0;
        const int num_vanguard = num_queries * vanguard_depth;
        const int num_far = total_probes - num_vanguard;

        if (num_vanguard > 0) {
            bitwiseParams.d_sorted_pairs = d_sorted_pairs;
            bitwiseParams.num_pairs = num_vanguard;
            const size_t smem_vanguard = vanguard_dynamic_smem_bytes(padded_dim);
            CUDA_CHECK(cudaFuncSetAttribute(computeInnerProducts1BitLutProducerVanguard,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            (int)smem_vanguard));
            computeInnerProducts1BitLutProducerVanguard<<<640, VANGUARD_BLOCK_THREADS,
                                                          smem_vanguard, stream>>>(
                bitwiseParams, d_global_buffer, d_vram_write_count, d_dynamic_thresholds,
                d_global_pair_counter, ws_d_vanguard_seed_indices, ws_d_vanguard_seed_counts);
            CUDA_CHECK(cudaGetLastError());
        } else if (requested_vanguard_depth > 0) {
            std::cerr << "[warning] topk=" << topk << " exceeds Vanguard streaming limit "
                      << VANGUARD_MAX_KEEP << "; all pairs will use the QBatch kernel.\n";
        }

        // Continue ordinary filtering while the CPU refines vanguard seeds.
        CUDA_CHECK(cudaMemsetAsync(d_global_pair_counter, 0, sizeof(int), stream));

        if (num_far > 0) {
            bitwiseParams.d_sorted_pairs = d_sorted_pairs + num_vanguard;
            bitwiseParams.num_pairs = num_far;

            int max_persistent_blocks = 640;

            auto launch_qbatch = [&](auto Q_type) {
                constexpr int Q = decltype(Q_type)::value;
                size_t shared_mem_size = (size_t)Q * (padded_dim / 4) * 16 * sizeof(half);

                CUDA_CHECK(cudaFuncSetAttribute(
                    computeInnerProducts1BitLutProducerPersistentQBatch<Q>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_mem_size));
                computeInnerProducts1BitLutProducerPersistentQBatch<Q>
                    <<<max_persistent_blocks, 256, shared_mem_size, stream>>>(
                        bitwiseParams, d_global_buffer, d_vram_write_count, d_dynamic_thresholds,
                        d_global_pair_counter);
                CUDA_CHECK(cudaGetLastError());
            };

            if (!cross_query_reuse_enabled_) {

                launch_qbatch(std::integral_constant<int, 1>{});
            } else {
                switch (batch_q) {
                case 1:
                    launch_qbatch(std::integral_constant<int, 1>{});
                    break;
                case 2:
                    launch_qbatch(std::integral_constant<int, 2>{});
                    break;
                case 3:
                    launch_qbatch(std::integral_constant<int, 3>{});
                    break;
                case 4:
                    launch_qbatch(std::integral_constant<int, 4>{});
                    break;
                case 5:
                    launch_qbatch(std::integral_constant<int, 5>{});
                    break;
                case 6:
                    launch_qbatch(std::integral_constant<int, 6>{});
                    break;
                default:
                    std::cerr << "[error] unsupported batch_q=" << batch_q
                              << " (allowed: 1,2,3,4,5,6)\n";
                    std::exit(EXIT_FAILURE);
                }
            }
        }

        // Recover non-seed vectors after the ordinary scan, using the thresholds
        // tightened by CPU refinement without a GPU wait for those thresholds.
        if (num_vanguard > 0) {
            CUDA_CHECK(cudaMemsetAsync(d_global_pair_counter, 0, sizeof(int), stream));
            bitwiseParams.d_sorted_pairs = d_sorted_pairs;
            bitwiseParams.num_pairs = num_vanguard;
            const size_t smem_recovery = vanguard_recovery_dynamic_smem_bytes(padded_dim);
            CUDA_CHECK(cudaFuncSetAttribute(computeInnerProducts1BitLutProducerVanguardRecovery,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            (int)smem_recovery));
            computeInnerProducts1BitLutProducerVanguardRecovery<<<640, VANGUARD_BLOCK_THREADS,
                                                                  smem_recovery, stream>>>(
                bitwiseParams, d_global_buffer, d_vram_write_count, d_dynamic_thresholds,
                d_global_pair_counter, ws_d_vanguard_seed_indices, ws_d_vanguard_seed_counts);
            CUDA_CHECK(cudaGetLastError());
        }

    } else {
        size_t shared_mem_size = (size_t)(RABITQ_BQ * num_chunks) * sizeof(uint32_t) +
                                 (size_t)(2 * this->max_cluster_size_) * sizeof(float) +
                                 (size_t)(1 * this->max_cluster_size_) * sizeof(int);

        int max_persistent_blocks = 480;

        computeInnerProducts1BitBitwiseProducerPersistent<<<max_persistent_blocks, 256,
                                                            shared_mem_size, stream>>>(
            bitwiseParams, d_global_buffer, d_vram_write_count, d_dynamic_thresholds,
            d_global_pair_counter);
    }

    if (gpu_filter_timing_enabled_) {
        CUDA_CHECK(cudaEventRecord(filter_stop_event, stream));
    }

    coraanns::queue::device::finish_queue<<<1, 256, 0, stream>>>(d_global_buffer,
                                                                 d_vram_write_count);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamSynchronize(stream_d2h));

    this->cpu_exact_rerank(h_queries, num_queries, topk, nprobe, h_final_dists, h_final_pids);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    last_spilled_count_ = h_queue_progress->spilled;
    if (h_queue_progress->overflow) {
        std::memset(h_global_buffer->items, 0xFF, queue_capacity_ * sizeof(GlobalTaskItem));
        throw std::runtime_error("HBM queue spill capacity exhausted; increase --queue-capacity or "
                                 "--queue-spill-capacity, or reduce --query-chunk.");
    }
    if (gpu_filter_timing_enabled_) {
        CUDA_CHECK(
            cudaEventElapsedTime(&last_gpu_filter_ms_, filter_start_event, filter_stop_event));
    }
    if (ip_kernel_type_ == IPKernelType::LUT_4BIT) {
        TOCK("7~9. Single-Stage Global Task Pool Pipeline (LUT_4BIT)");
    } else if (ip_kernel_type_ == IPKernelType::LUT_4BIT_Q2) {
        TOCK("7~9. Single-Stage Global Task Pool Pipeline (LUT_4BIT_Q2)");
    } else {
        TOCK("7~9. Single-Stage Global Task Pool Pipeline (BITWISE_DECOMP)");
    }
    if (profiling_enabled_) {
        std::cout << "------------------------------------------\n";
    }

#undef TICK
#undef TOCK
}
