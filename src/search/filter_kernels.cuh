#pragma once

#include "kernel_types.cuh"
#include "ring_queue.cuh"
#include <cub/cub.cuh>
#include <cfloat>

namespace coraanns::detail {

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

namespace {
#ifndef CORAANNS_VANGUARD_CAPACITY
#define CORAANNS_VANGUARD_CAPACITY 512
#endif
static_assert(CORAANNS_VANGUARD_CAPACITY == 512 || CORAANNS_VANGUARD_CAPACITY == 1024,
              "Vanguard capacity must be 512 or 1024");
constexpr int VANGUARD_BLOCK_THREADS = 256;
constexpr int VANGUARD_ITEMS_PER_THREAD = 2 * CORAANNS_VANGUARD_CAPACITY / VANGUARD_BLOCK_THREADS;
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

#define CHUNK_MAJOR_MAX_VEC_PER_THREAD 6

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

} // namespace coraanns::detail
