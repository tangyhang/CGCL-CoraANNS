#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
// 引入 CUB (只保留基础的)
#include <cub/cub.cuh>

// 引入 Thrust 迭代器
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cfloat> // 提供 FLT_MAX

#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/matrix/select_k.cuh>
#include <rmm/device_uvector.hpp>
#include <optional>

// 引入头文件
#include "coraanns/baselines/gpu.cuh"

// 辅助函数：带有严格边界保护的 extract_code
__device__ inline uint32_t extract_code(const uint8_t* ptr, uint32_t dim, uint32_t ex_bits, uint32_t max_bytes) {
    uint32_t bit_idx = dim * ex_bits;
    uint32_t byte_idx = bit_idx / 8;
    uint32_t bit_offset = bit_idx % 8;
    
    uint32_t val = ptr[byte_idx];
    // 防止跨字节读取时越出 long_code 的最大长度
    if (byte_idx + 1 < max_bytes) {
        val |= (ptr[byte_idx + 1] << 8);
    }
    return (val >> bit_offset) & ((1 << ex_bits) - 1);
}

// ==========================================
// 辅助宏：用于检查 CUDA 函数调用的返回值
// ==========================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__device__ inline const uint8_t* vector_long_code(
    const ComputeBitwiseKernelParams& params, size_t vector, size_t stride) {
    const uint8_t* codes = vector < params.cached_vectors
        ? params.d_cached_long_code : params.d_long_code;
    return codes + vector * stride;
}

__device__ inline float2 vector_ex_factors(
    const ComputeBitwiseKernelParams& params, size_t vector) {
    const float* factors = vector < params.cached_vectors
        ? params.d_cached_ex_factor : params.d_ex_factor;
    return reinterpret_cast<const float2*>(factors)[vector];
}

// ==========================================
// Kernel: 计算范数及组合距离
// ==========================================
__global__ void row_norms_fused_kernel(const float* __restrict__ A, int A_rows, int A_cols,
                                       const float* __restrict__ B, int B_rows, int B_cols,
                                       float* __restrict__ A_norms, float* __restrict__ B_norms) {
  extern __shared__ float sdata[];
  const int tid     = threadIdx.x;
  const int lane    = tid & 31;
  const int warp_id = tid >> 5;
  const int nWarps  = (blockDim.x + 31) / 32;
  const int total_rows = A_rows + B_rows;

  for (int g = blockIdx.x; g < total_rows; g += gridDim.x) {
    const float* row_ptr; int cols; float* out_ptr;
    if (g < A_rows) {
      row_ptr = A + (size_t)g * A_cols; cols = A_cols; out_ptr = A_norms + g;
    } else {
      const int br = g - A_rows;
      row_ptr = B + (size_t)br * B_cols; cols = B_cols; out_ptr = B_norms + br;
    }
    float sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
      float v = row_ptr[c]; sum += v * v;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_down_sync(0xffffffff, sum, off);
    if (lane == 0) sdata[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
      float warp_sum = (tid < nWarps) ? sdata[tid] : 0.0f;
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) warp_sum += __shfl_down_sync(0xffffffff, warp_sum, off);
      if (tid == 0) *out_ptr = warp_sum;
    }
    __syncthreads();
  }
}

__global__ void add_norms_kernel(float* distances, const float* query_norms, const float* centroid_norms, int Q, int K) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < Q * K) {
    int q = idx / K; int k = idx % K;
    distances[idx] = distances[idx] + query_norms[q] + centroid_norms[k];
  }
}

// ==========================================
// Kernel: 预计算 Query Factors (修复 Bug 3)
// ==========================================
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T>
__global__ void computeQueryFactorsWarpKernel(const T* d_query, float* d_G_k1xSumq, float* d_G_kbxSumq, size_t num_queries, size_t D, size_t ex_bits) {
  const int warp_id         = threadIdx.x / 32;
  const int lane_id         = threadIdx.x % 32;
  const int warps_per_block = blockDim.x / 32;
  const int query_idx       = blockIdx.x * warps_per_block + warp_id;
  
  if (query_idx >= num_queries) return;
  float c_1 = -static_cast<float>((1 << 1) - 1) / 2.0f;
  float c_b = -static_cast<float>((1 << (ex_bits + 1)) - 1) / 2.0f;
  const T* query = d_query + query_idx * D;
  T sum = 0;
  for (int i = lane_id; i < D; i += 32) { sum += query[i]; }
  sum = warpReduceSum(sum);
  if (lane_id == 0) {
    d_G_k1xSumq[query_idx] = sum * c_1;
    d_G_kbxSumq[query_idx] = sum * c_b;
  }
}

template <typename T>
void computeQueryFactors(const T* d_query, float* d_G_k1xSumq, float* d_G_kbxSumq, size_t num_queries, size_t D, size_t ex_bits, cudaStream_t stream) {
  const int threads_per_block = 256;
  const int warps_per_block   = threads_per_block / 32;
  const int blocks            = (num_queries + warps_per_block - 1) / warps_per_block;
  computeQueryFactorsWarpKernel<T><<<blocks, threads_per_block, 0, stream>>>(d_query, d_G_k1xSumq, d_G_kbxSumq, num_queries, D, ex_bits);
}

// ==========================================
// Kernel: 扁平化重排与 Top-K 聚合
// ==========================================
__global__ void flatten_pairs_kernel(const int* d_probe_idx, int num_queries, int nprobe, ClusterQueryPair* d_pairs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries * nprobe) {
        d_pairs[idx].cluster_idx = d_probe_idx[idx];
        d_pairs[idx].query_idx   = idx / nprobe;
    }
}

__global__ void select_topk_kernel(float* d_centroid_dists, int candidate_pool_size, int topk, float* d_final_vals, int* d_final_idx, unsigned long long* d_phase_cycles) {
    int query_id = blockIdx.x;
    int tid = threadIdx.x; 

    float* row_dists = d_centroid_dists + query_id * candidate_pool_size;
    float* out_vals  = d_final_vals + query_id * topk;
    int* out_idx     = d_final_idx + query_id * topk;

    __shared__ float s_vals[256];
    __shared__ int   s_idx[256];

    unsigned long long start_clock = clock64();

    for (int k = 0; k < topk; k++) {
        float local_min = FLT_MAX;
        int local_idx = -1;

        for (int i = tid; i < candidate_pool_size; i += blockDim.x) {
            float val = row_dists[i];
            if (val < local_min) {
                local_min = val;
                local_idx = i;
            }
        }
        s_vals[tid] = local_min; s_idx[tid]  = local_idx;
        __syncthreads(); 

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (s_vals[tid + stride] < s_vals[tid]) {
                    s_vals[tid] = s_vals[tid + stride];
                    s_idx[tid]  = s_idx[tid + stride];
                }
            }
            __syncthreads(); 
        }

        if (tid == 0) {
            out_vals[k] = s_vals[0];
            out_idx[k]  = s_idx[0];
            if (s_idx[0] != -1) { row_dists[s_idx[0]] = FLT_MAX; }
        }
        __syncthreads();
    }

    unsigned long long end_clock = clock64();
    if (tid == 0) atomicAdd(&d_phase_cycles[2], (unsigned long long)(end_clock - start_clock));
}

__global__ void select_topk_kernel(float* d_centroid_dists, int candidate_pool_size, int topk, float* d_final_vals, int* d_final_idx) {
    int query_id = blockIdx.x;
    int tid = threadIdx.x; 

    float* row_dists = d_centroid_dists + query_id * candidate_pool_size;
    float* out_vals  = d_final_vals + query_id * topk;
    int* out_idx     = d_final_idx + query_id * topk;

    __shared__ float s_vals[256];
    __shared__ int   s_idx[256];

    for (int k = 0; k < topk; k++) {
        float local_min = FLT_MAX;
        int local_idx = -1;

        for (int i = tid; i < candidate_pool_size; i += blockDim.x) {
            float val = row_dists[i];
            if (val < local_min) {
                local_min = val;
                local_idx = i;
            }
        }
        s_vals[tid] = local_min; s_idx[tid]  = local_idx;
        __syncthreads(); 

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (s_vals[tid + stride] < s_vals[tid]) {
                    s_vals[tid] = s_vals[tid + stride];
                    s_idx[tid]  = s_idx[tid + stride];
                }
            }
            __syncthreads(); 
        }

        if (tid == 0) {
            out_vals[k] = s_vals[0];
            out_idx[k]  = s_idx[0];
            if (s_idx[0] != -1) { row_dists[s_idx[0]] = FLT_MAX; }
        }
        __syncthreads();
    }
}

// 修复 Bug 2: 将局部 Index 映射回真实的全局 PID
__global__ void map_pid_kernel(int* final_idx, uint32_t* candidate_pids, uint32_t* final_pids, int total_elements, int candidate_per_query, int topk) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        int q_id = idx / topk;
        int local_idx = final_idx[idx];
        if (local_idx != -1) {
            final_pids[idx] = candidate_pids[q_id * candidate_per_query + local_idx];
        }
    }
}

// 从 cub 段排序后的候选池里，取每个 query 距离最小的前 topk 个 (dist, pid)。
// 候选池里 d_candidates_pids 已是真实 PID，故此核同时取代了 select_topk_kernel 和 map_pid_kernel。
// 填充项 (dist=FLT_MAX, pid=0xFFFFFFFF) 升序后已沉到段尾，前 topk 即为有效结果（不足 topk 时尾部为 0xFFFFFFFF 哨兵）。
__global__ void gather_topk_from_sorted_kernel(
    const float* __restrict__ sorted_dists, const uint32_t* __restrict__ sorted_pids,
    int pool_size, int topk, int num_queries,
    float* __restrict__ final_dists, uint32_t* __restrict__ final_pids)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_queries * topk;
    if (idx < total) {
        int q = idx / topk;
        int r = idx % topk;
        size_t src = (size_t)q * pool_size + r;
        final_dists[idx] = sorted_dists[src];
        final_pids[idx]  = sorted_pids[src];
    }
}

// ==========================================
// 类函数实现
// ==========================================
IVF_RaBitQ_Raw::IVF_RaBitQ_Raw() {
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);
}

IVF_RaBitQ_Raw::~IVF_RaBitQ_Raw() {
    // All searches synchronize before returning, but keep teardown ordering
    // explicit because mapped host memory must outlive every GPU access.
    CUDA_CHECK(cudaStreamSynchronize(stream));
    free_workspace();
    cublasDestroy(cublas_handle);
    if (d_cached_long_codes_) CUDA_CHECK(cudaFree(d_cached_long_codes_));
    if (d_cached_ex_factors_) CUDA_CHECK(cudaFree(d_cached_ex_factors_));
    if (d_centroids)     CUDA_CHECK(cudaFree(d_centroids));
    if (d_rotator)       CUDA_CHECK(cudaFree(d_rotator));
    if (d_short_data)    CUDA_CHECK(cudaFree(d_short_data));
    if (d_short_factors) CUDA_CHECK(cudaFree(d_short_factors));
    if (long_codes_are_host_mapped_) {
        CUDA_CHECK(cudaHostUnregister(
            const_cast<uint8_t*>(h_registered_long_codes_)));
    } else if (d_long_codes) {
        CUDA_CHECK(cudaFree(d_long_codes));
    }
    if (ex_factors_are_host_mapped_) {
        CUDA_CHECK(cudaHostUnregister(
            const_cast<float*>(h_registered_ex_factors_)));
    } else if (d_ex_factors) {
        CUDA_CHECK(cudaFree(d_ex_factors));
    }
    if (d_ids)           CUDA_CHECK(cudaFree(d_ids));
    if (d_cluster_meta)  CUDA_CHECK(cudaFree(d_cluster_meta));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

// 1. 简化的 Load 函数 (分配显存并拷贝重排好的数据)
void IVF_RaBitQ_Raw::load_from_raw_pointers(
        size_t num_vecs, size_t d, size_t p_d, size_t n_centroids, size_t ex_b,
        const float* h_centroids,
        const float* h_rotator,
        const uint32_t* h_short_data, 
        const float* h_short_factors,
        const uint8_t* h_long_codes,
        const float* h_ex_factors,
        const uint32_t* h_ids,
        const size_t* h_cluster_sizes,
        bool host_backed_ex_codes
) {
    this->num_vectors = num_vecs;
    this->dim = d;
    this->padded_dim = p_d;
    this->num_centroids = n_centroids;
    this->ex_bits = ex_b;

    // 构建 Cluster Meta
    std::vector<GPUClusterMeta> h_cluster_meta(num_centroids);
    size_t current_offset = 0;
    for (size_t i = 0; i < num_centroids; ++i) {
        h_cluster_meta[i] = GPUClusterMeta(h_cluster_sizes[i], current_offset);
        current_offset += h_cluster_sizes[i];
    }
    CUDA_CHECK(cudaMallocAsync(&d_cluster_meta, num_centroids * sizeof(GPUClusterMeta), stream));
    CUDA_CHECK(cudaMemcpyAsync(d_cluster_meta, h_cluster_meta.data(), num_centroids * sizeof(GPUClusterMeta), cudaMemcpyHostToDevice, stream));

    // 上传 Centroids
    size_t centroids_bytes = num_centroids * padded_dim * sizeof(float);
    CUDA_CHECK(cudaMallocAsync(&d_centroids, centroids_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_centroids, h_centroids, centroids_bytes, cudaMemcpyHostToDevice, stream));

    // 上传旋转矩阵 (D x PaddedD)
    size_t rotator_bytes = dim * padded_dim * sizeof(float);
    CUDA_CHECK(cudaMallocAsync(&d_rotator, rotator_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_rotator, h_rotator, rotator_bytes, cudaMemcpyHostToDevice, stream));

    // 上传我们自己重排好的 1-bit Codes 和 Factors
    size_t short_data_bytes = num_vectors * (padded_dim / 32) * sizeof(uint32_t);
    CUDA_CHECK(cudaMallocAsync(&d_short_data, short_data_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_short_data, h_short_data, short_data_bytes, cudaMemcpyHostToDevice, stream));

    size_t short_factors_bytes = num_vectors * 3 * sizeof(float);
    CUDA_CHECK(cudaMallocAsync(&d_short_factors, short_factors_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_short_factors, h_short_factors, short_factors_bytes, cudaMemcpyHostToDevice, stream));

    // [新增] 上传 Extended Data
    if (ex_b > 0) {
        size_t long_code_bytes = num_vectors * ((padded_dim * ex_b + 7) / 8);
        if (host_backed_ex_codes) {
            CUDA_CHECK(cudaHostRegister(
                const_cast<uint8_t*>(h_long_codes), long_code_bytes,
                cudaHostRegisterMapped));
            CUDA_CHECK(cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&d_long_codes),
                const_cast<uint8_t*>(h_long_codes), 0));
            long_codes_are_host_mapped_ = true;
            h_registered_long_codes_ = h_long_codes;
            std::cout << "Extended codes: host-mapped CPU DRAM (GPU direct access), "
                      << long_code_bytes << " bytes" << std::endl;
        } else {
            CUDA_CHECK(cudaMallocAsync(&d_long_codes, long_code_bytes, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                d_long_codes, h_long_codes, long_code_bytes,
                cudaMemcpyHostToDevice, stream));
        }

        size_t ex_factors_bytes = num_vectors * 2 * sizeof(float);
        if (host_backed_ex_codes) {
            CUDA_CHECK(cudaHostRegister(
                const_cast<float*>(h_ex_factors), ex_factors_bytes,
                cudaHostRegisterMapped));
            CUDA_CHECK(cudaHostGetDevicePointer(
                reinterpret_cast<void**>(&d_ex_factors),
                const_cast<float*>(h_ex_factors), 0));
            ex_factors_are_host_mapped_ = true;
            h_registered_ex_factors_ = h_ex_factors;
            std::cout << "Extended-code factors: host-mapped CPU DRAM "
                      << "(GPU direct access), " << ex_factors_bytes
                      << " bytes" << std::endl;
        } else {
            CUDA_CHECK(cudaMallocAsync(&d_ex_factors, ex_factors_bytes, stream));
            CUDA_CHECK(cudaMemcpyAsync(
                d_ex_factors, h_ex_factors, ex_factors_bytes,
                cudaMemcpyHostToDevice, stream));
        }
    }

    // 上传 PIDs
    size_t ids_bytes = num_vectors * sizeof(uint32_t);
    CUDA_CHECK(cudaMallocAsync(&d_ids, ids_bytes, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ids, h_ids, ids_bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::cout << "Engine loaded via Custom Manual Transpose successfully!" << std::endl;
}



void IVF_RaBitQ_Raw::configure_hbm_cache(
    size_t reserve_bytes, size_t max_cache_bytes) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (d_cached_long_codes_) CUDA_CHECK(cudaFree(d_cached_long_codes_));
    if (d_cached_ex_factors_) CUDA_CHECK(cudaFree(d_cached_ex_factors_));
    d_cached_long_codes_ = nullptr;
    d_cached_ex_factors_ = nullptr;
    cached_vectors_ = 0;
    if (ex_bits == 0 || !long_codes_are_host_mapped_) return;

    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const size_t stride = (padded_dim * ex_bits + 7) / 8;
    const size_t record_bytes = stride + 2 * sizeof(float);
    const size_t budget = std::min(
        free_bytes > reserve_bytes ? free_bytes - reserve_bytes : 0,
        max_cache_bytes);
    size_t count = std::min(num_vectors, budget / record_bytes);
    if (count < num_vectors) count -= count % 32;
    // Fragmentation can prevent the two allocations from fitting even when
    // their sum fits. Back off before any search, never inside timed rounds.
    while (count > 0) {
        cudaError_t error = cudaMalloc(&d_cached_long_codes_, count * stride);
        if (error == cudaSuccess) {
            error = cudaMalloc(&d_cached_ex_factors_, count * 2 * sizeof(float));
        }
        if (error == cudaSuccess) break;
        if (d_cached_long_codes_) CUDA_CHECK(cudaFree(d_cached_long_codes_));
        if (d_cached_ex_factors_) CUDA_CHECK(cudaFree(d_cached_ex_factors_));
        d_cached_long_codes_ = nullptr;
        d_cached_ex_factors_ = nullptr;
        if (error != cudaErrorMemoryAllocation) {
            throw std::runtime_error(std::string("HBM cache allocation failed: ") + cudaGetErrorString(error));
        }
        cudaGetLastError();
        const size_t decrement = std::max(size_t(32), (size_t(64) << 20) / record_bytes);
        count = count > decrement ? count - decrement : 0;
        count -= count % 32;
    }
    if (count > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_cached_long_codes_, h_registered_long_codes_,
                                  count * stride, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_cached_ex_factors_, h_registered_ex_factors_,
                                  count * 2 * sizeof(float), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    cached_vectors_ = count;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::cout << "[GPUCache] policy=flattened-prefix cached_vectors=" << count
              << " total_vectors=" << num_vectors << " cached_bytes=" << cached_bytes()
              << " reserve_bytes=" << reserve_bytes
              << " free_after_cache=" << free_bytes << " total_bytes=" << total_bytes << '\n';
}

void IVF_RaBitQ_Raw::prepare_workspace(size_t max_queries, size_t max_topk, size_t max_nprobe) {
    free_workspace(); // 如果已经分配过，先清空

    ws_max_queries = max_queries;
    ws_max_topk = max_topk;
    ws_max_nprobe = max_nprobe;

    size_t max_total_probes = max_queries * max_nprobe;
    // size_t max_total_candidates = max_total_probes * this->max_cluster_size_;
    size_t max_total_candidates = max_queries * max_nprobe * max_topk;
    // 在 prepare_workspace 中加入：
    CUDA_CHECK(cudaMallocAsync(&ws_d_threshold, max_queries * sizeof(float), stream));

    // 一次性分配所有可能会在 Search 中用到的临时显存
    CUDA_CHECK(cudaMallocAsync(&ws_d_queries, max_queries * padded_dim * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_raw_queries, max_queries * dim * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_centroid_dists, max_queries * num_centroids * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_q_norms, max_queries * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_c_norms, num_centroids * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_probe_vals, max_queries * max_nprobe * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_probe_idx, max_queries * max_nprobe * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_sorted_pairs, max_total_probes * sizeof(ClusterQueryPair), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_G_k1xSumq, max_queries * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_G_kbxSumq, max_queries * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_candidates_dists, max_total_candidates * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_candidates_pids, max_total_candidates * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_query_write_counters, max_queries * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_final_dev_dists, max_queries * max_topk * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_final_dev_pids, max_queries * max_topk * sizeof(uint32_t), stream));
    
    // raft::select_k reads ws_d_centroid_dists directly. A second distance
    // matrix and the legacy full-sort arrays are unused by the search path.
    // At 10K queries x 65,536 centroids they would waste 9.77 GiB of HBM,
    // crowding out RAFT's temporary radix-selection allocations at k > 256.
    
    size_t lut_elements_per_q = (padded_dim / 4) * 16;
    CUDA_CHECK(cudaMallocAsync(&ws_d_global_luts, max_queries * lut_elements_per_q * sizeof(half), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void IVF_RaBitQ_Raw::free_workspace() {
    if(ws_d_queries) cudaFree(ws_d_queries);
    if(ws_d_raw_queries) cudaFree(ws_d_raw_queries);
    if(ws_d_centroid_dists) cudaFree(ws_d_centroid_dists);
    if(ws_d_q_norms) cudaFree(ws_d_q_norms);
    if(ws_d_c_norms) cudaFree(ws_d_c_norms);
    if(ws_d_probe_vals) cudaFree(ws_d_probe_vals);
    if(ws_d_probe_idx) cudaFree(ws_d_probe_idx);
    if(ws_d_sorted_pairs) cudaFree(ws_d_sorted_pairs);
    if(ws_d_G_k1xSumq) cudaFree(ws_d_G_k1xSumq);
    if(ws_d_G_kbxSumq) cudaFree(ws_d_G_kbxSumq);
    if(ws_d_candidates_dists) cudaFree(ws_d_candidates_dists);
    if(ws_d_candidates_pids) cudaFree(ws_d_candidates_pids);
    if(ws_d_query_write_counters) cudaFree(ws_d_query_write_counters);
    if(ws_d_final_dev_dists) cudaFree(ws_d_final_dev_dists);
    if(ws_d_final_dev_pids) cudaFree(ws_d_final_dev_pids);

    if(ws_d_threshold) cudaFree(ws_d_threshold); ws_d_threshold = nullptr;

    if(ws_d_global_luts) cudaFree(ws_d_global_luts); ws_d_global_luts = nullptr;

    ws_d_queries = nullptr; ws_d_raw_queries = nullptr; ws_d_centroid_dists = nullptr;
    ws_d_q_norms = nullptr; ws_d_c_norms = nullptr; ws_d_probe_vals = nullptr;
    ws_d_probe_idx = nullptr; ws_d_sorted_pairs = nullptr; ws_d_G_k1xSumq = nullptr;
    ws_d_G_kbxSumq = nullptr; ws_d_candidates_dists = nullptr; ws_d_candidates_pids = nullptr;
    ws_d_query_write_counters = nullptr; ws_d_final_dev_dists = nullptr; ws_d_final_dev_pids = nullptr;
}

__global__ void computeInnerProductsWithBitwiseOpt(const ComputeBitwiseKernelParams params)
{
  const int block_id = blockIdx.x;
  if (block_id >= params.num_pairs) return;

  ClusterQueryPair pair = params.d_sorted_pairs[block_id];
  int cluster_idx       = pair.cluster_idx;
  int query_idx         = pair.query_idx;

  if (cluster_idx >= params.num_centroids || query_idx >= params.num_queries) return;

  size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
  size_t cluster_start_index    = params.d_cluster_meta[cluster_idx].start_index;

  // Shared memory layout
  extern __shared__ __align__(256) char shared_mem_raw_2[];

  const int tid         = threadIdx.x;
  const int num_threads = blockDim.x;

  // Allocate shared memory for candidates
  float* shared_query       = reinterpret_cast<float*>(shared_mem_raw_2);
  float* shared_ip2_results = shared_query + params.D;

  for (size_t i = tid; i < params.D; i += num_threads) {
    shared_query[i] = params.d_query[query_idx * params.D + i];
  }
  __syncthreads();

  // Step 1: Warp-level IP2 computation for better memory coalescing
  const int warp_id   = tid / WARP_SIZE;
  const int lane_id   = tid % WARP_SIZE;
  const int num_warps = num_threads / WARP_SIZE;

  // Calculate long code parameters
  const uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;

  // Each warp processes different candidates
  for (int cand_idx = warp_id; cand_idx < num_vectors_in_cluster; cand_idx += num_warps) {
    size_t global_vec_idx = cluster_start_index + cand_idx;

    // Pointer to this vector's long code
    const uint8_t* vec_long_code = vector_long_code(params, global_vec_idx, long_code_size);

    // Warp-level IP2 computation
    float ip2 = 0.0f;

    // 为了调试，我们开一个临时变量记录该线程处理的维度点乘和
    float debug_thread_ip2_sum = 0.0f;

    // Each thread in warp processes different dimensions
    for (uint32_t d = lane_id; d < params.D; d += WARP_SIZE) {
      uint32_t code_val = extract_code(vec_long_code, d, params.ex_bits, long_code_size);

      float ex_val      = (float)code_val;
      ip2 += shared_query[d] * ex_val;
    }

    // Warp-level reduction for ip2
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
    }

    // Lane 0 stores the result
    if (lane_id == 0) { 
      shared_ip2_results[cand_idx] = ip2; 
    }
  }

  __syncthreads();

  // Step 2: Load float query and compute exact IPs for candidates

  const size_t short_code_length = params.D / 32;
  float q_g_add   = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
  float q_kbxsumq = params.d_G_kbxSumq[query_idx];

  // Atomically get write position
  __shared__ int probe_slot;
  if (tid == 0) {
    probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], num_vectors_in_cluster);
  }
  __syncthreads();
  // Calculate output offset
  uint32_t output_offset = query_idx * params.max_candidates_per_query + probe_slot;

  for (size_t vec_base = 0; vec_base < num_vectors_in_cluster; vec_base += num_threads) {
    size_t vec_idx = vec_base + tid;
    if (vec_idx >= num_vectors_in_cluster) break;

    // Compute exact inner product with float query
    float exact_ip = 0.0f;

    // Process each uint32_t of the short code
    for (size_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
      // Access short code in transposed layout
      size_t short_code_offset =
        cluster_start_index * short_code_length + uint32_idx * num_vectors_in_cluster + vec_idx;
      uint32_t short_code_chunk = params.d_short_data[short_code_offset];

      // Process each bit in the uint32_t
      // Note: bit 31 is lowest dimension, bit 0 is highest
#pragma unroll 8
      for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
        size_t dim = uint32_idx * 32 + bit_idx;
        if (dim < params.D) {
          // Extract bit from MSB to LSB
          int bit_position = 31 - bit_idx;
          bool bit_value   = (short_code_chunk >> bit_position) & 0x1;

          // If bit is 1, add the query value
          if (bit_value) { 
            exact_ip += shared_query[dim]; 
          } 
        }
      }
    }


    // Get pre-computed values
    float ip2             = shared_ip2_results[vec_idx];
    size_t global_vec_idx = cluster_start_index + vec_idx;

    // vec load version
    float2 ex_factors = vector_ex_factors(params, global_vec_idx);
    float f_ex_add     = ex_factors.x;
    float f_ex_rescale = ex_factors.y;

    // Compute final distance using pre-computed ip2
    float ex_dist =
      f_ex_add + q_g_add +
      f_ex_rescale * (static_cast<float>(1 << params.ex_bits) * exact_ip + ip2 + q_kbxsumq);

    // Write to global memory
    params.d_candidates_dists[output_offset + vec_idx] = ex_dist;
    params.d_candidates_pids[output_offset + vec_idx]  = params.d_pids[global_vec_idx];
    
}
}

// 🌟 安全处理所有浮点数的原子求小操作 (保留)
__device__ __forceinline__ void atomicMinFloat(const float* addr, float value) {
    int* address_as_int = (int*)addr;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= value) break;
        old = atomicCAS(address_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

// __global__ void computeInnerProductsWithBitwiseOptExactPruning(const ComputeBitwiseKernelParams params) {
//     int block_id = blockIdx.x;
//     if (block_id >= params.num_pairs) return;

//     ClusterQueryPair pair = params.d_sorted_pairs[block_id];
//     int cluster_idx = pair.cluster_idx;
//     int query_idx = pair.query_idx;
//     size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
//     size_t cluster_start_index = params.d_cluster_meta[cluster_idx].start_index;

//     int tid = threadIdx.x;
//     int num_threads = blockDim.x;

//     // =========================================================
//     // 🌟 计时变量定义 (仅在 Block 0, Thread 0 记录)
//     // =========================================================
//     long long t_start = 0, t_phase1 = 0, t_phase2 = 0, t_phase3 = 0, t_phase4 = 0;

//     // --- 动态 Shared Memory 布局 ---
//     extern __shared__ char smem[];
//     float* s_query = (float*)smem;
//     float* s_exact_ip = s_query + params.D;
//     int* s_passed_c = (int*)(s_exact_ip + params.max_candidates_per_pair);
//     float* s_cand_ips = (float*)(s_passed_c + params.max_candidates_per_pair);
//     int* s_cand_idx = (int*)(s_cand_ips + params.max_candidates_per_pair);

//     // 🌟 LUT 内存映射：每个 4-bit 组 (Nibble) 有 16 种组合。共需要 (D / 4) * 16 个 float。
//     float* s_LUT = (float*)(s_cand_idx + params.max_candidates_per_pair);

//     // 加载 Query
//     for (int i = tid; i < params.D; i += num_threads) {
//         s_query[i] = params.d_query[query_idx * params.D + i];
//     }
//     __syncthreads();

//     // =========================================================
//     // 🌟 2. 预计算 4-bit 查表缓存 (Look-Up Table)
//     // 动用所有线程并发构建，耗时仅需几个时钟周期
//     // =========================================================
//     int num_nibbles = params.D / 4; 
//     for (int i = tid; i < num_nibbles * 16; i += num_threads) {
//         int nibble_idx = i / 16;   // 第几个 4-bit 组
//         int pattern = i % 16;      // 该组的 16 种组合模式 (0x0 到 0xF)
        
//         float sum = 0.0f;
//         // 匹配位图：8, 4, 2, 1 对应大端序的 4 个 bit
//         if (pattern & 0x8) sum += s_query[nibble_idx * 4 + 0];
//         if (pattern & 0x4) sum += s_query[nibble_idx * 4 + 1];
//         if (pattern & 0x2) sum += s_query[nibble_idx * 4 + 2];
//         if (pattern & 0x1) sum += s_query[nibble_idx * 4 + 3];
        
//         // 存入一维展平的 LUT 数组中
//         s_LUT[nibble_idx * 16 + pattern] = sum;
//     }
//     __syncthreads();

//     // 🌟 记录起点：屏蔽了加载 Query 的开销，直接测算核心逻辑
//     if (threadIdx.x == 0 && (block_id < 4 || block_id == params.num_pairs - 1 || block_id == params.num_pairs / 2 || block_id == params.num_pairs / 4 || block_id == (params.num_pairs * 3) / 4)) t_start = clock64();

//     __shared__ int num_passed;
//     if (tid == 0) num_passed = 0;
//     __syncthreads();

//     float q_g_add = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
//     float q_k1xsumq = params.d_G_k1xSumq[query_idx];
//     float q_kbxsumq = params.d_G_kbxSumq[query_idx];
//     float q_g_error = sqrtf(q_g_add);
//     float threshold = params.d_threshold[query_idx];

//     int actual_cands = min((int)num_vectors_in_cluster, params.max_candidates_per_pair);
//     size_t short_code_length = params.D / 32;

//     // -------------------------------------------------------------
//     // 阶段 1：无分支算 1-bit Exact IP + 精确初筛
//     // -------------------------------------------------------------
//     for (int c = tid; c < actual_cands; c += num_threads) {
//         size_t global_idx = cluster_start_index + c;
//         float exact_ip = 0.0f;

//         float exact_ip0 = 0.0f;
//         float exact_ip1 = 0.0f;
//         float exact_ip2 = 0.0f;
//         float exact_ip3 = 0.0f;

//         for (int w = 0; w < short_code_length; w++) {
//             uint32_t chunk = __ldg(&params.d_short_data[cluster_start_index * short_code_length + w * num_vectors_in_cluster + c]);
//             int base_nibble = w * 8; // 一个 32-bit chunk 包含 8 个 4-bit nibbles

//             // 🌟 终极降维：直接提取 4-bit 索引进行极速查表！
//             // 从最高位往最低位切分，对应 LUT 的预计算布局
//             exact_ip0 += s_LUT[(base_nibble + 0) * 16 + ((chunk >> 28) & 0xF)];
//             exact_ip1 += s_LUT[(base_nibble + 1) * 16 + ((chunk >> 24) & 0xF)];
//             exact_ip2 += s_LUT[(base_nibble + 2) * 16 + ((chunk >> 20) & 0xF)];
//             exact_ip3 += s_LUT[(base_nibble + 3) * 16 + ((chunk >> 16) & 0xF)];

//             exact_ip0 += s_LUT[(base_nibble + 4) * 16 + ((chunk >> 12) & 0xF)];
//             exact_ip1 += s_LUT[(base_nibble + 5) * 16 + ((chunk >>  8) & 0xF)];
//             exact_ip2 += s_LUT[(base_nibble + 6) * 16 + ((chunk >>  4) & 0xF)];
//             exact_ip3 += s_LUT[(base_nibble + 7) * 16 + ((chunk >>  0) & 0xF)];
//         }

//         exact_ip = exact_ip0 + exact_ip1 + exact_ip2 + exact_ip3;

//         s_exact_ip[c] = exact_ip; 

//         float3 factors = reinterpret_cast<const float3*>(params.d_short_factors)[global_idx];
        
//         float low_dist = factors.x + q_g_add + factors.y * (exact_ip + q_k1xsumq) - factors.z * q_g_error;
//         // if (tid == 0) {
//         //     printf("Block %d, Candidate %d: exact_ip=%.4f, low_dist=%.4f, threshold=%.4f\n", block_id, c, exact_ip, low_dist, threshold);
//         // }
//         if (low_dist < threshold) {
//             int slot = atomicAdd(&num_passed, 1);
//             if (slot < params.max_candidates_per_pair) {
//                 s_passed_c[slot] = c;
//             }
//         }
//     }
//     __syncthreads();

//     // 🌟 记录阶段 1 结束
//     if (threadIdx.x == 0 && (block_id < 4 || block_id == params.num_pairs - 1 || block_id == params.num_pairs / 2 || block_id == params.num_pairs / 4 || block_id == (params.num_pairs * 3) / 4)) t_phase1 = clock64();

//     // 全局写入偏移
//     __shared__ int probe_slot;
//     if (tid == 0) probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1);
//     __syncthreads();
//     if (probe_slot >= params.nprobe) return;
//     uint32_t out_offset = query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;

//     int passed = min(num_passed, params.max_candidates_per_pair);

//     // -------------------------------------------------------------
//     // 阶段 2：Warp 并行算长码 IP2 (仅针对通过初筛的少数精锐)
//     // -------------------------------------------------------------
//     int warp_id = tid / 32, lane_id = tid % 32, num_warps = num_threads / 32;
//     uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;

//     // 🌟 新增：用于记录【精确距离】算完后，真正及格的人数
//     __shared__ int s_refined_passed;
//     if (tid == 0) s_refined_passed = 0;
//     __syncthreads();

//     // 🌟 核心优化 1：脏读最新的全局阈值！(由其他 Block 的 CPU 消费者实时压低的防线)
//     float latest_threshold = params.d_threshold[query_idx];

//     if (passed > 0) {
//         for (int idx = warp_id; idx < passed; idx += num_warps) {
//             int c = s_passed_c[idx];
//             size_t global_idx = cluster_start_index + c;
            
//             float ip2 = 0.0f;
//             const uint8_t* vec_long_code = params.d_long_code + global_idx * long_code_size;
            
//             for (int d = lane_id; d < params.D; d += 32) {
//                 uint32_t code_val = extract_code(vec_long_code, d, params.ex_bits, long_code_size);
//                 ip2 += s_query[d] * (float)code_val;
//             }
            
//             #pragma unroll
//             for (int offset = 16; offset > 0; offset /= 2) ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
            
//             if (lane_id == 0) {
//                 float exact_ip = s_exact_ip[c];
//                 float2 ex_factors = reinterpret_cast<const float2*>(params.d_ex_factor)[global_idx];
//                 float ex_dist = ex_factors.x + q_g_add + ex_factors.y * (static_cast<float>(1 << params.ex_bits) * exact_ip + ip2 + q_kbxsumq);
                
//                 // 🌟 核心优化 2：二次剪枝 (Secondary Pruning)
//                 // 只有距离严格小于最新阈值的，才有资格进入候选名单参与后续排序！
//                 if (ex_dist < latest_threshold) {
//                     int pos = atomicAdd(&s_refined_passed, 1);
//                     if (pos < params.max_candidates_per_pair) {
//                         s_cand_ips[pos] = ex_dist;
//                         s_cand_idx[pos] = params.d_pids[global_idx];
//                     }
//                 }
//             }
//         }
//     }
//     __syncthreads();

//     // 🌟 记录阶段 2 结束
//     if (threadIdx.x == 0 && (block_id < 4 || block_id == params.num_pairs - 1 || block_id == params.num_pairs / 2 || block_id == params.num_pairs / 4 || block_id == (params.num_pairs * 3) / 4)) t_phase2 = clock64();


//     // -------------------------------------------------------------
//     // 阶段 3 & 4：自适应排序旁路 (Adaptive Sort Bypass)
//     // -------------------------------------------------------------
//     constexpr int BLOCK_THREADS = 256;
//     constexpr int ITEMS_PER_THREAD = 4; 
//     int max_sort_capacity = BLOCK_THREADS * ITEMS_PER_THREAD;

//     int refined_passed = min(s_refined_passed, params.max_candidates_per_pair);
//     refined_passed = min(refined_passed, max_sort_capacity); // 防止越界

//     // 🌟 核心优化 3：判定是否触发“排序旁路”
//     __shared__ bool s_needs_sort;
//     if (tid == 0) s_needs_sort = (refined_passed > params.topk);
//     __syncthreads();

//     if (s_needs_sort) {
//         // =============== 走重型 CUB 排序分支 ===============
//         typedef cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int> BlockRadixSort;
//         typename BlockRadixSort::TempStorage& temp_storage = *reinterpret_cast<typename BlockRadixSort::TempStorage*>(smem);

//         float thread_keys[ITEMS_PER_THREAD];
//         int   thread_vals[ITEMS_PER_THREAD];

//         for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
//             int idx = tid * ITEMS_PER_THREAD + i;
//             if (idx < refined_passed) {
//                 thread_keys[i] = s_cand_ips[idx];
//                 thread_vals[i] = s_cand_idx[idx];
//             } else {
//                 // 不足的部分用无穷大填充，直接排到最后
//                 thread_keys[i] = FLT_MAX;
//                 thread_vals[i] = -1;
//             }
//         }
//         __syncthreads(); 

//         BlockRadixSort(temp_storage).Sort(thread_keys, thread_vals);
        
//         if (threadIdx.x == 0 && (block_id < 4 || block_id == params.num_pairs - 1 || block_id == params.num_pairs / 2 || block_id == params.num_pairs / 4 || block_id == (params.num_pairs * 3) / 4)) t_phase3 = clock64();

//         // 截取 Top-K 写入全局内存
//         for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
//             int sorted_rank = tid * ITEMS_PER_THREAD + i;
//             if (sorted_rank < params.topk) {
//                 params.d_candidates_dists[out_offset + sorted_rank] = thread_keys[i];
//                 params.d_candidates_pids[out_offset + sorted_rank]  = thread_vals[i];
//             }
//         }

//         // 尝试用局部的第 K 个好成绩，反压全局动态阈值
//         if (tid * ITEMS_PER_THREAD <= (params.topk - 1) && (tid * ITEMS_PER_THREAD + ITEMS_PER_THREAD) > (params.topk - 1)) {
//             int local_offset = (params.topk - 1) % ITEMS_PER_THREAD;
//             float max_topk_dist = thread_keys[local_offset];
//             if (max_topk_dist < latest_threshold) {
//                 atomicMinFloat(params.d_threshold + query_idx, max_topk_dist);
//             }
//         }
//     } else {
//         // =============== 走极速排序旁路 (Bypass) 分支 ===============
//         if (threadIdx.x == 0 && (block_id < 4 || block_id == params.num_pairs - 1 || block_id == params.num_pairs / 2 || block_id == params.num_pairs / 4 || block_id == (params.num_pairs * 3) / 4)) t_phase3 = clock64();

//         // 及格的人数 <= Top-K，直接原封不动地全部搬入显存，省去成千上万的指令周期！
//         for (int i = tid; i < params.topk; i += num_threads) {
//             if (i < refined_passed) {
//                 params.d_candidates_dists[out_offset + i] = s_cand_ips[i];
//                 params.d_candidates_pids[out_offset + i]  = s_cand_idx[i];
//             } else {
//                 params.d_candidates_dists[out_offset + i] = FLT_MAX;
//                 params.d_candidates_pids[out_offset + i]  = 0xFFFFFFFF; // 填充无效 ID
//             }
//         }
//         // 注：由于这里产生的数据量连 Top-K 都凑不齐，无法代表前 K 个门槛，因此不更新全局阈值
//     }

//     // 🌟 记录阶段 4 结束并打印
//     if (threadIdx.x == 0 && (block_id < 4 || block_id == params.num_pairs - 1 || block_id == params.num_pairs / 2 || block_id == params.num_pairs / 4 || block_id == (params.num_pairs * 3) / 4)) {
//         t_phase4 = clock64();
//         float prune_rate = 0.0f;
//         if (actual_cands > 0) {
//             // 这里用 refined_passed 算真实剪枝率，你会看到极其恐怖的过滤效果！
//             prune_rate = (float)(actual_cands - refined_passed) / (float)actual_cands * 100.0f;
//         }
//         // printf("Block %d, Final Prune_rate=%.2f%%, Phase 1: %lld cycles, Phase 2: %lld cycles, Phase 3: %lld cycles, Phase 4: %lld cycles\n", 
//         //        block_id, prune_rate, t_phase1 - t_start, t_phase2 - t_phase1, t_phase3 - t_phase2, t_phase4 - t_phase3);
//     }
// }

// =============================================================================
// 复现 NVIDIA computeInnerProductsWithLUT16OptBlockSort 的共享内存复用方案:
//   first  区 : s_LUT (= max(lut_bytes, MC*sizeof(float)))
//               阶段1 当 1-bit LUT; 阶段2 复用为 s_ip2(MC float); 阶段3 复用为 cub 排序暂存
//   second 区 : s_cand_ips (MC float) + s_cand_idx (MC int)
//               阶段1 存通过初筛候选的 1-bit ip 与局部下标; 阶段2c 就地覆写为 ex_dist 与 pid
//   third  区 : s_query (D float)  仅阶段2加载
// 总量 = max(first+second+third, sizeof(cub TempStorage)), 与 NVIDIA 内核一致 (≈12D + 8*MC)
// =============================================================================
__global__ void computeInnerProductsWithBitwiseOptExactPruning(const ComputeBitwiseKernelParams params) {
    int block_id = blockIdx.x;
    if (block_id >= params.num_pairs) return;

    ClusterQueryPair pair = params.d_sorted_pairs[block_id];
    int cluster_idx = pair.cluster_idx;
    int query_idx = pair.query_idx;
    const uint32_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
    const uint32_t cluster_start_index = params.d_cluster_meta[cluster_idx].start_index;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    const bool collect_profile = params.d_phase_cycles != nullptr;
    long long c0 = collect_profile ? clock64() : 0;

    // long long t_start = 0, t_phase1 = 0, t_phase2 = 0, t_phase3 = 0, t_phase4 = 0;
    // const bool kSample = (block_id < 4 || block_id == params.num_pairs - 1 ||
    //                       block_id == params.num_pairs / 2 || block_id == params.num_pairs / 4 ||
    //                       block_id == (params.num_pairs * 3) / 4);

    // ---------------- 共享内存布局 (与 NVIDIA 一致, 三区复用) ----------------
    extern __shared__ __align__(256) char smem[];
    const uint32_t MC        = params.max_candidates_per_pair;
    const uint32_t   lut_bytes = (uint32_t)(params.D / 4) * 16 * sizeof(half);   // FP16 LUT = 8*D 字节
    // first 区起点: 既要能放下 LUT, 又要能在阶段2复用为 MC 个 float 的 s_ip2
    const uint32_t   first_bytes =
        (lut_bytes < (uint32_t)MC * sizeof(float)) ? ((const uint32_t)MC * sizeof(float)) : lut_bytes;

    half*  s_LUT      = reinterpret_cast<half*>(smem);                 // first
    float* s_ip2      = reinterpret_cast<float*>(smem);                // 复用 first (LUT 用完后)
    float* s_cand_ips = reinterpret_cast<float*>(smem + first_bytes);  // second-A
    int*   s_cand_idx = reinterpret_cast<int*>(s_cand_ips + MC);       // second-B
    float* s_query    = reinterpret_cast<float*>(s_cand_idx + MC);     // third

    // 1. Float4 向量化加载 FP16 LUT 到 first 区
    const int lut_elements = (params.D / 4) * 16;
    const half* q_global_lut = params.d_global_luts + (uint32_t)query_idx * lut_elements;
    const int num_vec_loads = lut_elements / 8;
    float4* s_LUT_vec = reinterpret_cast<float4*>(s_LUT);
    const float4* q_global_lut_vec = reinterpret_cast<const float4*>(q_global_lut);
    for (int i = tid; i < num_vec_loads; i += num_threads) {
        s_LUT_vec[i] = q_global_lut_vec[i];
    }
    __syncthreads();
    // if (tid == 0 && kSample) t_start = clock64();

    __shared__ int num_candidates;
    if (tid == 0) num_candidates = 0;
    __syncthreads();

    float q_g_add   = params.d_centroid_distances[(uint32_t)query_idx * params.num_centroids + cluster_idx];
    float q_k1xsumq = params.d_G_k1xSumq[query_idx];
    float q_kbxsumq = params.d_G_kbxSumq[query_idx];
    float q_g_error = sqrtf(q_g_add);
    float threshold = params.d_threshold[query_idx];

    int actual_cands = min((int)num_vectors_in_cluster, (int)MC);
    uint32_t short_code_length = params.D / 32;


    // ---------------- 阶段1: 1-bit LUT 内积 + low_dist 初筛 ----------------
    // 通过者把 (1-bit ip, 局部下标) 直接压入候选区; 不再保留全簇 s_exact_ip / s_passed_c
    for (int c = tid; c < actual_cands; c += num_threads) {
        uint32_t global_idx = cluster_start_index + c;
        float ip = 0.0f;
        for (uint32_t uint32_idx = 0; uint32_idx < short_code_length; uint32_idx++) {
            uint32_t short_code_offset =
                cluster_start_index * short_code_length + (uint32_t)uint32_idx * num_vectors_in_cluster + c;
            uint32_t short_code_chunk = params.d_short_data[short_code_offset];
            for (int chunk_in_uint32 = 0; chunk_in_uint32 < 8; chunk_in_uint32++) {
                int shift   = 28 - (chunk_in_uint32 * BITS_PER_CHUNK);
                int pattern = (short_code_chunk >> shift) & 0xF;
                uint32_t lut_chunk_idx = uint32_idx * 8 + chunk_in_uint32;
                uint32_t lut_offset    = lut_chunk_idx * LUT_SIZE + pattern;
                ip += __half2float(s_LUT[lut_offset]);
            }
        }
        float3 factors = reinterpret_cast<const float3*>(params.d_short_factors)[global_idx];
        float low_dist = factors.x + q_g_add + factors.y * (ip + q_k1xsumq) - factors.z * q_g_error;
        if (low_dist < threshold) {
            int slot = atomicAdd(&num_candidates, 1);
            if (slot < (int)MC) {
                s_cand_ips[slot] = ip;   // 先存 1-bit ip (阶段2精排要用)
                s_cand_idx[slot] = c;    // 局部向量下标
            }
        }
    }
    __syncthreads();
    // if (tid == 0 && kSample) t_phase1 = clock64();

    long long c1 = collect_profile ? clock64() : 0;   // ★ scan 结束
    if (collect_profile && tid == 0) {
        atomicAdd(&params.d_phase_cycles[0],
                  (unsigned long long)(c1 - c0));
    }

    // probe 计数 (与原实现一致: 超出 nprobe 直接退出, 省掉精排)
    __shared__ int probe_slot;
    if (tid == 0) probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1);
    __syncthreads();
    if (probe_slot >= params.nprobe) return;
    uint32_t out_offset = query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;

    int passed = min(num_candidates, (int)MC);

    // ★ 动机计数（仅统计真正进入精排的 block，即过了 probe_slot<nprobe 的）：
    //   [3] 通过 1-bit 初筛(low_dist<threshold)的候选数；[4] 实际做 ex-bit refine 的数量；
    //   [5] 1-bit 扫描总量(分母)。本 kernel 中所有过初筛者都会被 refine，故 [4]==[3]，refine 率≈1.0。
    if (collect_profile && tid == 0) {
        atomicAdd(&params.d_phase_cycles[3], (unsigned long long)num_candidates);
        atomicAdd(&params.d_phase_cycles[4], (unsigned long long)passed);
        atomicAdd(&params.d_phase_cycles[5], (unsigned long long)actual_cands);
    }

    int warp_id = tid / 32, lane_id = tid % 32, num_warps = num_threads / 32;
    uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;
    float latest_threshold = params.d_threshold[query_idx];

    long long c1b = collect_profile ? clock64() : 0;  // ★ rerank 起点
    if (passed > 0) {
        // 阶段2a: 加载 query 到 third 区
        for (int i = tid; i < params.D; i += num_threads) {
            s_query[i] = params.d_query[(uint32_t)query_idx * params.D + i];
        }
        __syncthreads();

        // 阶段2b: 每 warp 处理一个候选, 算 ip2, 结果写入 s_ip2 (复用 LUT 区)
        for (int idx = warp_id; idx < passed; idx += num_warps) {
            int c = s_cand_idx[idx];
            uint32_t global_idx = cluster_start_index + c;
            const uint8_t* vec_long_code = vector_long_code(params, global_idx, long_code_size);
            float ip2 = 0.0f;
            for (int d = lane_id; d < params.D; d += 32) {
                uint32_t code_val = extract_code(vec_long_code, d, params.ex_bits, long_code_size);
                ip2 += s_query[d] * (float)code_val;
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
            if (lane_id == 0) s_ip2[idx] = ip2;
        }
        __syncthreads();
        // if (tid == 0 && kSample) t_phase2 = clock64();

        // 阶段2c: 算 ex_dist + exact pruning (就地覆写候选区, 不用额外缓冲)
        //   过阈值 -> 存 ex_dist 与 pid; 未过 -> 置 FLT_MAX / -1, 由排序自然淘汰
        for (int idx = tid; idx < passed; idx += num_threads) {
            float ip  = s_cand_ips[idx];
            float ip2 = s_ip2[idx];
            int   c   = s_cand_idx[idx];
            uint32_t global_idx = cluster_start_index + c;
            float2 ex_factors = vector_ex_factors(params, global_idx);
            float ex_dist = ex_factors.x + q_g_add +
                            ex_factors.y * (static_cast<float>(1 << params.ex_bits) * ip + ip2 + q_kbxsumq);
            bool keep = (ex_dist < latest_threshold);
            s_cand_ips[idx] = keep ? ex_dist : FLT_MAX;
            s_cand_idx[idx] = keep ? (int)params.d_pids[global_idx] : -1;
        }
        __syncthreads();
    }
    long long c2 = collect_profile ? clock64() : 0;   // ★ rerank 结束
    if (collect_profile && tid == 0) {
        atomicAdd(&params.d_phase_cycles[1],
                  (unsigned long long)(c2 - c1b));
    }

    // ---------------- 阶段3: Block 内 top-k (cub), 暂存复用 first 区 ----------------
    constexpr int BLOCK_THREADS = 256;
    constexpr int ITEMS_PER_THREAD = 4;
    const int max_sort_capacity = BLOCK_THREADS * ITEMS_PER_THREAD;   // 1024 (cub 容量上限)
    int sort_count = min(passed, max_sort_capacity);

    __shared__ bool s_needs_sort;
    if (tid == 0) s_needs_sort = (sort_count > params.topk);
    __syncthreads();

    if (s_needs_sort) {
        typedef cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int> BlockRadixSort;
        // 排序暂存复用 first 区 (LUT/ip2 都已用完); 候选先读入寄存器再排序, 覆写安全
        typename BlockRadixSort::TempStorage& temp_storage =
            *reinterpret_cast<typename BlockRadixSort::TempStorage*>(smem);

        float thread_keys[ITEMS_PER_THREAD];
        int   thread_vals[ITEMS_PER_THREAD];
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int idx = tid * ITEMS_PER_THREAD + i;
            if (idx < sort_count) { thread_keys[i] = s_cand_ips[idx]; thread_vals[i] = s_cand_idx[idx]; }
            else                  { thread_keys[i] = FLT_MAX;         thread_vals[i] = -1; }
        }
        __syncthreads();

        BlockRadixSort(temp_storage).Sort(thread_keys, thread_vals);
        // if (tid == 0 && kSample) t_phase3 = clock64();

        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int sorted_rank = tid * ITEMS_PER_THREAD + i;
            if (sorted_rank < params.topk) {
                params.d_candidates_dists[out_offset + sorted_rank] = thread_keys[i];
                params.d_candidates_pids[out_offset + sorted_rank]  = thread_vals[i];
            }
        }
        if (tid * ITEMS_PER_THREAD <= (params.topk - 1) &&
            (tid * ITEMS_PER_THREAD + ITEMS_PER_THREAD) > (params.topk - 1)) {
            int local_offset = (params.topk - 1) % ITEMS_PER_THREAD;
            float max_topk_dist = thread_keys[local_offset];
            if (max_topk_dist < latest_threshold) atomicMinFloat(params.d_threshold + query_idx, max_topk_dist);
        }
    } else {
        // if (tid == 0 && kSample) t_phase3 = clock64();
        for (int i = tid; i < params.topk; i += num_threads) {
            if (i < sort_count && s_cand_ips[i] < FLT_MAX) {
                params.d_candidates_dists[out_offset + i] = s_cand_ips[i];
                params.d_candidates_pids[out_offset + i]  = (uint32_t)s_cand_idx[i];
            } else {
                params.d_candidates_dists[out_offset + i] = FLT_MAX;
                params.d_candidates_pids[out_offset + i]  = 0xFFFFFFFF;
            }
        }
    }
    long long c3 = collect_profile ? clock64() : 0;   // ★ sort 结束
    if (collect_profile && tid == 0) {
        atomicAdd(&params.d_phase_cycles[2],
                  (unsigned long long)(c3 - c2));
    }

    // if (tid == 0 && kSample) t_phase4 = clock64();
}

// LAION100M contains a very long cluster tail (the largest cluster has more
// than 90K vectors).  The original kernel sizes its candidate arrays from the
// largest cluster and therefore cannot launch within the GPU shared-memory
// limit.  This host-backed path scans a fixed-size tile at a time and merges
// each tile into a per-probe top-k.  GPU filtering and ex-code refinement are
// otherwise identical to the device-resident kernel above.
namespace {
constexpr int kFineTileSize = 512;
constexpr int kFineBlockThreads = 256;
constexpr int kFineItemsPerThread =
    RABITQ_GPU_FINE_CAPACITY <= 512 ? 4 : 8;
static_assert(kFineBlockThreads * kFineItemsPerThread >=
              RABITQ_GPU_FINE_CAPACITY + kFineTileSize,
              "Fine-search merge must hold the retained top-k and a full tile");
using TiledFineBlockSort =
    cub::BlockRadixSort<float, kFineBlockThreads,
                        kFineItemsPerThread, int>;
}  // namespace

__global__ void computeInnerProductsWithBitwiseOptExactPruningTiled(
    const ComputeBitwiseKernelParams params)
{
    const int block_id = blockIdx.x;
    if (block_id >= params.num_pairs) return;

    const ClusterQueryPair pair = params.d_sorted_pairs[block_id];
    const int cluster_idx = pair.cluster_idx;
    const int query_idx = pair.query_idx;
    const uint32_t cluster_size =
        params.d_cluster_meta[cluster_idx].num;
    const uint32_t cluster_start =
        params.d_cluster_meta[cluster_idx].start_index;
    const int tid = threadIdx.x;

    extern __shared__ __align__(256) char smem[];
    const size_t lut_elements = (size_t)(params.D / 4) * 16;
    half* s_lut = reinterpret_cast<half*>(smem);
    float* s_query = reinterpret_cast<float*>(s_lut + lut_elements);
    float* s_ip2 = s_query + params.D;
    float* s_candidate_dist = s_ip2 + kFineTileSize;
    int* s_candidate_id = reinterpret_cast<int*>(
        s_candidate_dist + kFineTileSize);
    float* s_best_dist = reinterpret_cast<float*>(
        s_candidate_id + kFineTileSize);
    int* s_best_id = reinterpret_cast<int*>(s_best_dist + params.topk);

    __shared__ TiledFineBlockSort::TempStorage s_sort_storage;
    __shared__ int s_probe_slot;
    __shared__ int s_candidate_count;
    __shared__ int s_keep_count;
    __shared__ int s_best_count;

    if (tid == 0) {
        s_probe_slot = atomicAdd(
            &params.d_query_write_counters[query_idx], 1);
        s_best_count = 0;
    }
    __syncthreads();
    if (s_probe_slot >= params.nprobe) return;

    const size_t output_offset =
        ((size_t)query_idx * params.nprobe + s_probe_slot) * params.topk;

    const half* global_lut =
        params.d_global_luts + (size_t)query_idx * lut_elements;
    for (size_t i = tid; i < lut_elements; i += blockDim.x) {
        s_lut[i] = global_lut[i];
    }
    for (int i = tid; i < params.D; i += blockDim.x) {
        s_query[i] = params.d_query[(size_t)query_idx * params.D + i];
    }
    __syncthreads();

    const bool collect_profile = params.d_phase_cycles != nullptr;
    unsigned long long scan_cycles = 0;
    unsigned long long refine_cycles = 0;
    unsigned long long sort_cycles = 0;
    unsigned long long filter_survivors = 0;
    unsigned long long refined_vectors = 0;

    const float q_g_add = params.d_centroid_distances[
        (size_t)query_idx * params.num_centroids + cluster_idx];
    const float q_k1xsumq = params.d_G_k1xSumq[query_idx];
    const float q_kbxsumq = params.d_G_kbxSumq[query_idx];
    const float q_g_error = sqrtf(q_g_add);
    const uint32_t short_words = params.D / 32;
    const uint32_t long_code_size =
        (params.D * params.ex_bits + 7) / 8;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    const int warps_per_block = blockDim.x / 32;

    for (uint32_t tile_begin = 0; tile_begin < cluster_size;
         tile_begin += kFineTileSize) {
        const int tile_count = min(
            kFineTileSize, (int)(cluster_size - tile_begin));
        if (tid == 0) {
            s_candidate_count = 0;
            s_keep_count = 0;
        }
        __syncthreads();

        const long long scan_start =
            collect_profile ? clock64() : 0;
        const float filter_threshold = params.d_threshold[query_idx];
        for (int local = tid; local < tile_count; local += blockDim.x) {
            const uint32_t cluster_local = tile_begin + local;
            const uint32_t global_idx = cluster_start + cluster_local;
            float ip = 0.0f;
            for (uint32_t word = 0; word < short_words; ++word) {
                const size_t code_offset =
                    (size_t)cluster_start * short_words +
                    (size_t)word * cluster_size + cluster_local;
                const uint32_t code = params.d_short_data[code_offset];
#pragma unroll
                for (int nibble = 0; nibble < 8; ++nibble) {
                    const int pattern =
                        (code >> (28 - nibble * BITS_PER_CHUNK)) & 0xF;
                    ip += __half2float(
                        s_lut[((size_t)word * 8 + nibble) * LUT_SIZE +
                              pattern]);
                }
            }

            const float3 factors =
                reinterpret_cast<const float3*>(
                    params.d_short_factors)[global_idx];
            const float low_dist =
                factors.x + q_g_add +
                factors.y * (ip + q_k1xsumq) -
                factors.z * q_g_error;
            if (low_dist < filter_threshold) {
                const int slot = atomicAdd(&s_candidate_count, 1);
                // There is at most one survivor per vector in this tile.
                if (slot < kFineTileSize) {
                    s_candidate_dist[slot] = ip;
                    s_candidate_id[slot] = cluster_local;
                }
            }
        }
        __syncthreads();
        const long long scan_end = collect_profile ? clock64() : 0;

        const int passed = min(s_candidate_count, kFineTileSize);
        if (collect_profile && tid == 0) {
            scan_cycles += (unsigned long long)(scan_end - scan_start);
            filter_survivors += (unsigned long long)s_candidate_count;
            refined_vectors += (unsigned long long)passed;
        }

        const long long refine_start =
            collect_profile ? clock64() : 0;
        for (int idx = warp_id; idx < passed; idx += warps_per_block) {
            const int cluster_local = s_candidate_id[idx];
            const uint32_t global_idx = cluster_start + cluster_local;
            const uint8_t* long_code = vector_long_code(params, global_idx, long_code_size);
            float ip2 = 0.0f;
            for (int d = lane_id; d < params.D; d += 32) {
                const uint32_t code = extract_code(
                    long_code, d, params.ex_bits, long_code_size);
                ip2 += s_query[d] * (float)code;
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
            }
            if (lane_id == 0) s_ip2[idx] = ip2;
        }
        __syncthreads();

        const float refine_threshold = params.d_threshold[query_idx];
        for (int idx = tid; idx < passed; idx += blockDim.x) {
            const float ip = s_candidate_dist[idx];
            const int cluster_local = s_candidate_id[idx];
            const uint32_t global_idx = cluster_start + cluster_local;
            const float2 ex_factors = vector_ex_factors(params, global_idx);
            const float ex_dist =
                ex_factors.x + q_g_add +
                ex_factors.y *
                    (static_cast<float>(1 << params.ex_bits) * ip +
                     s_ip2[idx] + q_kbxsumq);
            const bool keep = ex_dist < refine_threshold;
            s_candidate_dist[idx] = keep ? ex_dist : FLT_MAX;
            s_candidate_id[idx] =
                keep ? (int)params.d_pids[global_idx] : -1;
            if (keep) atomicAdd(&s_keep_count, 1);
        }
        __syncthreads();
        const long long refine_end = collect_profile ? clock64() : 0;
        if (collect_profile && tid == 0) {
            refine_cycles +=
                (unsigned long long)(refine_end - refine_start);
        }

        const long long sort_start = collect_profile ? clock64() : 0;
        const int previous_best = s_best_count;
        const int merge_count = previous_best + passed;
        float thread_keys[kFineItemsPerThread];
        int thread_values[kFineItemsPerThread];
#pragma unroll
        for (int item = 0; item < kFineItemsPerThread; ++item) {
            const int rank = tid * kFineItemsPerThread + item;
            if (rank < previous_best) {
                thread_keys[item] = s_best_dist[rank];
                thread_values[item] = s_best_id[rank];
            } else if (rank < merge_count) {
                const int candidate = rank - previous_best;
                thread_keys[item] = s_candidate_dist[candidate];
                thread_values[item] = s_candidate_id[candidate];
            } else {
                thread_keys[item] = FLT_MAX;
                thread_values[item] = -1;
            }
        }
        __syncthreads();
        TiledFineBlockSort(s_sort_storage).Sort(
            thread_keys, thread_values);

#pragma unroll
        for (int item = 0; item < kFineItemsPerThread; ++item) {
            const int rank = tid * kFineItemsPerThread + item;
            if (rank < params.topk) {
                s_best_dist[rank] = thread_keys[item];
                s_best_id[rank] = thread_values[item];
            }
        }
        if (tid == 0) {
            s_best_count = min(
                params.topk, previous_best + s_keep_count);
        }
        __syncthreads();

        if (tid == 0 && s_best_count == params.topk) {
            const float kth = s_best_dist[params.topk - 1];
            if (kth < FLT_MAX) {
                atomicMinFloat(params.d_threshold + query_idx, kth);
            }
        }
        __syncthreads();
        const long long sort_end = collect_profile ? clock64() : 0;
        if (collect_profile && tid == 0) {
            sort_cycles += (unsigned long long)(sort_end - sort_start);
        }
    }

    for (int rank = tid; rank < params.topk; rank += blockDim.x) {
        if (rank < s_best_count && s_best_dist[rank] < FLT_MAX) {
            params.d_candidates_dists[output_offset + rank] =
                s_best_dist[rank];
            params.d_candidates_pids[output_offset + rank] =
                (uint32_t)s_best_id[rank];
        } else {
            params.d_candidates_dists[output_offset + rank] = FLT_MAX;
            params.d_candidates_pids[output_offset + rank] = 0xFFFFFFFF;
        }
    }

    if (collect_profile && tid == 0) {
        atomicAdd(&params.d_phase_cycles[0], scan_cycles);
        atomicAdd(&params.d_phase_cycles[1], refine_cycles);
        atomicAdd(&params.d_phase_cycles[2], sort_cycles);
        atomicAdd(&params.d_phase_cycles[3], filter_survivors);
        atomicAdd(&params.d_phase_cycles[4], refined_vectors);
        atomicAdd(
            &params.d_phase_cycles[5],
            (unsigned long long)cluster_size);
    }
}

// -----------------------------------------------------------------------------
// Controlled microbenchmarks for experiment 3.
//
// A fused kernel cannot provide independent wall-clock times for phases that
// overlap across resident blocks. These kernels replay filtering and
// cluster-local top-k work over the same selected <query, cluster> pairs.
// They are causal controls, not additive pieces of production wall time.
// Refinement is intentionally not replayed: synthesizing a different ex-code
// access order does not measure the production refinement stage.
// -----------------------------------------------------------------------------

__global__ void prepareProfilePairScanOffsets(
    const ComputeBitwiseKernelParams params,
    unsigned long long* d_pair_scan_offsets)
{
    const int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id >= params.num_pairs) return;
    const int cluster_idx = params.d_sorted_pairs[block_id].cluster_idx;
    d_pair_scan_offsets[block_id] =
        params.d_cluster_meta[cluster_idx].num;
}

__device__ __forceinline__ uint32_t profile_target_count(
    int block_id,
    uint32_t cluster_size,
    const unsigned long long* d_pair_scan_offsets,
    unsigned long long total_refined,
    unsigned long long total_scanned)
{
    if (cluster_size == 0 || total_refined == 0 || total_scanned == 0) {
        return 0;
    }
    const unsigned long long begin = d_pair_scan_offsets[block_id];
    const unsigned long long end = begin + cluster_size;
    // For DEEP100M these products are safely below uint64_t's limit. Using
    // prefix differences makes the replay total exactly total_refined.
    const unsigned long long target_begin =
        (begin * total_refined) / total_scanned;
    const unsigned long long target_end =
        (end * total_refined) / total_scanned;
    return static_cast<uint32_t>(target_end - target_begin);
}

__device__ __forceinline__ uint32_t profile_candidate_index(
    uint32_t rank, uint32_t target, uint32_t cluster_size)
{
    if (target == 0) return 0;
    const unsigned long long numerator =
        (2ULL * rank + 1ULL) * cluster_size;
    return min(
        static_cast<uint32_t>(numerator / (2ULL * target)),
        cluster_size - 1);
}

__device__ __forceinline__ float profile_block_sum(float value)
{
    __shared__ float warp_sums[kFineBlockThreads / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    float total = threadIdx.x < kFineBlockThreads / 32
                      ? warp_sums[lane]
                      : 0.0f;
    if (warp == 0) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            total += __shfl_down_sync(0xFFFFFFFF, total, offset);
        }
    }
    return total;
}

__global__ void profileFilteringIsolated(
    const ComputeBitwiseKernelParams params,
    const unsigned long long* d_pair_scan_offsets,
    unsigned long long total_refined,
    unsigned long long total_scanned,
    float* d_pair_checksums)
{
    const int block_id = blockIdx.x;
    if (block_id >= params.num_pairs) return;
    const ClusterQueryPair pair = params.d_sorted_pairs[block_id];
    const int cluster_idx = pair.cluster_idx;
    const int query_idx = pair.query_idx;
    const uint32_t cluster_size =
        params.d_cluster_meta[cluster_idx].num;
    const uint32_t cluster_start =
        params.d_cluster_meta[cluster_idx].start_index;
    const int tid = threadIdx.x;
    if (cluster_size == 0) {
        if (tid == 0) d_pair_checksums[block_id] = 0.0f;
        return;
    }

    extern __shared__ __align__(256) char smem[];
    const size_t lut_elements = (size_t)(params.D / 4) * 16;
    half* s_lut = reinterpret_cast<half*>(smem);
    for (size_t i = tid; i < lut_elements; i += blockDim.x) {
        s_lut[i] = params.d_global_luts[
            (size_t)query_idx * lut_elements + i];
    }
    __syncthreads();

    const float q_g_add = params.d_centroid_distances[
        (size_t)query_idx * params.num_centroids + cluster_idx];
    const float q_k1xsumq = params.d_G_k1xSumq[query_idx];
    const float q_g_error = sqrtf(q_g_add);
    const uint32_t short_words = params.D / 32;
    const uint32_t target = profile_target_count(
        block_id, cluster_size, d_pair_scan_offsets,
        total_refined, total_scanned);
    const float threshold = params.d_threshold[query_idx];
    float checksum = 0.0f;
    __shared__ int s_survivors;

    for (uint32_t tile_begin = 0; tile_begin < cluster_size;
         tile_begin += kFineTileSize) {
        const int tile_count = min(
            kFineTileSize, (int)(cluster_size - tile_begin));
        if (tid == 0) s_survivors = 0;
        __syncthreads();
        for (int local = tid; local < tile_count; local += blockDim.x) {
            const uint32_t cluster_local = tile_begin + local;
            const uint32_t global_idx = cluster_start + cluster_local;
            float ip = 0.0f;
            for (uint32_t word = 0; word < short_words; ++word) {
                const size_t code_offset =
                    (size_t)cluster_start * short_words +
                    (size_t)word * cluster_size + cluster_local;
                const uint32_t code = params.d_short_data[code_offset];
#pragma unroll
                for (int nibble = 0; nibble < 8; ++nibble) {
                    const int pattern =
                        (code >> (28 - nibble * BITS_PER_CHUNK)) & 0xF;
                    ip += __half2float(
                        s_lut[((size_t)word * 8 + nibble) * LUT_SIZE +
                              pattern]);
                }
            }
            const float3 factors =
                reinterpret_cast<const float3*>(
                    params.d_short_factors)[global_idx];
            const float low_dist =
                factors.x + q_g_add +
                factors.y * (ip + q_k1xsumq) -
                factors.z * q_g_error;

            // Select the work-equivalent, proportionally assigned vectors.
            // Still evaluate the real threshold predicate so the bound
            // calculation and comparison cannot be optimized away.
            const uint32_t before = static_cast<uint32_t>(
                (unsigned long long)cluster_local * target / cluster_size);
            const uint32_t after = static_cast<uint32_t>(
                (unsigned long long)(cluster_local + 1) * target /
                cluster_size);
            if (after != before) {
                atomicAdd(&s_survivors, 1);
                checksum += (low_dist < threshold) ? low_dist : ip;
            }
        }
        __syncthreads();
        if (tid == 0) checksum += (float)s_survivors * 1.0e-7f;
        __syncthreads();
    }
    const float total = profile_block_sum(checksum);
    if (tid == 0) d_pair_checksums[block_id] = total;
}

__global__ void profileClusterTopkIsolated(
    const ComputeBitwiseKernelParams params,
    const unsigned long long* d_pair_scan_offsets,
    unsigned long long total_refined,
    unsigned long long total_scanned,
    float* d_pair_checksums)
{
    const int block_id = blockIdx.x;
    if (block_id >= params.num_pairs) return;
    const ClusterQueryPair pair = params.d_sorted_pairs[block_id];
    const int query_idx = pair.query_idx;
    const uint32_t cluster_size =
        params.d_cluster_meta[pair.cluster_idx].num;
    const int tid = threadIdx.x;
    if (cluster_size == 0) {
        if (tid == 0) d_pair_checksums[block_id] = FLT_MAX;
        return;
    }
    const uint32_t target = profile_target_count(
        block_id, cluster_size, d_pair_scan_offsets,
        total_refined, total_scanned);

    extern __shared__ __align__(256) char smem[];
    const size_t lut_elements = (size_t)(params.D / 4) * 16;
    half* s_lut = reinterpret_cast<half*>(smem);
    float* s_query = reinterpret_cast<float*>(s_lut + lut_elements);
    float* s_ip2 = s_query + params.D;
    float* s_candidate_dist = s_ip2 + kFineTileSize;
    int* s_candidate_id = reinterpret_cast<int*>(
        s_candidate_dist + kFineTileSize);
    float* s_best_dist = reinterpret_cast<float*>(
        s_candidate_id + kFineTileSize);
    int* s_best_id = reinterpret_cast<int*>(s_best_dist + params.topk);

    __shared__ TiledFineBlockSort::TempStorage s_sort_storage;
    __shared__ int s_best_count;
    if (tid == 0) s_best_count = 0;
    __syncthreads();

    for (uint32_t tile_begin = 0; tile_begin < cluster_size;
         tile_begin += kFineTileSize) {
        const uint32_t tile_end = min(
            cluster_size, tile_begin + (uint32_t)kFineTileSize);
        const uint32_t prefix_begin = static_cast<uint32_t>(
            (unsigned long long)tile_begin * target / cluster_size);
        const uint32_t prefix_end = static_cast<uint32_t>(
            (unsigned long long)tile_end * target / cluster_size);
        const int passed = (int)(prefix_end - prefix_begin);
        for (int idx = tid; idx < passed; idx += blockDim.x) {
            const uint32_t rank = prefix_begin + idx;
            const uint32_t local =
                profile_candidate_index(rank, target, cluster_size);
            const uint32_t hashed =
                local * 2654435761U + (uint32_t)query_idx;
            s_candidate_dist[idx] =
                (float)(hashed & 0x00FFFFFFU) / 16777216.0f;
            s_candidate_id[idx] = (int)local;
        }
        __syncthreads();

        const int previous_best = s_best_count;
        const int merge_count = previous_best + passed;
        float thread_keys[kFineItemsPerThread];
        int thread_values[kFineItemsPerThread];
#pragma unroll
        for (int item = 0; item < kFineItemsPerThread; ++item) {
            const int rank = tid * kFineItemsPerThread + item;
            if (rank < previous_best) {
                thread_keys[item] = s_best_dist[rank];
                thread_values[item] = s_best_id[rank];
            } else if (rank < merge_count) {
                const int candidate = rank - previous_best;
                thread_keys[item] = s_candidate_dist[candidate];
                thread_values[item] = s_candidate_id[candidate];
            } else {
                thread_keys[item] = FLT_MAX;
                thread_values[item] = -1;
            }
        }
        __syncthreads();
        TiledFineBlockSort(s_sort_storage).Sort(
            thread_keys, thread_values);
#pragma unroll
        for (int item = 0; item < kFineItemsPerThread; ++item) {
            const int rank = tid * kFineItemsPerThread + item;
            if (rank < params.topk) {
                s_best_dist[rank] = thread_keys[item];
                s_best_id[rank] = thread_values[item];
            }
        }
        if (tid == 0) {
            s_best_count = min(params.topk, previous_best + passed);
        }
        __syncthreads();
        if (tid == 0 && s_best_count == params.topk) {
            const float kth = s_best_dist[params.topk - 1];
            if (kth < FLT_MAX) {
                atomicMinFloat(params.d_threshold + query_idx, kth);
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        d_pair_checksums[block_id] =
            s_best_count > 0 ? s_best_dist[0] : FLT_MAX;
    }
}

template <typename Launch>
float measure_isolated_kernel_ms(
    cudaStream_t stream, size_t rounds, Launch&& launch)
{
    launch();
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> samples;
    samples.reserve(rounds);
    for (size_t round = 0; round < rounds; ++round) {
        CUDA_CHECK(cudaEventRecord(start, stream));
        launch();
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaEventRecord(stop, stream));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        samples.push_back(elapsed);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::sort(samples.begin(), samples.end());
    const size_t middle = samples.size() / 2;
    return samples.size() % 2 == 0
               ? 0.5f * (samples[middle - 1] + samples[middle])
               : samples[middle];
}
// __global__ void computeInnerProductsWithBitwiseOptExactPruning(const ComputeBitwiseKernelParams params) {
//     int block_id = blockIdx.x;
//     if (block_id >= params.num_pairs) return;

//     ClusterQueryPair pair = params.d_sorted_pairs[block_id];
//     int cluster_idx = pair.cluster_idx;
//     int query_idx = pair.query_idx;
//     size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
//     size_t cluster_start_index = params.d_cluster_meta[cluster_idx].start_index;

//     int tid = threadIdx.x;
//     int num_threads = blockDim.x;

//     // =========================================================
//     // 🌟 计时变量定义 (仅在 Block 0, Thread 0 记录)
//     // =========================================================
//     long long t_start = 0, t_phase1 = 0, t_phase2 = 0, t_phase3 = 0;

//     // --- 动态 Shared Memory 布局 ---
//     extern __shared__ char smem[];
//     float* s_query = (float*)smem;
//     float* s_exact_ip = s_query + params.D;
//     int* s_passed_c = (int*)(s_exact_ip + params.max_candidates_per_pair);
//     float* s_cand_ips = (float*)(s_passed_c + params.max_candidates_per_pair);
//     int* s_cand_idx = (int*)(s_cand_ips + params.max_candidates_per_pair);

//     // 🌟 LUT 内存映射：每个 4-bit 组 (Nibble) 有 16 种组合。共需要 (D / 4) * 16 个 float。
//     float* s_LUT = (float*)(s_cand_idx + params.max_candidates_per_pair);

//     // 加载 Query
//     for (int i = tid; i < params.D; i += num_threads) {
//         s_query[i] = params.d_query[query_idx * params.D + i];
//     }
//     __syncthreads();

//     // =========================================================
//     // 🌟 2. 预计算 4-bit 查表缓存 (Look-Up Table)
//     // 动用所有线程并发构建，耗时仅需几个时钟周期
//     // =========================================================
//     int num_nibbles = params.D / 4; 
//     for (int i = tid; i < num_nibbles * 16; i += num_threads) {
//         int nibble_idx = i / 16;   // 第几个 4-bit 组
//         int pattern = i % 16;      // 该组的 16 种组合模式 (0x0 到 0xF)
        
//         float sum = 0.0f;
//         // 匹配位图：8, 4, 2, 1 对应大端序的 4 个 bit
//         if (pattern & 0x8) sum += s_query[nibble_idx * 4 + 0];
//         if (pattern & 0x4) sum += s_query[nibble_idx * 4 + 1];
//         if (pattern & 0x2) sum += s_query[nibble_idx * 4 + 2];
//         if (pattern & 0x1) sum += s_query[nibble_idx * 4 + 3];
        
//         // 存入一维展平的 LUT 数组中
//         s_LUT[nibble_idx * 16 + pattern] = sum;
//     }
//     __syncthreads();

//     // 🌟 记录起点：屏蔽了加载 Query 的开销，直接测算核心逻辑
//     if (block_id < 4 || block_id == params.num_pairs - 1 || block_id == params.num_pairs / 2 || block_id == params.num_pairs / 4 || block_id == (params.num_pairs * 3) / 4) t_start = clock64();

//     __shared__ int num_passed;
//     if (tid == 0) num_passed = 0;
//     __syncthreads();

//     float q_g_add = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
//     float q_k1xsumq = params.d_G_k1xSumq[query_idx];
//     float q_kbxsumq = params.d_G_kbxSumq[query_idx];
//     float q_g_error = sqrtf(q_g_add);
//     float threshold = params.d_threshold[query_idx];

//     int actual_cands = min((int)num_vectors_in_cluster, params.max_candidates_per_pair);
//     size_t short_code_length = params.D / 32;

//     // -------------------------------------------------------------
//     // 阶段 1：无分支算 1-bit Exact IP + 精确初筛
//     // -------------------------------------------------------------
//     for (int c = tid; c < actual_cands; c += num_threads) {
//         size_t global_idx = cluster_start_index + c;
//         float exact_ip = 0.0f;

//         // // 【最快】无分支浮点相乘，暴力碾压
//         // for (int w = 0; w < short_code_length; w++) {
//         //     uint32_t chunk = params.d_short_data[cluster_start_index * short_code_length + w * num_vectors_in_cluster + c];
//         //     #pragma unroll 32
//         //     for (int b = 0; b < 32; b++) {
//         //         // 提取当前 bit 并直接转为浮点数 (0.0f 或 1.0f)
//         //         float mask = (float)((chunk >> (31 - b)) & 1U);
                
//         //         // 🌟 优化 3：此时编译器会自动将其优化为 GPU 最底层的单周期 FMA 指令！
//         //         // 因为所有同 Warp 线程访问同一个 s_query 地址，这里还会触发完美的 Shared Memory 广播
//         //         exact_ip += s_query[w * 32 + b] * mask; 
//         //     }
//         // }
//         float exact_ip0 = 0.0f;
//         float exact_ip1 = 0.0f;
//         float exact_ip2 = 0.0f;
//         float exact_ip3 = 0.0f;

//         // for (int w = 0; w < short_code_length; w++) {
//         //     uint32_t chunk = __ldg(&params.d_short_data[cluster_start_index * short_code_length + w * num_vectors_in_cluster + c]);
//         //     const float* sq = &s_query[w * 32]; // 提取首地址，加速内存寻址

//         //     #pragma unroll 8
//         //     for (int b = 0; b < 8; b++) {
//         //         // 🌟 优化 2：舍弃浮点转换与乘法！使用底层谓词加法 (@p add.f32)
//         //         // 此时编译器会并发派发这 4 条指令，极速收割！
//         //         if ((chunk >> (31 - b))      & 1U) exact_ip0 += sq[b];
//         //         if ((chunk >> (23 - b))      & 1U) exact_ip1 += sq[b + 8];
//         //         if ((chunk >> (15 - b))      & 1U) exact_ip2 += sq[b + 16];
//         //         if ((chunk >> (7  - b))      & 1U) exact_ip3 += sq[b + 24];
//         //     }
//         // }

//         for (int w = 0; w < short_code_length; w++) {
//             uint32_t chunk = __ldg(&params.d_short_data[cluster_start_index * short_code_length + w * num_vectors_in_cluster + c]);
//             int base_nibble = w * 8; // 一个 32-bit chunk 包含 8 个 4-bit nibbles

//             // 🌟 终极降维：直接提取 4-bit 索引进行极速查表！
//             // 从最高位往最低位切分，对应 LUT 的预计算布局
//             exact_ip0 += s_LUT[(base_nibble + 0) * 16 + ((chunk >> 28) & 0xF)];
//             exact_ip1 += s_LUT[(base_nibble + 1) * 16 + ((chunk >> 24) & 0xF)];
//             exact_ip2 += s_LUT[(base_nibble + 2) * 16 + ((chunk >> 20) & 0xF)];
//             exact_ip3 += s_LUT[(base_nibble + 3) * 16 + ((chunk >> 16) & 0xF)];

//             exact_ip0 += s_LUT[(base_nibble + 4) * 16 + ((chunk >> 12) & 0xF)];
//             exact_ip1 += s_LUT[(base_nibble + 5) * 16 + ((chunk >>  8) & 0xF)];
//             exact_ip2 += s_LUT[(base_nibble + 6) * 16 + ((chunk >>  4) & 0xF)];
//             exact_ip3 += s_LUT[(base_nibble + 7) * 16 + ((chunk >>  0) & 0xF)];
//         }

//         exact_ip = exact_ip0 + exact_ip1 + exact_ip2 + exact_ip3;

//         s_exact_ip[c] = exact_ip; 

//         float3 factors = reinterpret_cast<const float3*>(params.d_short_factors)[global_idx];
        
//         float low_dist = factors.x + q_g_add + factors.y * (exact_ip + q_k1xsumq) - factors.z * q_g_error;
//         // if (tid == 0) {
//         //     printf("Block %d, Candidate %d: exact_ip=%.4f, low_dist=%.4f, threshold=%.4f\n", block_id, c, exact_ip, low_dist, threshold);
//         // }
//         if (low_dist < threshold) {
//             int slot = atomicAdd(&num_passed, 1);
//             if (slot < params.max_candidates_per_pair) {
//                 s_passed_c[slot] = c;
//             }
//         }
//     }
//     __syncthreads();

//     // 🌟 记录阶段 1 结束
//     if (block_id < 4 || block_id == params.num_pairs - 1 || block_id == params.num_pairs / 2 || block_id == params.num_pairs / 4 || block_id == (params.num_pairs * 3) / 4) t_phase1 = clock64();

//     // 全局写入偏移
//     __shared__ int probe_slot;
//     if (tid == 0) probe_slot = atomicAdd(&params.d_query_write_counters[query_idx], 1);
//     __syncthreads();
//     if (probe_slot >= params.nprobe) return;
//     uint32_t out_offset = query_idx * (params.topk * params.nprobe) + probe_slot * params.topk;

//     int passed = min(num_passed, params.max_candidates_per_pair);

//     // -------------------------------------------------------------
//     // 阶段 2：Warp 并行算长码 IP2 (仅针对通过初筛的少数精锐)
//     // -------------------------------------------------------------
//     int warp_id = tid / 32, lane_id = tid % 32, num_warps = num_threads / 32;
//     uint32_t long_code_size = (params.D * params.ex_bits + 7) / 8;

//     if (passed > 0) {
//         for (int idx = warp_id; idx < passed; idx += num_warps) {
//             int c = s_passed_c[idx];
//             size_t global_idx = cluster_start_index + c;
            
//             float ip2 = 0.0f;
//             const uint8_t* vec_long_code = params.d_long_code + global_idx * long_code_size;
            
//             for (int d = lane_id; d < params.D; d += 32) {
//                 uint32_t code_val = extract_code(vec_long_code, d, params.ex_bits, long_code_size);
//                 ip2 += s_query[d] * (float)code_val;
//             }
            
//             #pragma unroll
//             for (int offset = 16; offset > 0; offset /= 2) ip2 += __shfl_down_sync(0xFFFFFFFF, ip2, offset);
            
//             if (lane_id == 0) {
//                 float exact_ip = s_exact_ip[c];
//                 float2 ex_factors = reinterpret_cast<const float2*>(params.d_ex_factor)[global_idx];
//                 float ex_dist = ex_factors.x + q_g_add + ex_factors.y * (static_cast<float>(1 << params.ex_bits) * exact_ip + ip2 + q_kbxsumq);
                
//                 s_cand_ips[idx] = ex_dist;
//                 s_cand_idx[idx] = params.d_pids[global_idx];
//             }
//         }
//     }
//     __syncthreads();

//     // 🌟 记录阶段 2 结束
//     if (block_id < 4 || block_id == params.num_pairs - 1 || block_id == params.num_pairs / 2 || block_id == params.num_pairs / 4 || block_id == (params.num_pairs * 3) / 4) t_phase2 = clock64();

//     // -------------------------------------------------------------
//     // 阶段 3 & 4：CUB BlockRadixSort 极速排序
//     // -------------------------------------------------------------
//     // 设定 Block 的线程数和每个线程处理的元素数
//     // 假设 blockDim.x 固定为 256，每个线程处理 4 个元素，可以支撑最多 1024 个 candidates
//     constexpr int BLOCK_THREADS = 256;
//     constexpr int ITEMS_PER_THREAD = 4; 
    
//     // 定义 CUB 的基数排序类型 (按 float 升序排列，带上 int 的 payload)
//     typedef cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int> BlockRadixSort;
    
//     // 我们直接复用已经不再需要的 shared memory (s_query) 来作为 CUB 的临时交换区，做到零额外内存开销！
//     typename BlockRadixSort::TempStorage& temp_storage = *reinterpret_cast<typename BlockRadixSort::TempStorage*>(smem);

//     // 线程私有寄存器数组
//     float thread_keys[ITEMS_PER_THREAD];
//     int   thread_vals[ITEMS_PER_THREAD];

//     // 1. 将数据从 Shared Memory 搬运到线程专属的寄存器中
//     for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
//         int idx = tid * ITEMS_PER_THREAD + i;
//         if (idx < passed) {
//             thread_keys[i] = s_cand_ips[idx];
//             thread_vals[i] = s_cand_idx[idx];
//         } else {
//             // 对不足的部分用无穷大填充，CUB 会自动把它们排到最后面
//             thread_keys[i] = FLT_MAX;
//             thread_vals[i] = -1;
//         }
//     }
    
//     // 必须同步，确保所有人已经把 shared memory 中的数据读走了，才能覆盖 temp_storage
//     __syncthreads(); 

//     // 2. 召唤 CUB 魔法：纯寄存器级极速排序 (几千 cycle 瞬间完成！)
//     BlockRadixSort(temp_storage).Sort(thread_keys, thread_vals);

//     // 3. 将排好序的前 Top-K 个元素写回 Global Memory
//     for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
//         int idx = tid * ITEMS_PER_THREAD + i;
//         if (idx < params.topk) {
//             params.d_candidates_dists[out_offset + idx] = thread_keys[i];
//             params.d_candidates_pids[out_offset + idx]  = thread_vals[i];
//         }
//     }

//     // 4. 更新全局动态阈值 (由于是有序的，第 topk-1 个元素就是门槛)
//     if (passed >= params.topk && tid == 0) {
//         // CUB 排序后，整个 Block 的数据分布在所有线程的 registers 里
//         // 只有特定的线程拿着第 (topk - 1) 个元素
//     }
//     // [注意]：由于 CUB 排序后数据分散在寄存器中，我们需要把第 topk-1 个数据塞回 shared memory 让 tid=0 来更新阈值
//     if (tid * ITEMS_PER_THREAD <= (params.topk - 1) && (tid * ITEMS_PER_THREAD + ITEMS_PER_THREAD) > (params.topk - 1)) {
//         int local_offset = (params.topk - 1) % ITEMS_PER_THREAD;
//         float max_topk_dist = thread_keys[local_offset];
//         if (max_topk_dist < threshold) {
//             atomicMinFloat(params.d_threshold + query_idx, max_topk_dist);
//         }
//     }

//     // 🌟 记录阶段 3 & 4 结束并打印
//     if (threadIdx.x == 0 && (block_id < 4 || block_id == params.num_pairs - 1 || block_id == params.num_pairs / 2 || block_id == params.num_pairs / 4 || block_id == (params.num_pairs * 3) / 4)) {
//         t_phase3 = clock64();
//         float prune_rate = 0.0f;
//         if (actual_cands > 0) {
//             prune_rate = (float)(actual_cands - passed) / (float)actual_cands * 100.0f;
//         }
//         printf("Block %d, Prune_rate=%.2f%%, Phase 1: %lld cycles, Phase 2: %lld cycles, Phase 3: %lld cycles\n", block_id, prune_rate, t_phase1 - t_start, t_phase2 - t_phase1, t_phase3 - t_phase2);
        
//         // printf("\n========== [Kernel Internal Profiler (Block 0)] ==========\n");
//         // printf("  ├─ 1. Phase 1 (1-bit IP + Prune) : %lld cycles\n", t_phase1 - t_start);
//         // printf("  ├─ 2. Phase 2 (Long Code IP2)    : %lld cycles\n", t_phase2 - t_phase1);
//         // printf("  └─ 3. Phase 3 & 4 (Sort & Atomic): %lld cycles\n", t_phase3 - t_phase2);
//         // printf("==========================================================\n");
//     }
// }

struct PoolOffsetOp {
    int pool_size;
    __host__ __device__ __forceinline__ int operator()(int idx) const { 
        return idx * pool_size; 
    }
};

__global__ void init_local_indices_kernel(int* indices, int pool_size, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) indices[idx] = idx % pool_size;
}

__global__ void gather_nprobe_kernel(int num_queries, int nprobe, int num_centroids,
        const float* d_sorted_dists, const int* d_sorted_indices, float* out_probe_vals, int* out_probe_idx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries * nprobe) {
        int src_idx = (idx / nprobe) * num_centroids + (idx % nprobe);
        out_probe_vals[idx] = d_sorted_dists[src_idx];
        out_probe_idx[idx]  = d_sorted_indices[src_idx];
    }
}

// =========================================================
// 🌟 提前全局计算所有 Query 的 FP16 LUT
// =========================================================
__global__ void precompute_lut_fp16_kernel(const float* d_query, half* d_lut, int D) {
    int q_idx = blockIdx.y; 
    int tid = threadIdx.x;
    int num_nibbles = D / 4;
    
    for (int i = tid; i < num_nibbles * 16; i += blockDim.x) {
        int nibble_idx = i / 16;
        int pattern = i % 16;
        
        float sum = 0.0f;
        if (pattern & 0x8) sum += d_query[q_idx * D + nibble_idx * 4 + 0];
        if (pattern & 0x4) sum += d_query[q_idx * D + nibble_idx * 4 + 1];
        if (pattern & 0x2) sum += d_query[q_idx * D + nibble_idx * 4 + 2];
        if (pattern & 0x1) sum += d_query[q_idx * D + nibble_idx * 4 + 3];
        
        d_lut[q_idx * (num_nibbles * 16) + i] = __float2half(sum);
    }
}

void IVF_RaBitQ_Raw::reset_buffer(const float* h_queries, size_t num_queries)
{
    // 1. GPU 异步 H2D 拷贝与 CPU 状态重置 (完全重叠)
    CUDA_CHECK(cudaMemcpyAsync(ws_d_raw_queries, h_queries, num_queries * dim * sizeof(float), cudaMemcpyHostToDevice, stream));
}

void IVF_RaBitQ_Raw::search(const float* h_queries, size_t num_queries,
                            size_t topk, size_t nprobe,
                            float* h_final_dists, uint32_t* h_final_pids,
                            int* h_selected_cluster_ids) {
    // -------------------------------------------------------------
    // 安全检查：确保当前查询的规模没有超出我们预分配的 Workspace 容量
    // -------------------------------------------------------------
    if (num_queries > ws_max_queries || topk > ws_max_topk || nprobe > ws_max_nprobe) {
        std::cerr << "[Error] Search parameters exceed workspace capacity!" << std::endl;
        return;
    }

    // 将预分配好的显存指针映射到局部变量，避免重写后续业务代码
    float* d_queries              = ws_d_queries;
    float* d_raw_queries          = ws_d_raw_queries;
    float* d_centroid_dists       = ws_d_centroid_dists;
    float* d_q_norms              = ws_d_q_norms;
    float* d_c_norms              = ws_d_c_norms;
    float* d_probe_vals           = ws_d_probe_vals;
    int* d_probe_idx            = ws_d_probe_idx;
    ClusterQueryPair* d_sorted_pairs = ws_d_sorted_pairs;
    float* d_G_k1xSumq            = ws_d_G_k1xSumq;
    float* d_G_kbxSumq            = ws_d_G_kbxSumq;
    float* d_candidates_dists     = ws_d_candidates_dists;
    uint32_t* d_candidates_pids   = ws_d_candidates_pids;
    int* d_query_write_counters   = ws_d_query_write_counters;
    float* d_final_dev_dists      = ws_d_final_dev_dists;
    uint32_t* d_final_dev_pids    = ws_d_final_dev_pids;

    // 🌟 raft handle：粗排簇选择(阶段3) 与 全局 top-k(阶段7.5) 共用，建一次即可
    raft::device_resources handle{rmm::cuda_stream_view{stream}};
    
    // =========================================================
    // 🌟 初始化 CUDA 事件，用于高精度测速
    // =========================================================
    cudaEvent_t start = nullptr, stop = nullptr;
    if (profiling_enabled_) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    float ms = 0.0f;

    // 定义测速宏，保持代码整洁
    #define TICK() do { \
        if (profiling_enabled_) CUDA_CHECK(cudaEventRecord(start, stream)); \
    } while (0)
    #define TOCK(step_name) do { \
        if (profiling_enabled_) { \
            CUDA_CHECK(cudaEventRecord(stop, stream)); \
            CUDA_CHECK(cudaEventSynchronize(stop)); \
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); \
            std::cout << "[Profile] " << step_name << " time: " << ms << " ms\n"; \
        } \
    } while (0)


    // =========================================================
    // 0. Query 数据拷贝 (H2D)
    // =========================================================
    // TICK();
    // size_t queries_bytes = num_queries * padded_dim * sizeof(float);
    // CUDA_CHECK(cudaMemcpyAsync(d_queries, h_queries, queries_bytes, cudaMemcpyHostToDevice, stream));
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // TOCK("0. Query H2D Transfer");

    // TICK();
    // size_t raw_bytes = num_queries * dim * sizeof(float);
    // CUDA_CHECK(cudaMemcpyAsync(d_raw_queries, h_queries, raw_bytes, cudaMemcpyHostToDevice, stream));
    // TOCK("0. Raw Query H2D Transfer");

    // 🌟 0.5 GPU 旋转：Q_rotated = Q_raw * Rotator
    // 对应 cuBLAS 列主序计算：(Q_rotated)^T = (Rotator)^T * (Q_raw)^T
    TICK();
    const float alpha_rot = 1.0f; const float beta_rot = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                padded_dim, num_queries, dim,
                &alpha_rot, d_rotator, padded_dim, 
                d_raw_queries, dim, 
                &beta_rot, d_queries, padded_dim);
    TOCK("0.5. Query Rotation (cuBLAS)");

    // =========================================================
    // 1. 粗量化检索 (CUBLAS GEMM)
    // 利用矩阵乘法快速计算 Query 与所有簇中心 (Centroids) 的内积
    // =========================================================
    TICK();
    const float alpha = -2.0f; const float beta = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                num_centroids, num_queries, padded_dim,
                &alpha, d_centroids, padded_dim, 
                d_queries, padded_dim, 
                &beta, d_centroid_dists, num_centroids);
    TOCK("1. Coarse Search (cuBLAS)");

    // =========================================================
    // 2. 补全 L2 距离范数
    // 加上 Query 自身范数和 Centroid 范数，构成完整的 L2 距离
    // =========================================================
    TICK();
    int norm_block_size = 256;
    size_t norm_shared_mem = ((norm_block_size + 31) / 32) * sizeof(float);
    row_norms_fused_kernel<<<num_centroids + num_queries, norm_block_size, norm_shared_mem, stream>>>(
        d_queries, num_queries, padded_dim, d_centroids, num_centroids, padded_dim, d_q_norms, d_c_norms);
    int add_threads = 256;
    int add_blocks  = (num_queries * num_centroids + add_threads - 1) / add_threads;
    add_norms_kernel<<<add_blocks, add_threads, 0, stream>>>(d_centroid_dists, d_q_norms, d_c_norms, num_queries, num_centroids);
    TOCK("2. L2 Norms Completion");

    // =========================================================
    // 3. Top-nprobe 簇选择
    // 为每个 Query 选出距离最近的 nprobe 个簇，准备进行细排
    // =========================================================
    TICK();
    // 🌟 与 NVIDIA PrepareClusterSearchInputs 完全对齐：
    //    用 raft select_k 选每个 query 最近的 nprobe 个簇，替代朴素选择(<50)/全量段排序(>=50)。
    //    select_min=true 取最小 ‖q-c‖²；sorted=true 让簇按距离升序（vanguard 需最近簇优先）。
    //    输入 d_centroid_dists (num_queries × num_centroids) 只读不被修改，
    //    仍完整保留给后续 fine kernel 取 q_g_add 用。
    raft::matrix::select_k<float, int>(
        handle,
        raft::make_device_matrix_view<const float, int64_t>(
            d_centroid_dists, (int)num_queries, (int)num_centroids),
        std::nullopt,
        raft::make_device_matrix_view<float, int64_t>(
            d_probe_vals, (int)num_queries, (int)nprobe),
        raft::make_device_matrix_view<int,   int64_t>(
            d_probe_idx,  (int)num_queries, (int)nprobe),
        /*select_min=*/true,
        /*sorted=*/true);
    TOCK("3. Select Top-nprobe Clusters");

    // =========================================================
    // 4. Pair 扁平化与聚类排序
    // 将待查的簇转为 <Cluster, Query> 键值对，并按 Cluster 排序以增加访存局部性
    // =========================================================
    TICK();
    int total_probes = num_queries * nprobe;
    flatten_pairs_kernel<<<(total_probes + 255) / 256, 256, 0, stream>>>(d_probe_idx, num_queries, nprobe, d_sorted_pairs);
    thrust::sort(thrust::cuda::par.on(stream), d_sorted_pairs, d_sorted_pairs + total_probes, PairSortCompare());
    TOCK("4. Flatten & Sort Pairs");

    // =========================================================
    // 5. 预计算 Query 还原因子
    // 提前算出 RaBitQ 公式中与 Query 相关的常数项
    // =========================================================
    TICK();
    computeQueryFactors<float>(d_queries, d_G_k1xSumq, d_G_kbxSumq, num_queries, padded_dim, ex_bits, stream);
    TOCK("5. Precompute Query Factors");

    // -------------------------------------------------------------
    // 初始化动态阈值为无穷大 (让第一个 Block 畅通无阻，建立基准)
    // -------------------------------------------------------------
    TICK();
    thrust::fill(thrust::cuda::par.on(stream), thrust::device_pointer_cast(ws_d_threshold), thrust::device_pointer_cast(ws_d_threshold) + num_queries, FLT_MAX);
    CUDA_CHECK(cudaMemsetAsync(d_query_write_counters, 0, num_queries * sizeof(int), stream));
    
    // 🌟 在进入 Fine Search 之前，预计算所有的 FP16 LUT
    dim3 grid_lut(1, num_queries);
    precompute_lut_fp16_kernel<<<grid_lut, 256, 0, stream>>>(d_queries, ws_d_global_luts, padded_dim);
    TOCK("6. Candidate Threshold & Counters Init");

    // -------------------------------------------------------------
    // 7. 终极细粒度检索 (动态剪枝 + __clz 无分支 + BlockSort)
    // -------------------------------------------------------------
    TICK();
    int num_words = (padded_dim + 31) / 32;

    unsigned long long* d_phase_cycles = nullptr;
    if (profiling_enabled_) {
        CUDA_CHECK(cudaMalloc(&d_phase_cycles, 6 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(
            d_phase_cycles, 0, 6 * sizeof(unsigned long long)));
    }

    ComputeBitwiseKernelParams bitwiseParams;
    bitwiseParams.d_phase_cycles = d_phase_cycles;
    bitwiseParams.d_sorted_pairs = d_sorted_pairs;
    bitwiseParams.d_cluster_meta = d_cluster_meta;
    bitwiseParams.d_query = d_queries; 
    bitwiseParams.d_packed_queries = nullptr; // 无需量化
    bitwiseParams.d_widths = nullptr;         // 无需量化
    bitwiseParams.d_threshold = ws_d_threshold; // 🌟 传入剪枝阈值！
    bitwiseParams.d_G_k1xSumq = d_G_k1xSumq; 
    bitwiseParams.d_centroid_distances = d_centroid_dists;
    bitwiseParams.d_short_data = d_short_data;
    bitwiseParams.d_short_factors = d_short_factors;
    bitwiseParams.d_long_code = d_long_codes;         
    bitwiseParams.d_ex_factor = d_ex_factors;         
    bitwiseParams.d_cached_long_code = d_cached_long_codes_;
    bitwiseParams.d_cached_ex_factor = d_cached_ex_factors_;
    bitwiseParams.cached_vectors = cached_vectors_;
    bitwiseParams.d_G_kbxSumq = d_G_kbxSumq;
    bitwiseParams.d_global_luts = ws_d_global_luts; // 🌟 传入预计算的全局 LUT
    bitwiseParams.ex_bits = ex_bits;
    bitwiseParams.d_pids = d_ids;
    bitwiseParams.d_candidates_dists = d_candidates_dists;
    bitwiseParams.d_candidates_pids = d_candidates_pids;
    bitwiseParams.d_query_write_counters = d_query_write_counters;
    bitwiseParams.num_queries = num_queries;
    bitwiseParams.num_centroids = num_centroids;
    bitwiseParams.D = padded_dim;
    bitwiseParams.num_words = num_words;
    bitwiseParams.num_pairs = total_probes;
    bitwiseParams.max_candidates_per_pair = this->max_cluster_size_;
    bitwiseParams.topk = topk;
    bitwiseParams.nprobe = nprobe;

    // // 共享内存分配：确保足够容纳所有中间状态
    // size_t query_bytes = padded_dim * sizeof(float) * 5;
    // size_t cand_bytes = this->max_cluster_size_ * (3 * sizeof(float) + 2 * sizeof(int));
    // size_t reduce_bytes = 256 * (sizeof(float) + sizeof(int));
    // size_t shared_mem_size = query_bytes + cand_bytes + reduce_bytes;

    // computeInnerProductsWithBitwiseOptExactPruning<<<total_probes, 256, shared_mem_size, stream>>>(bitwiseParams);
    
    // 🌟 共享内存分配：
    // s_query : padded_dim * sizeof(float)
    // 4个数组 : 4 * max_cluster_size_ * sizeof(float或int)
    // s_LUT   : (padded_dim / 4) * 16 * sizeof(half)
    // 排序区   : 256 * (sizeof(float) + sizeof(int))
    // 🌟 共享内存: 与 NVIDIA computeInnerProductsWithLUT16OptBlockSort 完全一致的三区复用
    //   first  = max(LUT, MC*float);  second = MC*(float+int);  third = D*float
    //   排序暂存复用 first 区, 故总量 = max(三区之和, cub TempStorage 大小)
    size_t lut_bytes      = (size_t)(padded_dim / 4) * 16 * sizeof(half);                 // 8*D
    size_t mc_float_bytes = (size_t)this->max_cluster_size_ * sizeof(float);
    size_t first_bytes    = (lut_bytes > mc_float_bytes) ? lut_bytes : mc_float_bytes;    // max(LUT, MC*float)
    size_t second_bytes   = (size_t)this->max_cluster_size_ * (sizeof(float) + sizeof(int)); // 8*MC
    size_t third_bytes    = (size_t)padded_dim * sizeof(float);                           // 4*D
    size_t parts_bytes    = first_bytes + second_bytes + third_bytes;
    size_t cub_temp_bytes = sizeof(cub::BlockRadixSort<float, 256, 4, int>::TempStorage);
    size_t shared_mem_size = (parts_bytes > cub_temp_bytes) ? parts_bytes : cub_temp_bytes;
    cudaEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr;
    if (profiling_enabled_) {
        CUDA_CHECK(cudaEventCreate(&e0));
        CUDA_CHECK(cudaEventCreate(&e1));
        CUDA_CHECK(cudaEventCreate(&e2));
        CUDA_CHECK(cudaEventRecord(e0, stream));
    }
    if (long_codes_are_host_mapped_ || force_tiled_fine_kernel_) {
        // Fixed shared-memory footprint: LUT + query + three 512-item tile
        // arrays + two runtime-topk arrays.  Passing --fine-kernel tiled makes
        // both residency modes use this path for a controlled comparison.
        const size_t tiled_shared_mem_size =
            lut_bytes +
            (size_t)padded_dim * sizeof(float) +
            (size_t)kFineTileSize * (2 * sizeof(float) + sizeof(int)) +
            (size_t)topk * (sizeof(float) + sizeof(int));
        computeInnerProductsWithBitwiseOptExactPruningTiled
            <<<total_probes, kFineBlockThreads,
               tiled_shared_mem_size, stream>>>(bitwiseParams);
        const cudaError_t launch_error = cudaGetLastError();
        if (launch_error != cudaSuccess) {
            throw std::runtime_error(
                std::string("Tiled GPU fine-search launch failed: ") +
                cudaGetErrorString(launch_error) +
                "; dynamic_shared_bytes=" + std::to_string(tiled_shared_mem_size));
        }
    } else {
        computeInnerProductsWithBitwiseOptExactPruning
            <<<total_probes, 256, shared_mem_size, stream>>>(bitwiseParams);
        // Check here so an oversized shared-memory launch is attributed to
        // fine search instead of surfacing inside the following RAFT top-k.
        const cudaError_t launch_error = cudaGetLastError();
        if (launch_error != cudaSuccess) {
            throw std::runtime_error(
                std::string("Original GPU fine-search launch failed: ") +
                cudaGetErrorString(launch_error) +
                "; max_cluster_size=" + std::to_string(max_cluster_size_) +
                "; dynamic_shared_bytes=" + std::to_string(shared_mem_size) +
                "; use --fine-kernel tiled for clusters exceeding shared-memory limits");
        }
    }
    TOCK("7. Fine Search (Exact Dynamic Pruning)");
    if (profiling_enabled_) CUDA_CHECK(cudaEventRecord(e1, stream));
    
    // -------------------------------------------------------------
    // 8. 全局聚合 
    // -------------------------------------------------------------
    TICK();
    size_t final_bytes = num_queries * topk * sizeof(float);
    size_t reduced_pool_size = nprobe * topk; 
    
    // 复用 search() 顶部创建的 handle（粗排簇选择阶段已使用）
    const int n_rows = (int)num_queries;
    const int pool   = (int)reduced_pool_size;   // 候选池每行宽度 = nprobe*topk 或你的 reduced 宽度
    const int k      = (int)topk;

    raft::matrix::select_k<float, uint32_t>(
        handle,
        raft::make_device_matrix_view<const float,    int64_t>(d_candidates_dists, n_rows, pool),
        std::make_optional(
        raft::make_device_matrix_view<const uint32_t, int64_t>(d_candidates_pids,  n_rows, pool)),
        raft::make_device_matrix_view<float,    int64_t>(d_final_dev_dists, n_rows, k),
        raft::make_device_matrix_view<uint32_t, int64_t>(d_final_dev_pids,  n_rows, k),
        /*select_min=*/true,     // 距离越小越好 → 选最小的 k 个
        /*sorted=*/false);       // 组内不排序，最快

    handle.sync_stream();        // Host immediately consumes the result below.
    float t_fine_ms = 0.0f, t_gtopk_ms = 0.0f;
    if (profiling_enabled_) {
        CUDA_CHECK(cudaEventRecord(e2, stream));
        CUDA_CHECK(cudaEventSynchronize(e2));
        CUDA_CHECK(cudaEventElapsedTime(&t_fine_ms, e0, e1));
        CUDA_CHECK(cudaEventElapsedTime(&t_gtopk_ms, e1, e2));
    }
    TOCK("7.5 Global Top-K (cub)");

    // 动机测量：fine kernel 内部 scan/rerank/sort 周期占比（由 d_phase_cycles 在 fine kernel 内累加）
    if (profiling_enabled_) {
        unsigned long long h[6];
        CUDA_CHECK(cudaMemcpy(
            h, d_phase_cycles, sizeof(h), cudaMemcpyDeviceToHost));
        const double cyc = static_cast<double>(h[0] + h[1] + h[2]);
        const double scan_ms = cyc ? t_fine_ms * (h[0] / cyc) : 0.0;
        const double rerank_ms = cyc ? t_fine_ms * (h[1] / cyc) : 0.0;
        const double sort_ms = (cyc ? t_fine_ms * (h[2] / cyc) : 0.0) +
                               t_gtopk_ms;
        const double total_ms = t_fine_ms + t_gtopk_ms;
        printf("scan=%.1f%%  rerank=%.1f%%  sort=%.1f%%  | rerank+sort=%.1f%%\n",
            total_ms ? 100 * scan_ms / total_ms : 0.0,
            total_ms ? 100 * rerank_ms / total_ms : 0.0,
            total_ms ? 100 * sort_ms / total_ms : 0.0,
            total_ms ? 100 * (rerank_ms + sort_ms) / total_ms : 0.0);

        // 1-bit scan count and ex-code refinement count for Figure 2(b).
        const unsigned long long n_cand = h[3];
        const unsigned long long n_refine = h[4];
        const unsigned long long n_scan = h[5];
        last_scanned_vectors_ = n_scan;
        last_refined_vectors_ = n_refine;
        printf("[Figure3] mode=cluster-local topk=%zu nprobe=%zu filtering_ms=%.6f refinement_ms=%.6f topk_maintenance_ms=%.6f refinement_ratio=%.6f scanned=%llu candidates=%llu refined=%llu\n",
            topk, nprobe, scan_ms, rerank_ms, sort_ms,
            n_scan ? (double)n_refine / (double)n_scan : 0.0,
            n_scan, n_cand, n_refine);
        printf("[ProductionTiming] mode=fused topk=%zu nprobe=%zu "
               "fine_search_ms=%.6f final_topk_ms=%.6f\n",
               topk, nprobe, t_fine_ms, t_gtopk_ms);
        printf("[Refine] scanned=%llu candidates=%llu refined=%llu refinement_ratio=%.6f candidate_ratio=%.6f\n",
            n_scan, n_cand, n_refine,
            n_scan ? (double)n_refine / (double)n_scan : 0.0,
            n_scan ? (double)n_cand / (double)n_scan : 0.0);
    }
    // 注：map_pid_kernel 已被 gather 取代，这里不再需要 PID Mapping 阶段。

    // =========================================================
    // 9. 结果拷贝回 Host (D2H) & 清理状态
    // =========================================================
    TICK();
    CUDA_CHECK(cudaMemcpyAsync(h_final_dists, d_final_dev_dists, final_bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_final_pids, d_final_dev_pids, num_queries * topk * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    if (h_selected_cluster_ids != nullptr) {
        CUDA_CHECK(cudaMemcpyAsync(
            h_selected_cluster_ids, d_probe_idx,
            num_queries * nprobe * sizeof(int),
            cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    //打印查询0的h_final_dists和h_final_pids的前10个元素以验证结果
    // for (int i = 0; i < std::min(10, (int)(num_queries * topk)); i++) {
    //     if (i < topk) {
    //         std::cout << "Query 0, Rank " << i << ": Dist = " << h_final_dists[i] << ", PID = " << h_final_pids[i] << "\n";
    //     }
    // }
    CUDA_CHECK(cudaMemsetAsync(d_final_dev_pids, 0, num_queries * topk * sizeof(uint32_t), stream));
    TOCK("9. Results D2H Transfer");

    if (profiling_enabled_) {
        std::cout << "------------------------------------------\n";
    }

    // 释放测速组件
    if (profiling_enabled_) {
        CUDA_CHECK(cudaFree(d_phase_cycles));
        CUDA_CHECK(cudaEventDestroy(e0));
        CUDA_CHECK(cudaEventDestroy(e1));
        CUDA_CHECK(cudaEventDestroy(e2));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    #undef TICK
    #undef TOCK
}

void IVF_RaBitQ_Raw::profile_isolated_last_search(
    size_t num_queries, size_t topk, size_t nprobe)
{
    if (!isolated_profiling_enabled_) return;
    if (!profiling_enabled_ || last_scanned_vectors_ == 0 ||
        last_refined_vectors_ == 0) {
        throw std::runtime_error(
            "isolated profiling requires a completed --profile true search");
    }
    if (isolated_profile_rounds_ == 0) {
        throw std::runtime_error("isolated profile rounds must be positive");
    }
    if (num_queries > ws_max_queries || topk > ws_max_topk ||
        nprobe > ws_max_nprobe) {
        throw std::runtime_error(
            "isolated profiling parameters exceed workspace capacity");
    }

    const int total_probes = static_cast<int>(num_queries * nprobe);
    if (last_refined_vectors_ >
        std::numeric_limits<unsigned long long>::max() /
            last_scanned_vectors_) {
        throw std::runtime_error(
            "isolated profiling count product would overflow uint64");
    }
    // Allocate only the diagnostic scratch actually needed, after the timed
    // search. Normal performance runs never reserve these per-pair arrays.
    rmm::device_uvector<float> pair_checksums(
        total_probes, rmm::cuda_stream_view{stream});
    rmm::device_uvector<unsigned long long> pair_scan_offsets(
        total_probes, rmm::cuda_stream_view{stream});
    float* d_pair_checksums = pair_checksums.data();
    unsigned long long* d_pair_scan_offsets = pair_scan_offsets.data();

    ComputeBitwiseKernelParams params{};
    params.d_sorted_pairs = ws_d_sorted_pairs;
    params.d_cluster_meta = d_cluster_meta;
    params.d_global_luts = ws_d_global_luts;
    params.d_G_k1xSumq = ws_d_G_k1xSumq;
    params.d_centroid_distances = ws_d_centroid_dists;
    params.d_threshold = ws_d_threshold;
    params.d_short_data = d_short_data;
    params.d_short_factors = d_short_factors;
    params.num_queries = static_cast<int>(num_queries);
    params.num_centroids = static_cast<int>(num_centroids);
    params.D = static_cast<int>(padded_dim);
    params.num_words = static_cast<int>((padded_dim + 31) / 32);
    params.num_pairs = total_probes;
    params.nprobe = static_cast<int>(nprobe);
    params.topk = static_cast<int>(topk);

    const size_t lut_bytes =
        (size_t)(padded_dim / 4) * 16 * sizeof(half);
    const size_t shared_mem_size =
        lut_bytes +
        padded_dim * sizeof(float) +
        (size_t)kFineTileSize *
            (2 * sizeof(float) + sizeof(int)) +
        topk * (sizeof(float) + sizeof(int));
    const size_t rounds = isolated_profile_rounds_;

    prepareProfilePairScanOffsets
        <<<(total_probes + 255) / 256, 256, 0, stream>>>(
            params, d_pair_scan_offsets);
    CUDA_CHECK(cudaPeekAtLastError());
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        d_pair_scan_offsets,
        d_pair_scan_offsets + total_probes,
        d_pair_scan_offsets);

    const float filtering_ms = measure_isolated_kernel_ms(
        stream, rounds, [&]() {
            profileFilteringIsolated
                <<<total_probes, kFineBlockThreads,
                   shared_mem_size, stream>>>(
                    params, d_pair_scan_offsets,
                    last_refined_vectors_, last_scanned_vectors_,
                    d_pair_checksums);
        });
    const float cluster_topk_ms = measure_isolated_kernel_ms(
        stream, rounds, [&]() {
            profileClusterTopkIsolated
                <<<total_probes, kFineBlockThreads,
                   shared_mem_size, stream>>>(
                    params, d_pair_scan_offsets,
                    last_refined_vectors_, last_scanned_vectors_,
                    d_pair_checksums);
        });

    printf(
        "[ControlledProfile] mode=work-equivalent-non-additive "
        "median_rounds=%zu topk=%zu nprobe=%zu filtering_ms=%.6f "
        "cluster_topk_ms=%.6f refinement_ratio=%.6f "
        "scanned=%llu refined=%llu "
        "candidate_model=proportional-prefix-exact-count\n",
        rounds, topk, nprobe, filtering_ms, cluster_topk_ms,
        last_scanned_vectors_
            ? (double)last_refined_vectors_ /
                  (double)last_scanned_vectors_
            : 0.0,
        last_scanned_vectors_, last_refined_vectors_);
}
