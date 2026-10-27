#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <cfloat> // 提供 FLT_MAX
#include <cub/cub.cuh>
#include <omp.h>
#include <queue>

// 引入头文件
#include "coraanns/experimental/cpu_gpu.cuh"

struct PoolOffsetOp {
    int pool_size;
    __host__ __device__ __forceinline__ int operator()(int idx) const { 
        return idx * pool_size; 
    }
};

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
        int q_idx = idx / nprobe;
        int p_idx = idx % nprobe; // 局部的序号 (0 ~ nprobe-1)

        d_pairs[idx].cluster_idx = d_probe_idx[idx];
        d_pairs[idx].query_idx = q_idx;
        d_pairs[idx].probe_idx = p_idx; // 🌟 存入结构体
    }
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

// ==========================================
// 类函数实现
// ==========================================
IVF_RaBitQ_Raw::IVF_RaBitQ_Raw() {
    CUDA_CHECK(cudaStreamCreate(&stream));
    cublasCreate(&cublas_handle);
    cublasSetStream(cublas_handle, stream);
}

IVF_RaBitQ_Raw::~IVF_RaBitQ_Raw() {
    cublasDestroy(cublas_handle);
    CUDA_CHECK(cudaStreamDestroy(stream));
    if (d_centroids)     CUDA_CHECK(cudaFree(d_centroids));
    if (d_rotator)       CUDA_CHECK(cudaFree(d_rotator));
    if (d_short_data)    CUDA_CHECK(cudaFree(d_short_data));
    if (d_short_factors) CUDA_CHECK(cudaFree(d_short_factors));
    if (d_cluster_meta)  CUDA_CHECK(cudaFree(d_cluster_meta));
    free_workspace();
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
        const size_t* h_cluster_sizes
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

    h_long_codes_ref = h_long_codes;
    h_ex_factors_ref = h_ex_factors;
    h_pids_ref = h_ids;
    std::cout << "Engine loaded via Custom Manual Transpose successfully!" << std::endl;
}



void IVF_RaBitQ_Raw::prepare_workspace(size_t max_queries, size_t max_topk, size_t max_nprobe) {
    free_workspace(); // 如果已经分配过，先清空

    ws_max_queries = max_queries;
    ws_max_topk = max_topk;
    ws_max_nprobe = max_nprobe;

    size_t max_total_probes = max_queries * max_nprobe;
    size_t candidate_pool_size = max_queries * max_nprobe * max_topk;
    size_t M = M_candidates_to_cpu; // 1000

    CUDA_CHECK(cudaMallocAsync(&ws_d_candidates_low_dists_sorted, candidate_pool_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_out_indices_temp, candidate_pool_size * sizeof(int), stream));

    CUDA_CHECK(cudaMallocAsync(&ws_d_candidates_low_dists, candidate_pool_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_candidates_exact_ips, candidate_pool_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_candidates_q_g_adds, candidate_pool_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_candidates_gids, candidate_pool_size * sizeof(uint32_t), stream));

    CUDA_CHECK(cudaMallocAsync(&ws_d_out_low_dists, max_queries * M * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_out_exact_ips, max_queries * M * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_out_q_g_adds, max_queries * M * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_out_gids, max_queries * M * sizeof(uint32_t), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_out_indices, max_queries * M * sizeof(int), stream));

    // 分配 Host 侧接收缓存 (使用 cudaMallocHost 可加速 PCIe 传输)
    cudaMallocHost(&h_rotated_queries, max_queries * padded_dim * sizeof(float));
    cudaMallocHost(&h_G_kbxSumq, max_queries * sizeof(float));
    cudaMallocHost(&h_out_low_dists, max_queries * M * sizeof(float));
    cudaMallocHost(&h_out_exact_ips, max_queries * M * sizeof(float));
    cudaMallocHost(&h_out_q_g_adds, max_queries * M * sizeof(float));
    cudaMallocHost(&h_out_gids, max_queries * M * sizeof(uint32_t));

    // 一次性分配所有可能会在 Search 中用到的临时显存
    CUDA_CHECK(cudaMallocAsync(&ws_d_queries, max_queries * padded_dim * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_raw_queries, max_queries * dim * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_centroid_dists, max_queries * num_centroids * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_centroid_dists_copy, max_queries * num_centroids * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_q_norms, max_queries * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_c_norms, num_centroids * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_probe_vals, max_queries * max_nprobe * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_probe_idx, max_queries * max_nprobe * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_sorted_pairs, max_total_probes * sizeof(ClusterQueryPair), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_G_k1xSumq, max_queries * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_G_kbxSumq, max_queries * sizeof(float), stream));

    CUDA_CHECK(cudaMallocAsync(&ws_d_candidates_low_dists_sorted, candidate_pool_size * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_out_indices_temp, candidate_pool_size * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_out_indices_sorted, candidate_pool_size * sizeof(int), stream));
    
    size_t total_centroids_items = max_queries * num_centroids;
    CUDA_CHECK(cudaMallocAsync(&ws_d_centroid_dists_sorted, total_centroids_items * sizeof(float), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_centroid_indices_temp, total_centroids_items * sizeof(int), stream));
    CUDA_CHECK(cudaMallocAsync(&ws_d_centroid_indices_sorted, total_centroids_items * sizeof(int), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void IVF_RaBitQ_Raw::free_workspace() {
    if(ws_d_queries) cudaFree(ws_d_queries);
    if(ws_d_raw_queries) cudaFree(ws_d_raw_queries);
    if(ws_d_centroid_dists) cudaFree(ws_d_centroid_dists);
    if(ws_d_centroid_dists_copy) cudaFree(ws_d_centroid_dists_copy);
    if(ws_d_q_norms) cudaFree(ws_d_q_norms);
    if(ws_d_c_norms) cudaFree(ws_d_c_norms);
    if(ws_d_probe_vals) cudaFree(ws_d_probe_vals);
    if(ws_d_probe_idx) cudaFree(ws_d_probe_idx);
    if(ws_d_sorted_pairs) cudaFree(ws_d_sorted_pairs);
    if(ws_d_G_k1xSumq) cudaFree(ws_d_G_k1xSumq);
    if(ws_d_G_kbxSumq) cudaFree(ws_d_G_kbxSumq);


    ws_d_queries = nullptr; ws_d_centroid_dists = nullptr; ws_d_centroid_dists_copy = nullptr;
    ws_d_q_norms = nullptr; ws_d_c_norms = nullptr; ws_d_probe_vals = nullptr;
    ws_d_probe_idx = nullptr; ws_d_sorted_pairs = nullptr; ws_d_G_k1xSumq = nullptr;
    ws_d_G_kbxSumq = nullptr;
}

// 🌟 全新 1-Bit 粗筛 Kernel：只算 low_dist 和 exact_ip + CUB 极速排序
__global__ void computeInnerProducts1BitBlockSort(const ComputeBitwiseKernelParams params) {
    int block_id = blockIdx.x;
    if (block_id >= params.num_pairs) return;

    ClusterQueryPair pair = params.d_sorted_pairs[block_id];
    int cluster_idx = pair.cluster_idx;
    int query_idx = pair.query_idx;
    size_t num_vectors_in_cluster = params.d_cluster_meta[cluster_idx].num;
    size_t cluster_start_index = params.d_cluster_meta[cluster_idx].start_index;

    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // =========================================================
    // 🌟 计时变量定义 (仅在 Block 0, Thread 0 记录)
    // =========================================================
    long long t_start = 0, t_phase1 = 0, t_phase2 = 0;

    extern __shared__ char smem[];
    float* s_query = (float*)smem;
    float* s_exact_ip = s_query + params.D;
    float* s_low_dist = s_exact_ip + params.max_candidates_per_pair;
    uint32_t* s_gid = (uint32_t*)(s_low_dist + params.max_candidates_per_pair);

    for (int i = tid; i < params.D; i += num_threads) {
        s_query[i] = params.d_query[query_idx * params.D + i];
    }
    __syncthreads();

    // 🌟 记录起点：屏蔽了加载 Query 的开销，直接测算核心逻辑
    if (block_id % 20 == 0 && tid == 0) t_start = clock64();

    float q_g_add = params.d_centroid_distances[query_idx * params.num_centroids + cluster_idx];
    float q_k1xsumq = params.d_G_k1xSumq[query_idx];
    float q_g_error = sqrtf(q_g_add);

    int actual_cands = min((int)num_vectors_in_cluster, params.max_candidates_per_pair);
    size_t short_code_length = params.D / 32;

    // for (int c = tid; c < actual_cands; c += num_threads) {
    //     size_t global_idx = cluster_start_index + c;
    //     float exact_ip = 0.0f;

    //     for (int w = 0; w < short_code_length; w++) {
    //         // 🌟 优化 1：使用 __ldg 强制走只读数据缓存，大幅降低 L1 Cache 压力
    //         uint32_t chunk = __ldg(&params.d_short_data[cluster_start_index * short_code_length + w * num_vectors_in_cluster + c]);
            
    //         // 🌟 优化 2：彻底移除 if 判断，并将 32 次循环完全强制展开
    //         #pragma unroll 32
    //         for (int b = 0; b < 32; b++) {
    //             // 提取当前 bit 并直接转为浮点数 (0.0f 或 1.0f)
    //             float mask = (float)((chunk >> (31 - b)) & 1U);
    //             exact_ip += s_query[w * 32 + b] * mask; 
    //         }
    //     }
        
    //     // 同样使用 __ldg 优化因子读取
    //     float3 factors = reinterpret_cast<const float3*>(params.d_short_factors)[global_idx];
    //     float low_dist = factors.x + q_g_add + factors.y * (exact_ip + q_k1xsumq) - factors.z * q_g_error;

    //     s_exact_ip[c] = exact_ip;
    //     s_low_dist[c] = low_dist;
    //     s_gid[c] = global_idx;
    // }
    //优化代码
    for (int c = tid; c < actual_cands; c += num_threads) {
        size_t global_idx = cluster_start_index + c;

        // 🌟 优化 1：打破数据依赖链，使用 4 个独立累加器榨干并发流水线 (ILP)
        float exact_ip0 = 0.0f;
        float exact_ip1 = 0.0f;
        float exact_ip2 = 0.0f;
        float exact_ip3 = 0.0f;

        for (int w = 0; w < short_code_length; w++) {
            // 使用 __ldg 强制走只读数据缓存，大幅降低 L1 Cache 压力
            uint32_t chunk = __ldg(&params.d_short_data[cluster_start_index * short_code_length + w * num_vectors_in_cluster + c]);
            const float* sq = &s_query[w * 32]; // 🌟 提取当前 word 的查询向量首地址，加速寻址

            // 🌟 优化 2：彻底移除浮点转换与乘法！使用底层谓词加法 (@p add.f32)
            // 此时编译器会并发派发这 4 条指令，吞吐量提升 300%
            #pragma unroll 8
            for (int b = 0; b < 8; b++) {
                if ((chunk >> (31 - b))      & 1U) exact_ip0 += sq[b];
                if ((chunk >> (23 - b))      & 1U) exact_ip1 += sq[b + 8];
                if ((chunk >> (15 - b))      & 1U) exact_ip2 += sq[b + 16];
                if ((chunk >> (7  - b))      & 1U) exact_ip3 += sq[b + 24];
            }
        }
        
        // 最后将 4 条支流汇聚
        float exact_ip = exact_ip0 + exact_ip1 + exact_ip2 + exact_ip3;

        float3 factors = reinterpret_cast<const float3*>(params.d_short_factors)[global_idx];
        float low_dist = factors.x + q_g_add + factors.y * (exact_ip + q_k1xsumq) - factors.z * q_g_error;

        // 存入共享内存
        s_exact_ip[c] = exact_ip;
        s_low_dist[c] = low_dist;
        s_gid[c] = global_idx;
    }
    __syncthreads();

    // 🌟 记录阶段 1 结束
    if (block_id % 20 == 0 && tid == 0) t_phase1 = clock64();

    // -------------------------------------------------------------
    // 阶段 2：CUB BlockRadixSort 极速排序
    // -------------------------------------------------------------
    // 设定 Block 的线程数和每个线程处理的元素数
    // 假设 blockDim.x 为 256，每个线程 4 个元素，单 Block 最大支撑 1024 个候选
    constexpr int BLOCK_THREADS = 256;
    constexpr int ITEMS_PER_THREAD = 4; 
    
    typedef cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int> BlockRadixSort;
    
    // 使用静态共享内存为 CUB 分配交换空间，与你的 extern 动态共享内存互不干扰
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    // 线程私有寄存器数组
    float thread_keys[ITEMS_PER_THREAD];
    int   thread_vals[ITEMS_PER_THREAD];

    // 1. 将数据从 Shared Memory 读入寄存器
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = tid * ITEMS_PER_THREAD + i;
        if (idx < actual_cands) {
            thread_keys[i] = s_low_dist[idx]; // Key 是距离
            thread_vals[i] = idx;             // Value 是在共享内存里的局部下标
        } else {
            thread_keys[i] = FLT_MAX;
            thread_vals[i] = -1;
        }
    }
    
    __syncthreads(); 

    // 2. 召唤 CUB 魔法：纯寄存器级极速排序 (数千 cycle 瞬间完成)
    BlockRadixSort(temp_storage).Sort(thread_keys, thread_vals);

    // 3. 将排好序的前 Top-K 提取并写回 Global Memory
    uint32_t out_offset = query_idx * (params.topk * params.nprobe) + pair.probe_idx * params.topk;

    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int sorted_rank = tid * ITEMS_PER_THREAD + i;
        if (sorted_rank < params.topk) {
            int original_idx = thread_vals[i];
            
            if (original_idx != -1) {
                // 通过局部下标原路反查出 exact_ip 和 gid
                params.d_candidates_low_dists[out_offset + sorted_rank] = thread_keys[i];
                params.d_candidates_exact_ips[out_offset + sorted_rank] = s_exact_ip[original_idx];
                params.d_candidates_q_g_adds[out_offset + sorted_rank]  = q_g_add;
                params.d_candidates_gids[out_offset + sorted_rank]      = s_gid[original_idx];
            } else {
                params.d_candidates_low_dists[out_offset + sorted_rank] = FLT_MAX;
                params.d_candidates_exact_ips[out_offset + sorted_rank] = 0.0f;
                params.d_candidates_q_g_adds[out_offset + sorted_rank]  = 0.0f;
                params.d_candidates_gids[out_offset + sorted_rank]      = 0xFFFFFFFF;
            }
        }
    }

    // 🌟 记录阶段 2 结束
    if (block_id % 20 == 0 && tid == 0) {
        t_phase2 = clock64();
        // printf("Block %d, Phase 1 (1-Bit IP): %lld cycles, Phase 2 (CUB Sort): %lld cycles\n", 
        //         block_id, t_phase1 - t_start, t_phase2 - t_phase1);
    }
}

// 🌟 收集 Kernel：根据索引收集属性矩阵
__global__ void gather_topM_kernel(int num_queries, int M, int pool_size,
        const int* d_topM_idx,
        const float* d_exact_ips, const float* d_q_g_adds, const uint32_t* d_gids,
        float* out_exact_ips, float* out_q_g_adds, uint32_t* out_gids)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries * M) {
        int q_id = idx / M;
        int local_idx = d_topM_idx[idx];
        if (local_idx != -1) {
            int src_idx = q_id * pool_size + local_idx;
            out_exact_ips[idx] = d_exact_ips[src_idx];
            out_q_g_adds[idx]  = d_q_g_adds[src_idx];
            out_gids[idx]      = d_gids[src_idx];
        } else {
            out_gids[idx] = 0xFFFFFFFF;
        }
    }
}

// 1. 用于初始化局部下标数组 [0, 1, 2, ..., pool_size-1]
__global__ void init_local_indices_kernel(int* indices, int pool_size, int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_elements) {
        indices[idx] = idx % pool_size;
    }
}

// 2. 升级版的 Gather Kernel (处理 CUB 排序后紧凑提取前 M 个)
__global__ void gather_topM_kernel_v2(int num_queries, int M, int pool_size,
        const float* d_sorted_low_dists,
        const int* d_sorted_indices,
        const float* d_exact_ips, 
        const float* d_q_g_adds, 
        const uint32_t* d_gids,
        float* out_low_dists,
        float* out_exact_ips, 
        float* out_q_g_adds, 
        uint32_t* out_gids)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries * M) {
        int q_id = idx / M;
        int m_id = idx % M;
        
        // CUB 排序后的数组，每个 query 的全量数据依然占据 pool_size 长度
        int sorted_idx = q_id * pool_size + m_id;
        int local_idx = d_sorted_indices[sorted_idx];
        
        // 补充导出排好序的 low_dist
        out_low_dists[idx] = d_sorted_low_dists[sorted_idx]; 
        
        if (local_idx != -1 && local_idx < pool_size) {
            int src_idx = q_id * pool_size + local_idx;
            out_exact_ips[idx] = d_exact_ips[src_idx];
            out_q_g_adds[idx]  = d_q_g_adds[src_idx];
            out_gids[idx]      = d_gids[src_idx];
        } else {
            out_gids[idx] = 0xFFFFFFFF;
        }
    }
}

// 专门用于从排序好的 Centroid 距离中提取前 nprobe 个
__global__ void gather_nprobe_kernel(int num_queries, int nprobe, int num_centroids,
        const float* d_sorted_dists,
        const int* d_sorted_indices,
        float* out_probe_vals,
        int* out_probe_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_queries * nprobe) {
        int q_id = idx / nprobe;
        int n_id = idx % nprobe;
        
        // 排序后的全量数组，每个 query 占 num_centroids 个位置
        int src_idx = q_id * num_centroids + n_id;
        
        out_probe_vals[idx] = d_sorted_dists[src_idx];
        out_probe_idx[idx]  = d_sorted_indices[src_idx];
    }
}

void IVF_RaBitQ_Raw::search(const float* h_queries, size_t num_queries, size_t topk, size_t nprobe, float* h_final_dists, uint32_t* h_final_pids) {
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
    float* d_centroid_dists_copy  = ws_d_centroid_dists_copy;
    float* d_q_norms              = ws_d_q_norms;
    float* d_c_norms              = ws_d_c_norms;
    float* d_probe_vals           = ws_d_probe_vals;
    int* d_probe_idx            = ws_d_probe_idx;
    ClusterQueryPair* d_sorted_pairs = ws_d_sorted_pairs;
    float* d_G_k1xSumq            = ws_d_G_k1xSumq;
    float* d_G_kbxSumq            = ws_d_G_kbxSumq;
    
    // =========================================================
    // 🌟 初始化 CUDA 事件，用于高精度测速
    // =========================================================
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms = 0.0f;

    // 定义测速宏，保持代码整洁
    #define TICK() cudaEventRecord(start, stream)
    #define TOCK(step_name) \
        cudaEventRecord(stop, stream); \
        cudaEventSynchronize(stop); \
        cudaEventElapsedTime(&ms, start, stop); \
        std::cout << "[Profile] " << step_name << " 耗时: " << ms << " ms\n"


    // =========================================================
    // 0. Query 数据拷贝 (H2D)
    // =========================================================
    TICK();
    size_t raw_bytes = num_queries * dim * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(d_raw_queries, h_queries, raw_bytes, cudaMemcpyHostToDevice, stream));
    TOCK("0. Raw Query H2D Transfer");

    // 🌟 0.5 GPU 旋转：Q_rotated = Q_raw * Rotator
    // 对应 cuBLAS 列主序计算：(Q_rotated)^T = (Rotator)^T * (Q_raw)^T
    TICK();
    const float alpha_rot = 1.0f; const float beta_rot = 0.0f;
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                padded_dim, num_queries, dim,
                &alpha_rot, d_rotator, padded_dim, 
                d_raw_queries, dim, 
                &beta_rot, d_queries, padded_dim);
    CUDA_CHECK(cudaStreamSynchronize(stream));
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
    CUDA_CHECK(cudaStreamSynchronize(stream));
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
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int add_threads = 256;
    int add_blocks  = (num_queries * num_centroids + add_threads - 1) / add_threads;
    add_norms_kernel<<<add_blocks, add_threads, 0, stream>>>(d_centroid_dists, d_q_norms, d_c_norms, num_queries, num_centroids);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOCK("2. L2 Norms Completion");
    
    if (topk < 200) {
        // =========================================================
        // 3. Top-nprobe 簇选择
        // 为每个 Query 选出距离最近的 nprobe 个簇，准备进行细排
        // =========================================================
        TICK();
        // 拷贝一份用于破坏性 TopK 筛选
        CUDA_CHECK(cudaMemcpyAsync(d_centroid_dists_copy, d_centroid_dists, num_queries * num_centroids * sizeof(float), cudaMemcpyDeviceToDevice, stream));
        select_topk_kernel<<<num_queries, 256, 0, stream>>>(d_centroid_dists_copy, num_centroids, nprobe, d_probe_vals, d_probe_idx);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        TOCK("3. Select Top-nprobe Clusters");
    } else {
        // =========================================================
        // 3. Top-nprobe 簇选择 (CUB 分段基数排序极速版)
        // =========================================================
        TICK();
        int total_centroids_items = num_queries * num_centroids;

        // a. 准备局部索引 [0, 1, ... num_centroids-1]
        // 复用之前的 init_local_indices_kernel
        int c_threads = 256;
        int c_grid = (total_centroids_items + c_threads - 1) / c_threads;
        init_local_indices_kernel<<<c_grid, c_threads, 0, stream>>>(ws_d_centroid_indices_temp, num_centroids, total_centroids_items);

        // b. 构建虚拟的 Offset 迭代器 (复用之前定义的全局 PoolOffsetOp)
        cub::TransformInputIterator<int, PoolOffsetOp, cub::CountingInputIterator<int>> c_offsets(
            cub::CountingInputIterator<int>(0), PoolOffsetOp{(int)num_centroids}
        );

        // c. 调用 CUB 魔法排序 (直接从 d_centroid_dists 读，不破坏原数组！)
        size_t c_temp_storage_bytes = 0;
        cub::DeviceSegmentedRadixSort::SortPairs(
            nullptr, c_temp_storage_bytes,
            d_centroid_dists, ws_d_centroid_dists_sorted,          // keys
            ws_d_centroid_indices_temp, ws_d_centroid_indices_sorted, // values
            total_centroids_items, num_queries, c_offsets, c_offsets + 1, 
            0, sizeof(float)*8, stream
        );

        void* d_c_temp_storage = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_c_temp_storage, c_temp_storage_bytes, stream));

        cub::DeviceSegmentedRadixSort::SortPairs(
            d_c_temp_storage, c_temp_storage_bytes,
            d_centroid_dists, ws_d_centroid_dists_sorted,
            ws_d_centroid_indices_temp, ws_d_centroid_indices_sorted, 
            total_centroids_items, num_queries, c_offsets, c_offsets + 1,
            0, sizeof(float)*8, stream
        );
        CUDA_CHECK(cudaFreeAsync(d_c_temp_storage, stream));

        // d. 提取前 nprobe 个目标簇
        int p_threads = 256;
        int p_blocks = (num_queries * nprobe + p_threads - 1) / p_threads;
        gather_nprobe_kernel<<<p_blocks, p_threads, 0, stream>>>(num_queries, nprobe, num_centroids,
            ws_d_centroid_dists_sorted, ws_d_centroid_indices_sorted,
            d_probe_vals, d_probe_idx);
            
        CUDA_CHECK(cudaStreamSynchronize(stream));
        TOCK("3. Select Top-nprobe Clusters (CUB)");
    }

    // =========================================================
    // 4. Pair 扁平化与聚类排序
    // 将待查的簇转为 <Cluster, Query> 键值对，并按 Cluster 排序以增加访存局部性
    // =========================================================
    TICK();
    int total_probes = num_queries * nprobe;
    flatten_pairs_kernel<<<(total_probes + 255) / 256, 256, 0, stream>>>(d_probe_idx, num_queries, nprobe, d_sorted_pairs);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    thrust::sort(thrust::cuda::par.on(stream), d_sorted_pairs, d_sorted_pairs + total_probes, PairSortCompare());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOCK("4. Flatten & Sort Pairs");

    // =========================================================
    // 5. 预计算 Query 还原因子
    // 提前算出 RaBitQ 公式中与 Query 相关的常数项
    // =========================================================
    TICK();
    computeQueryFactors<float>(d_queries, d_G_k1xSumq, d_G_kbxSumq, num_queries, padded_dim, ex_bits, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOCK("5. Precompute Query Factors");

    // -------------------------------------------------------------
    // 6. GPU 1-Bit 筛选，直接输出局部 TopK
    // -------------------------------------------------------------
    TICK();
    ComputeBitwiseKernelParams bitwiseParams;
    bitwiseParams.d_sorted_pairs = d_sorted_pairs;
    bitwiseParams.d_cluster_meta = d_cluster_meta;
    bitwiseParams.d_query = d_queries; 
    bitwiseParams.d_G_k1xSumq = d_G_k1xSumq; 
    bitwiseParams.d_centroid_distances = d_centroid_dists;
    bitwiseParams.d_short_data = d_short_data;
    bitwiseParams.d_short_factors = d_short_factors;
    bitwiseParams.d_candidates_low_dists = ws_d_candidates_low_dists;
    bitwiseParams.d_candidates_exact_ips = ws_d_candidates_exact_ips;
    bitwiseParams.d_candidates_q_g_adds  = ws_d_candidates_q_g_adds;
    bitwiseParams.d_candidates_gids      = ws_d_candidates_gids;
    bitwiseParams.num_queries = num_queries;
    bitwiseParams.num_centroids = num_centroids;
    bitwiseParams.D = padded_dim;
    bitwiseParams.num_pairs = num_queries * nprobe;
    bitwiseParams.max_candidates_per_pair = this->max_cluster_size_;
    bitwiseParams.topk = topk;
    bitwiseParams.nprobe = nprobe;

    size_t shared_mem_size = (padded_dim + 2 * this->max_cluster_size_) * sizeof(float) + this->max_cluster_size_ * sizeof(uint32_t);
    computeInnerProducts1BitBlockSort<<<bitwiseParams.num_pairs, 256, shared_mem_size, stream>>>(bitwiseParams);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOCK("6. GPU 1-Bit Filter");

    // -------------------------------------------------------------
    // 6.5 & 7. GPU 全局聚合 Top-M (使用 CUB 极速分段基数排序)
    // -------------------------------------------------------------
    TICK();
    int pool_size = nprobe * topk;
    int M = M_candidates_to_cpu; // 1000
    int total_items = num_queries * pool_size;

    // a. 准备局部索引 [0, 1, ... pool_size-1]
    int threads = 256;
    int grid = (total_items + threads - 1) / threads;
    init_local_indices_kernel<<<grid, threads, 0, stream>>>(ws_d_out_indices_temp, pool_size, total_items);

    // b. 构建虚拟的 Offset 迭代器 (告诉 CUB 每个 Query 的数据段是从哪里切开的)
    cub::TransformInputIterator<int, PoolOffsetOp, cub::CountingInputIterator<int>> d_offsets(
        cub::CountingInputIterator<int>(0), PoolOffsetOp{pool_size}
    );

    // c. 调用 CUB 魔法排序
    size_t temp_storage_bytes = 0;
    
    // 第一次调用，获取 CUB 排序所需的内部临时显存大小
    cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, temp_storage_bytes,
        ws_d_candidates_low_dists, ws_d_candidates_low_dists_sorted, // keys: low_dists
        ws_d_out_indices_temp, ws_d_out_indices_sorted,              // values: indices
        total_items, num_queries, d_offsets, d_offsets + 1, 
        0, sizeof(float)*8, stream
    );

    // 分配 CUB 工作内存
    void* d_temp_storage = nullptr;
    cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);

    // 第二次调用，执行真实的极速并行排序
    cub::DeviceSegmentedRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        ws_d_candidates_low_dists, ws_d_candidates_low_dists_sorted,
        ws_d_out_indices_temp, ws_d_out_indices_sorted, 
        total_items, num_queries, d_offsets, d_offsets + 1,
        0, sizeof(float)*8, stream
    );
    
    // 释放 CUB 工作内存 (或者将其放在 prepare_workspace 里常驻，能再省 0.1ms)
    cudaFreeAsync(d_temp_storage, stream);
    TOCK("6.5 CUB Segmented Radix Sort");

    // d. 将排好序的前 M 个抽取压实并映射全局属性
    TICK();
    int blocks = (num_queries * M + 255) / 256;
    gather_topM_kernel_v2<<<blocks, 256, 0, stream>>>(num_queries, M, pool_size, 
        ws_d_candidates_low_dists_sorted, ws_d_out_indices_sorted, 
        ws_d_candidates_exact_ips, ws_d_candidates_q_g_adds, ws_d_candidates_gids,
        ws_d_out_low_dists, ws_d_out_exact_ips, ws_d_out_q_g_adds, ws_d_out_gids);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOCK("7. Gather Global Top-M");

    // // -------------------------------------------------------------
    // // 7. GPU 全局聚合 Top-M (输送给 CPU 的弹药)
    // // -------------------------------------------------------------
    // TICK();
    // int pool_size = nprobe * topk;
    // int M = M_candidates_to_cpu; // 比如 1000
    
    // // 从候选池中提取全局最小的 M 个 low_dist
    // select_topk_kernel<<<num_queries, 256, 0, stream>>>(ws_d_candidates_low_dists, pool_size, M, ws_d_out_low_dists, ws_d_out_indices);
    // TOCK("6.5 Select Top-M Low_dist");

    // TICK();
    // // 对应映射其他的属性值
    // int blocks = (num_queries * M + 255) / 256;
    // gather_topM_kernel<<<blocks, 256, 0, stream>>>(num_queries, M, pool_size, ws_d_out_indices, 
    //     ws_d_candidates_exact_ips, ws_d_candidates_q_g_adds, ws_d_candidates_gids,
    //     ws_d_out_exact_ips, ws_d_out_q_g_adds, ws_d_out_gids);
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    // TOCK("7. Gather Global Top-M");

    // -------------------------------------------------------------
    // 8. 弹药空投：GPU -> CPU D2H Transfer
    // -------------------------------------------------------------
    TICK();
    CUDA_CHECK(cudaMemcpyAsync(h_out_low_dists, ws_d_out_low_dists, num_queries * M * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_out_exact_ips, ws_d_out_exact_ips, num_queries * M * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_out_q_g_adds, ws_d_out_q_g_adds, num_queries * M * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_out_gids, ws_d_out_gids, num_queries * M * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_rotated_queries, d_queries, num_queries * padded_dim * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_G_kbxSumq, d_G_kbxSumq, num_queries * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    TOCK("8. Memory Transfer to CPU");

    // -------------------------------------------------------------
    // [Debug] 检查 D2H 传输后的内存数据 (Step 7 & 8)
    // -------------------------------------------------------------
    // printf("\n[Debug Step 8] Query 0 Top-5 Candidates from GPU:\n");
    // printf("  q_kbx: %.6f\n", h_G_kbxSumq[0]);
    // for (int i = 0; i < std::min(5, M); i++) {
    //     printf("  Rank %d: GID=%u, LowDist=%.6f, ExactIP=%.6f, q_g_add=%.6f\n", 
    //            i, h_out_gids[i], h_out_low_dists[i], h_out_exact_ips[i], h_out_q_g_adds[i]);
    // }
    // // 顺便检查旋转后的 Query 向量前 4 维是否正常
    // printf("  Rotated Query[0][0-3]: %.4f, %.4f, %.4f, %.4f\n\n", 
    //        h_rotated_queries[0], h_rotated_queries[1], h_rotated_queries[2], h_rotated_queries[3]);

    // -------------------------------------------------------------
    // 9. CPU 多线程火力网 (精确重排)
    // -------------------------------------------------------------
    TICK();
    // 呼叫定义在 .cpp 中的纯 CPU 函数，彻底规避 NVCC 编译冲突
    this->cpu_exact_rerank(num_queries, topk, h_final_dists, h_final_pids);
    
    TOCK("9. CPU Ex-bit Filtering (OpenMP)");

    std::cout << "------------------------------------------\n";

    // 释放测速组件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}