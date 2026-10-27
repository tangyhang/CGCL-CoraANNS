#pragma once
#include <string>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static constexpr int WARP_SIZE      = 32;
constexpr size_t FAST_SIZE = 32;

static constexpr int BITS_PER_CHUNK = 4;
static constexpr int LUT_SIZE       = (1 << BITS_PER_CHUNK);  // 16

// --- Tunables ---
using T    = float;
using IdxT = uint32_t;

using lut_dtype = __half;  // FP16 alternative

typedef uint32_t PID;

struct ClusterQueryPair {
  int cluster_idx;
  int query_idx;
  int probe_idx;
};

// 3. 自定义 Thrust 排序规则 (按 cluster_idx 从小到大排)
struct PairSortCompare {
    __host__ __device__ 
    bool operator()(const ClusterQueryPair& a, const ClusterQueryPair& b) const {
        return a.cluster_idx < b.cluster_idx;
    }
};

class GPUClusterMeta {
   public:
    size_t num;          // Number of vectors in this cluster.
    size_t iter;         // Number of iterations for FastScan.
    size_t remain;       // Number of leftover vectors after blocks.
    size_t start_index;  // Combined offset: index of first vector in the flattened arrays.

    // Constructor: computes iter and REMAIN based on FAST_SIZE.
    __host__ __device__ GPUClusterMeta(size_t num, size_t start_idx)
      : num(num), start_index(start_idx)
    {
      iter   = num / FAST_SIZE;
      remain = num - iter * FAST_SIZE;
    }

    // default constructor (useful for vector initialize)
    GPUClusterMeta() : num(0), iter(0), remain(0), start_index(0) {}

    // Copy constructor.
    GPUClusterMeta(const GPUClusterMeta& other) = default;

    // Copy assignment.
    GPUClusterMeta& operator=(const GPUClusterMeta& other) = default;

    // Destructor.
    ~GPUClusterMeta() = default;
};


struct ComputeBitwiseKernelParams {
    const ClusterQueryPair* d_sorted_pairs;
    const GPUClusterMeta* d_cluster_meta;
    const float* d_query;
    const float* d_G_k1xSumq;
    const float* d_centroid_distances;
    const uint32_t* d_short_data;
    const float* d_short_factors;

    // 输出候选池
    float* d_candidates_low_dists;
    float* d_candidates_exact_ips;
    float* d_candidates_q_g_adds;
    uint32_t* d_candidates_gids;
    
    int num_queries;
    int num_centroids;
    int D;
    int num_pairs;
    int max_candidates_per_pair;
    int nprobe;
    int topk; // 局部输出 Top-K
};

class IVF_RaBitQ_Raw {
private:
    cublasHandle_t cublas_handle;
    cudaStream_t stream;

    // rabitqlib::ivf::IVF cpu_ivf;

    float* d_centroids = nullptr;
    float* d_rotator = nullptr;     
    uint32_t* d_short_data = nullptr;
    float* d_short_factors = nullptr;

    GPUClusterMeta* d_cluster_meta = nullptr; // 存储每个 cluster 的 num_points 

    // --- 🌟 CPU 内存指针池 (用于 CPU 精排) ---
    const uint8_t* h_long_codes_ref = nullptr;
    const float* h_ex_factors_ref = nullptr;
    const uint32_t* h_pids_ref = nullptr;

    float* h_rotated_queries = nullptr;
    float* h_G_kbxSumq = nullptr;
    float* h_out_low_dists = nullptr;
    float* h_out_exact_ips = nullptr;
    float* h_out_q_g_adds = nullptr;
    uint32_t* h_out_gids = nullptr;

    size_t num_vectors = 0;
    size_t dim = 0;
    size_t num_centroids = 0;
    size_t ex_bits = 0;
    size_t padded_dim = 0;
    int M_candidates_to_cpu = 1000; // 每个 Query 返回给 CPU 细排的数量 (Top-M)

    // ==========================================
    // 🌟 Workspace Memory Pointers (中间变量池)
    // ==========================================
    size_t ws_max_queries = 0;
    size_t ws_max_topk = 0;
    size_t ws_max_nprobe = 0;

    float* ws_d_queries = nullptr;
    float* ws_d_raw_queries = nullptr;
    float* ws_d_centroid_dists = nullptr;
    float* ws_d_centroid_dists_copy = nullptr;
    float* ws_d_q_norms = nullptr;
    float* ws_d_c_norms = nullptr;
    float* ws_d_probe_vals = nullptr;
    int* ws_d_probe_idx = nullptr;
    ClusterQueryPair* ws_d_sorted_pairs = nullptr;
    float* ws_d_G_k1xSumq = nullptr;
    float* ws_d_G_kbxSumq = nullptr;

    // 🌟 1-bit 候选池
    float* ws_d_candidates_low_dists = nullptr;
    float* ws_d_candidates_exact_ips = nullptr;
    float* ws_d_candidates_q_g_adds = nullptr;
    uint32_t* ws_d_candidates_gids = nullptr;

    // 🌟 Top-M 输出池 (传回给 CPU)
    float* ws_d_out_low_dists = nullptr;
    float* ws_d_out_exact_ips = nullptr;
    float* ws_d_out_q_g_adds = nullptr;
    uint32_t* ws_d_out_gids = nullptr;
    int* ws_d_out_indices = nullptr;

    float* ws_d_candidates_low_dists_sorted;
    int* ws_d_out_indices_temp;
    int* ws_d_out_indices_sorted;

    float* ws_d_centroid_dists_sorted = nullptr;
    int* ws_d_centroid_indices_temp = nullptr;
    int* ws_d_centroid_indices_sorted = nullptr;

public:
    IVF_RaBitQ_Raw();
    ~IVF_RaBitQ_Raw();

    size_t max_cluster_size_ = 0;
    float best_rescaling_factor = 0.0f;

    // 🌟 预分配与清理工作区
    void prepare_workspace(size_t max_queries, size_t max_topk, size_t max_nprobe);
    void free_workspace();

    // 🌟 新增：专门用于 CPU 执行 AVX/SIMD 精排的函数 (物理隔离)
    void cpu_exact_rerank(size_t num_queries, 
                          size_t topk, 
                          float* h_final_dists, 
                          uint32_t* h_final_pids);

    // 接收原生 C 指针
    void load_from_raw_pointers(
        size_t num_vecs, size_t d, size_t p_d, size_t n_centroids, size_t ex_b,
        const float* h_centroids,
        const float* h_rotator,
        const uint32_t* h_short_data, 
        const float* h_short_factors,
        const uint8_t* h_long_code,
        const float* h_ex_factors,
        const uint32_t* h_ids,
        const size_t* h_cluster_sizes
    );

    void search(const float* h_queries, 
                size_t num_queries, 
                size_t topk, 
                size_t nprobe, 
                float* h_final_dists, 
                uint32_t* h_final_pids); 
};