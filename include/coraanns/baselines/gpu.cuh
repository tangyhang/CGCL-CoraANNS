#pragma once
#include <string>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// Keep the original 512-entry build. The optional high-k target only expands
// the tiled fine-search merge capacity; it does not change benchmark timing.
#ifndef RABITQ_GPU_FINE_CAPACITY
#define RABITQ_GPU_FINE_CAPACITY 512
#endif
static_assert(RABITQ_GPU_FINE_CAPACITY == 512 ||
              RABITQ_GPU_FINE_CAPACITY == 1024,
              "Supported GPU fine-search capacities are 512 and 1024");

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


// =========================================================
// Bitwise Kernel 参数结构体
// =========================================================
struct ComputeBitwiseKernelParams {
    const ClusterQueryPair* d_sorted_pairs;
    const GPUClusterMeta* d_cluster_meta; // 注意替换为你实际的 Meta 结构体名
    const half* d_global_luts;
    
    // Query 相关
    const float* d_query;
    const uint32_t* d_packed_queries; // 4-bit 打包后的 Query 位平面
    const float* d_widths;            // Query 量化的缩放宽度
    const float* d_G_k1xSumq;         // Query 的预计算因子
    const float* d_centroid_distances;// 步骤1中算出的 Query 到 Centroid 距离
    float* d_threshold; // 每个 Query 的当前 topk 距离阈值 (用于动态剪枝)
    
    // 底库相关
    const uint32_t* d_short_data;     // 1-bit 底库数据
    const float* d_short_factors;     // 底库的还原 Factors (SoA 布局)
    const uint32_t* d_pids;           // 原始 ID

    const float* d_G_kbxSumq;       // 新增
    const uint8_t* d_long_code;     // 新增
    const float* d_ex_factor;       // 新增
    size_t ex_bits;                 // 新增
    const uint8_t* d_cached_long_code = nullptr;
    const float* d_cached_ex_factor = nullptr;
    size_t cached_vectors = 0;
    
    // 输出相关
    float* d_candidates_dists;
    uint32_t* d_candidates_pids;
    int* d_query_write_counters;      // 用于原子累加分配输出位置
    
    unsigned long long* d_phase_cycles;   // [0]=scan, [1]=rerank, [2]=sort
                                           // [3]=candidates, [4]=refinements,
                                           // [5]=scanned vectors
    
    // 维度与控制参数
    int num_queries;
    int num_centroids;
    int D;                            // 向量维度 (Padded)
    int num_bits;                     // = 4
    int num_words;                    // D / 32
    int num_pairs;                    // nprobe * num_queries
    int max_candidates_per_query;     // 每个 Query 最多输出多少个候选 (通常是 nprobe * max_cluster_size)
    int nprobe;                        // 扫描多少个 Cluster
    int topk;                          // 最终返回多少个近邻
    int max_candidates_per_pair;     // max storage per pair, 1000 suggested
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
    uint8_t* d_long_codes = nullptr;   
    float* d_ex_factors = nullptr;

    // In host-backed mode the extended codes and their factors are CUDA
    // device aliases for registered CPU memory rather than cudaMalloc-owned
    // allocations. Keeping both pieces of extended data on the host is what
    // makes the GPU-native baseline fit the 1B-vector index on a 32 GiB GPU.
    bool long_codes_are_host_mapped_ = false;
    const uint8_t* h_registered_long_codes_ = nullptr;
    bool ex_factors_are_host_mapped_ = false;
    const float* h_registered_ex_factors_ = nullptr;
    uint8_t* d_cached_long_codes_ = nullptr;
    float* d_cached_ex_factors_ = nullptr;
    size_t cached_vectors_ = 0;

    uint32_t* d_ids = nullptr;
    GPUClusterMeta* d_cluster_meta = nullptr; // 存储每个 cluster 的 num_points 

    size_t num_vectors = 0;
    size_t dim = 0;
    size_t num_centroids = 0;
    size_t ex_bits = 0;
    size_t padded_dim = 0;

    // ==========================================
    // 🌟 Workspace Memory Pointers (中间变量池)
    // ==========================================
    size_t ws_max_queries = 0;
    size_t ws_max_topk = 0;
    size_t ws_max_nprobe = 0;

    bool profiling_enabled_ = false;
    bool force_tiled_fine_kernel_ = false;
    bool isolated_profiling_enabled_ = false;
    size_t isolated_profile_rounds_ = 5;
    unsigned long long last_scanned_vectors_ = 0;
    unsigned long long last_refined_vectors_ = 0;

    float* ws_d_queries = nullptr;
    float* ws_d_raw_queries = nullptr;
    float* ws_d_centroid_dists = nullptr;
    float* ws_d_q_norms = nullptr;
    float* ws_d_c_norms = nullptr;
    float* ws_d_probe_vals = nullptr;
    int* ws_d_probe_idx = nullptr;
    ClusterQueryPair* ws_d_sorted_pairs = nullptr;
    float* ws_d_G_k1xSumq = nullptr;
    float* ws_d_G_kbxSumq = nullptr;
    float* ws_d_candidates_dists = nullptr;
    uint32_t* ws_d_candidates_pids = nullptr;
    int* ws_d_query_write_counters = nullptr;
    float* ws_d_final_dev_dists = nullptr;
    uint32_t* ws_d_final_dev_pids = nullptr;
    float* ws_d_query_ranges = nullptr;
    float* ws_d_widths = nullptr;
    int8_t* ws_d_quantized_queries = nullptr;
    uint32_t* ws_d_packed_queries = nullptr;
    float* ws_d_threshold = nullptr;

    half* ws_d_global_luts = nullptr;

public:
    IVF_RaBitQ_Raw();
    ~IVF_RaBitQ_Raw();

    size_t max_cluster_size_ = 0;
    float best_rescaling_factor = 0.0f;

    void set_profiling(bool enabled) { profiling_enabled_ = enabled; }
    void set_force_tiled_fine_kernel(bool enabled) {
        force_tiled_fine_kernel_ = enabled;
    }
    void set_isolated_profiling(bool enabled, size_t rounds = 5) {
        isolated_profiling_enabled_ = enabled;
        isolated_profile_rounds_ = rounds;
    }

    // Runs filtering and cluster-local top-k control microbenchmarks against
    // the most recent search state.  It deliberately does not replay
    // refinement: a synthetic ex-code access order is not a production-stage
    // wall time. Call after search() so it does not contaminate [Run] QPS.
    void profile_isolated_last_search(
        size_t num_queries, size_t topk, size_t nprobe);

    // 🌟 预分配与清理工作区
    void prepare_workspace(size_t max_queries, size_t max_topk, size_t max_nprobe);
    void free_workspace();
    void reset_buffer(const float* h_queries, size_t num_queries);
    // Called after workspace allocation; cache a fixed, query-independent
    // prefix of flattened vectors, leaving reserve_bytes for temporary work.
    void configure_hbm_cache(
        size_t reserve_bytes, size_t max_cache_bytes = static_cast<size_t>(-1));
    size_t cached_vectors() const { return cached_vectors_; }
    size_t cached_bytes() const {
        return cached_vectors_ * (((padded_dim * ex_bits + 7) / 8) + 2 * sizeof(float));
    }

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
        const size_t* h_cluster_sizes,
        bool host_backed_ex_codes = false
    );

    void search(const float* h_queries, 
                size_t num_queries, 
                size_t topk, 
                size_t nprobe, 
                float* h_final_dists, 
                uint32_t* h_final_pids,
                int* h_selected_cluster_ids = nullptr);
};
