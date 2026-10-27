#pragma once
#include <string>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <queue>
#include <atomic>
#include <chrono>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static constexpr int WARP_SIZE = 32;
constexpr size_t FAST_SIZE = 32;
constexpr size_t VANGUARD_SIZE = 0;

static constexpr int BITS_PER_CHUNK = 4;
static constexpr int LUT_SIZE       = (1 << BITS_PER_CHUNK);  // 16

// ============================================================================
//  Bitwise Decomposition 策略参数 (Section 3.4.1 (2) of IVF-RaBitQ paper)
// ----------------------------------------------------------------------------
//  RABITQ_BQ = 每个 query 维度的量化比特数。
//    - Bq=4: 量化误差 ~1/7，速度最快，但高维场景精度可能不够
//    - Bq=8: 量化误差 ~1/127（精度提升 ~18×），shared memory 仍远小于 LUT 版
//            推荐用于 D >= 512 或对召回敏感的场景
//  如需更高精度也可以调到更大的值（最大 31），shared memory 线性增加。
// ============================================================================
constexpr int RABITQ_BQ = 8;

// ============================================================================
//  内积 Kernel 选择：1-bit 粗筛阶段用哪种实现
// ----------------------------------------------------------------------------
//   - LUT_4BIT       : 4-bit 查表，pair-major（1 block 处理 1 个 pair）
//   - LUT_4BIT_Q2    : 🌟 Q=2 batched pair-major：1 block 抢 2 个连续 pair，
//                      若同簇 → chunk 在寄存器里复用，1 次 chunk load 摊到
//                      2 个 query 的 16 次 LUT lookup，算术强度翻倍。
//                      跨簇时退化成串行处理 2 个 pair（性能与 LUT_4BIT 相当）。
//   - BITWISE_DECOMP : 论文 Section 3.4.1 (2) 的位运算分解
//
//  可在两次 search() 之间随时通过 set_ip_kernel_type() 切换，无需 reset workspace。
// ============================================================================
enum class IPKernelType {
    LUT_4BIT,
    LUT_4BIT_Q2,
    BITWISE_DECOMP
};

using T = float;
using IdxT = uint32_t;
typedef uint32_t PID;

struct ClusterQueryPair {
    int cluster_idx;
    int query_idx;
    int probe_idx;
};

// 🌟 定义每个 Block 同时处理的 Query 数量 (推荐 2, 3 或 4)
constexpr int BATCH_Q = 4;

struct ClusterBatchTask {
    int cluster_idx;
    int num_active_queries;          // 本次打包了几个 Query (1 ~ BATCH_Q)
    int query_indices[BATCH_Q];      // 打包的 Query ID
    int probe_indices[BATCH_Q];      // 对应的 probe 排名
};

// 🌟 1. 结构体完美压缩为 16 字节，并强制 16 字节对齐
struct alignas(16) GlobalTaskItem {
    uint16_t query_idx;
    uint16_t probe_idx;
    uint32_t gid;
    float exact_ip;
    float low_dist;
};

// 🌟 唯一全局任务队列 (100000000容量，绝对防爆仓)
#define GLOBAL_ASYNC_BUFFER_SIZE 1000000000 
struct GlobalAsyncBuffer {
    GlobalTaskItem items[GLOBAL_ASYNC_BUFFER_SIZE]; 
};

// 🌟 自定义 Thrust 排序规则 (Best-First 核心：按探针深度优先！)
// struct PairSortCompare {
//     __host__ __device__ 
//     bool operator()(const ClusterQueryPair& a, const ClusterQueryPair& b) const {
//         if (a.probe_idx != b.probe_idx) return a.probe_idx < b.probe_idx;
//         return a.cluster_idx < b.cluster_idx;
//     }
// };

// 自定义 Thrust 排序规则 (按 cluster_idx 从小到大排)
// struct PairSortCompare {
//     __host__ __device__ 
//     bool operator()(const ClusterQueryPair& a, const ClusterQueryPair& b) const {
//         return a.cluster_idx < b.cluster_idx;
//         // if (a.cluster_idx != b.cluster_idx) {
//         //         return a.cluster_idx < b.cluster_idx;
//         //     }

//         //     // 4. 如果连 cluster 都一样，按 probe 深度排序
//         //     if (a.probe_idx != b.probe_idx) {
//         //         return a.probe_idx < b.probe_idx;
//         //     }

//         //     // 5. 最后按 query_idx 排序，保证底层的 Strict Weak Ordering，防止 Thrust 排序器崩溃
//         //     return a.query_idx < b.query_idx;
//         // }
// };

// 🌟 终极混合调度规则：先遣队破点降阈，大部队按簇平推压榨 L2 缓存！
struct PairSortCompare {
    size_t vanguard_size;
    __host__ __device__
    explicit PairSortCompare(size_t size = VANGUARD_SIZE) : vanguard_size(size) {}
    __host__ __device__ 
    bool operator()(const ClusterQueryPair& a, const ClusterQueryPair& b) const {
        // 1. 身份鉴定：是否为 Top-5 聚类 (先遣队)
        bool a_is_top5 = (static_cast<size_t>(a.probe_idx) < vanguard_size);
        bool b_is_top5 = (static_cast<size_t>(b.probe_idx) < vanguard_size);

        // 2. 战略最高优先级：先遣队无条件排在最前面！
        if (a_is_top5 != b_is_top5) {
            return a_is_top5; // 如果 a 是先遣队而 b 不是，a 排前面
        }

        // 3. 在同一梯队内 (同为先遣队，或同为大部队)，严格按照 cluster_idx 排序，打满 GPU L2 Cache！
        if (a.cluster_idx != b.cluster_idx) {
            return a.cluster_idx < b.cluster_idx;
        }

        // 4. 如果连 cluster 都一样，按 probe 深度排序
        if (a.probe_idx != b.probe_idx) {
            return a.probe_idx < b.probe_idx;
        }

        // 5. 最后按 query_idx 排序，保证底层的 Strict Weak Ordering，防止 Thrust 排序器崩溃
        return a.query_idx < b.query_idx;
    }
};

class GPUClusterMeta {
public:
    size_t num;
    size_t iter;
    size_t remain;
    size_t start_index;

    __host__ __device__ GPUClusterMeta(size_t num, size_t start_idx) : num(num), start_index(start_idx) {
        iter   = num / FAST_SIZE;
        remain = num - iter * FAST_SIZE;
    }
    GPUClusterMeta() : num(0), iter(0), remain(0), start_index(0) {}
    GPUClusterMeta(const GPUClusterMeta& other) = default;
    GPUClusterMeta& operator=(const GPUClusterMeta& other) = default;
    ~GPUClusterMeta() = default;
};

// ============================================================================
//  Bitwise Decomposition 版的 Kernel 传参结构体
// ----------------------------------------------------------------------------
//  相比旧版（LUT），新增三个 query bit-plane 相关字段：
//      d_q_bits  : (NQ, Bq, D/32) 量化后打包的 query bit-planes
//      d_q_scale : (NQ,)  per-query 量化步长 Δq
//      d_q_min   : (NQ,)  per-query 量化下界 q_min
// ============================================================================
struct ComputeBitwiseKernelParams {
    const ClusterQueryPair* d_sorted_pairs;
    const GPUClusterMeta*   d_cluster_meta;
    const float*            d_query;            // 保留：调试/兼容用，主路径不再依赖
    const float*            d_G_k1xSumq;
    const float*            d_centroid_distances;
    const uint32_t*         d_short_data;
    const uint4* d_short_data_u4;   // 🌟 uint4 优化布局指针
    int          num_chunks;        // = D / 32
    int          num_groups;        // = ceil(num_chunks / 4)
    const float*            d_short_factors;
    const half*             d_global_luts;

    // === Bitwise 专用 ===
    const uint32_t*         d_q_bits;           // (NQ, Bq, D/32) packed query bit-planes
    const float*            d_q_scale;          // (NQ,)  Δq
    const float*            d_q_min;            // (NQ,)  q_min

    int num_queries;
    int num_centroids;
    int D;
    int max_candidates_per_pair;
    int num_pairs;
    int topk;
};

class IVF_RaBitQ_Raw {
private:
    cublasHandle_t cublas_handle;
    cudaStream_t stream;
    cudaStream_t stream_d2h;     // 传输专线
    cudaStream_t stream_factors; // 🌟 辅计算专线
    
    cudaEvent_t event_rot_done;
    cudaEvent_t event_probe_done;
    cudaEvent_t event_factors_done;
    
    cudaEvent_t start_event, stop_event; // 计时器

    float* d_centroids = nullptr;
    float* d_rotator = nullptr;     
    uint32_t* d_short_data = nullptr;
    uint4*    d_short_data_u4 = nullptr;   // 🌟 uint4 优化布局（LUT_4BIT 专用）
    float* d_short_factors = nullptr;
    GPUClusterMeta* d_cluster_meta = nullptr;

    const uint8_t* h_long_codes_ref = nullptr;
    const float* h_ex_factors_ref = nullptr;
    const uint32_t* h_pids_ref = nullptr;

    float* h_rotated_queries = nullptr;
    float* h_G_kbxSumq = nullptr;

    float* h_probe_vals = nullptr;

    size_t num_vectors = 0;
    size_t dim = 0;
    size_t num_centroids = 0;
    size_t ex_bits = 0;
    size_t padded_dim = 0;

    size_t ws_max_queries = 0;
    size_t ws_max_topk = 0;
    size_t ws_max_nprobe = 0;
    size_t vanguard_size_ = VANGUARD_SIZE;
    bool cpu_secondary_filter_enabled_ = true;
    bool profiling_enabled_ = false;

    // 内部计算显存池
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

    float* ws_d_centroid_dists_sorted = nullptr;
    int* ws_d_centroid_indices_temp = nullptr;
    int* ws_d_centroid_indices_sorted = nullptr;

    ClusterBatchTask* ws_d_batched_tasks = nullptr; // 新增的任务队列
    half* ws_d_global_luts = nullptr;

    // === 🌟 Bitwise Decomposition 专用工作区 ===
    uint32_t* ws_d_q_bits  = nullptr;   // (max_queries, Bq, D/32)
    float*    ws_d_q_scale = nullptr;   // (max_queries,)  Δq
    float*    ws_d_q_min   = nullptr;   // (max_queries,)  q_min

    // --- 🌟 异构协同专属指针 ---
    GlobalAsyncBuffer* h_global_buffer = nullptr; 
    GlobalAsyncBuffer* d_global_buffer = nullptr;

    int* d_vram_write_count = nullptr;

    int* h_global_write_count = nullptr; 
    int* d_global_write_count = nullptr; 
    int* h_gpu_done = nullptr;
    int* d_gpu_done = nullptr;

    // Backlog trace: zero-copy progress counter updated by GPU during producer kernels.
    // It is separate from h_global_write_count because the latter is finalized only when
    // the GPU-done flag kernel runs.
    unsigned int* h_gpu_progress_count = nullptr;
    unsigned int* d_gpu_progress_count = nullptr;

    // Backlog trace: live CPU completion counter. It counts tasks actually handled by CPU,
    // not tasks pre-claimed by chunk stealing.
    std::atomic<size_t> trace_cpu_processed_live{0};
    bool trace_enabled_ = true;
    std::string trace_output_file_ = "backlog_trace.csv";
    std::chrono::steady_clock::time_point trace_epoch_;
    double trace_cpu_finished_ms_ = 0.0;

    float* h_dynamic_thresholds = nullptr;
    float* d_dynamic_thresholds = nullptr;

    int* d_global_pair_counter = nullptr;

    int last_written_count = 0; // 记录上一轮队列使用长度，极速清零哨兵

    // === 🌟 内积 Kernel 选择 ===
    //  默认走 LUT_4BIT（你之前实测最快的路径）；如需对比可调用 set_ip_kernel_type()
    IPKernelType ip_kernel_type_ = IPKernelType::LUT_4BIT;

public:
    IVF_RaBitQ_Raw();
    ~IVF_RaBitQ_Raw();

    size_t max_cluster_size_ = 0;
    float best_rescaling_factor = 0.0f;
    void set_vanguard_size(size_t size) { vanguard_size_ = size; }
    void set_cpu_secondary_filter(bool enabled) { cpu_secondary_filter_enabled_ = enabled; }
    void set_profiling(bool enabled) { profiling_enabled_ = enabled; }
    void set_trace_enabled(bool enabled) { trace_enabled_ = enabled; }
    void set_trace_output(const std::string& path) { trace_output_file_ = path; }

    // 运行时切换内积 kernel；可在两次 search() 之间任意调用
    void set_ip_kernel_type(IPKernelType t) { ip_kernel_type_ = t; }
    IPKernelType get_ip_kernel_type() const { return ip_kernel_type_; }

    void prepare_workspace(size_t max_queries, size_t max_topk, size_t max_nprobe);
    void free_workspace();
    void reset_buffer(const float* h_queries, size_t num_queries);

    // 纯粹的雇佣兵模式 CPU 精排函数
    void cpu_exact_rerank(size_t num_queries, size_t topk, int nprobe, float* h_final_dists, uint32_t* h_final_pids);

    void load_from_raw_pointers(
        size_t num_vecs, size_t d, size_t p_d, size_t n_centroids, size_t ex_b,
        const float* h_centroids, const float* h_rotator,
        const uint32_t* h_short_data, const float* h_short_factors,
        const uint8_t* h_long_code, const float* h_ex_factors,
        const uint32_t* h_ids, const size_t* h_cluster_sizes);

    void capture_thresholds(float* out, size_t nq) const;   // 读出收敛后的阈值(= 真实第 k 近)
    void seed_thresholds(const float* in, size_t nq);        // kernel 启动前预置种子

    void search(const float* h_queries, size_t num_queries, size_t topk, size_t nprobe, float* h_final_dists, uint32_t* h_final_pids, int batch_q = BATCH_Q);
};
