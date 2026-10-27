#pragma once
#include <string>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <queue>
#include <atomic>
#include <cuda_runtime.h>
#include <cublas_v2.h>

static constexpr int WARP_SIZE = 32;
constexpr size_t FAST_SIZE = 32;
constexpr size_t DEFAULT_VANGUARD_SIZE = 5;

static constexpr int BITS_PER_CHUNK = 4;
static constexpr int LUT_SIZE = (1 << BITS_PER_CHUNK);

constexpr int RABITQ_BQ = 8;

enum class IPKernelType { LUT_4BIT, LUT_4BIT_Q2, BITWISE_DECOMP };

using T = float;
using IdxT = uint32_t;
typedef uint32_t PID;

// Probe scheduling retains vanguard-first ordering, then cluster/probe/query order.
struct ClusterQueryPair {
    int cluster_idx;
    int query_idx;
    int probe_idx;
};

constexpr int BATCH_Q = 4;

struct ClusterBatchTask {
    int cluster_idx;
    int num_active_queries;
    int query_indices[BATCH_Q];
    int probe_indices[BATCH_Q];
};

// Shared candidate payload; preserve its binary layout and publication protocol.
struct alignas(16) GlobalTaskItem {
    uint16_t query_idx;
    uint16_t probe_idx;
    uint32_t gid;
    float exact_ip;
    float low_dist;
};

#define GLOBAL_ASYNC_BUFFER_SIZE 1000000000
struct GlobalAsyncBuffer {
    GlobalTaskItem items[GLOBAL_ASYNC_BUFFER_SIZE];
};

struct PairSortCompare {
    size_t vanguard_size;

    __host__ __device__ explicit PairSortCompare(size_t size = DEFAULT_VANGUARD_SIZE)
        : vanguard_size(size) {}

    __host__ __device__ bool operator()(const ClusterQueryPair &a,
                                        const ClusterQueryPair &b) const {

        bool a_is_vanguard = (static_cast<size_t>(a.probe_idx) < vanguard_size);
        bool b_is_vanguard = (static_cast<size_t>(b.probe_idx) < vanguard_size);

        if (a_is_vanguard != b_is_vanguard) {
            return a_is_vanguard;
        }

        if (a.cluster_idx != b.cluster_idx) {
            return a.cluster_idx < b.cluster_idx;
        }

        if (a.probe_idx != b.probe_idx) {
            return a.probe_idx < b.probe_idx;
        }

        return a.query_idx < b.query_idx;
    }
};

class GPUClusterMeta {
  public:
    size_t num;
    size_t iter;
    size_t remain;
    size_t start_index;

    __host__ __device__ GPUClusterMeta(size_t num, size_t start_idx)
        : num(num), start_index(start_idx) {
        iter = num / FAST_SIZE;
        remain = num - iter * FAST_SIZE;
    }
    GPUClusterMeta() : num(0), iter(0), remain(0), start_index(0) {}
    GPUClusterMeta(const GPUClusterMeta &other) = default;
    GPUClusterMeta &operator=(const GPUClusterMeta &other) = default;
    ~GPUClusterMeta() = default;
};

struct ComputeBitwiseKernelParams {
    const ClusterQueryPair *d_sorted_pairs;
    const GPUClusterMeta *d_cluster_meta;
    const float *d_query;
    const float *d_G_k1xSumq;
    const float *d_centroid_distances;
    const uint32_t *d_short_data;
    const uint4 *d_short_data_u4;
    int num_chunks;
    int num_groups;
    const float *d_short_factors;
    const half *d_global_luts;

    const uint32_t *d_q_bits;
    const float *d_q_scale;
    const float *d_q_min;

    int num_queries;
    int num_centroids;
    int D;
    int max_candidates_per_pair;
    int num_pairs;
    int topk;
};

// Production hybrid engine. Historical type and member names are kept for compatibility.
class IVF_RaBitQ_Raw {
  private:
    cublasHandle_t cublas_handle;
    cudaStream_t stream;
    cudaStream_t stream_d2h;
    cudaStream_t stream_factors;

    cudaEvent_t event_rot_done;
    cudaEvent_t event_probe_done;
    cudaEvent_t event_factors_done;

    cudaEvent_t start_event, stop_event;
    cudaEvent_t filter_start_event, filter_stop_event;

    float *d_centroids = nullptr;
    float *d_rotator = nullptr;
    uint32_t *d_short_data = nullptr;
    uint4 *d_short_data_u4 = nullptr;
    float *d_short_factors = nullptr;
    GPUClusterMeta *d_cluster_meta = nullptr;

    const uint8_t *h_long_codes_ref = nullptr;
    const float *h_ex_factors_ref = nullptr;
    const uint32_t *h_pids_ref = nullptr;

    const uint8_t *h_exact_base_mapping = nullptr;
    size_t exact_base_rows = 0;
    size_t exact_base_dim = 0;
    size_t exact_base_row_stride_bytes = 0;
    size_t exact_base_data_offset_bytes = 0;

    float *h_rotated_queries = nullptr;
    float *h_G_kbxSumq = nullptr;

    float *h_probe_vals = nullptr;

    size_t num_vectors = 0;
    size_t dim = 0;
    size_t num_centroids = 0;
    size_t ex_bits = 0;
    size_t padded_dim = 0;

    size_t ws_max_queries = 0;
    size_t ws_max_topk = 0;
    size_t ws_max_nprobe = 0;

    bool profiling_enabled_ = false;
    bool candidate_stats_enabled_ = false;
    bool gpu_filter_timing_enabled_ = false;
    bool cross_query_reuse_enabled_ = true;
    bool cpu_secondary_filter_enabled_ = true;
    size_t vanguard_size_ = DEFAULT_VANGUARD_SIZE;

    float *ws_d_queries = nullptr;
    float *ws_d_raw_queries = nullptr;
    float *ws_d_centroid_dists = nullptr;
    float *ws_d_centroid_dists_copy = nullptr;
    float *ws_d_q_norms = nullptr;
    float *ws_d_c_norms = nullptr;
    float *ws_d_probe_vals = nullptr;
    int *ws_d_probe_idx = nullptr;
    ClusterQueryPair *ws_d_sorted_pairs = nullptr;
    float *ws_d_G_k1xSumq = nullptr;
    float *ws_d_G_kbxSumq = nullptr;

    float *ws_d_centroid_dists_sorted = nullptr;
    int *ws_d_centroid_indices_temp = nullptr;
    int *ws_d_centroid_indices_sorted = nullptr;

    int *ws_d_vanguard_seed_indices = nullptr;
    int *ws_d_vanguard_seed_counts = nullptr;

    ClusterBatchTask *ws_d_batched_tasks = nullptr;
    half *ws_d_global_luts = nullptr;

    uint32_t *ws_d_q_bits = nullptr;
    float *ws_d_q_scale = nullptr;
    float *ws_d_q_min = nullptr;

    GlobalAsyncBuffer *h_global_buffer = nullptr;
    GlobalAsyncBuffer *d_global_buffer = nullptr;

    int *d_vram_write_count = nullptr;

    int *h_global_write_count = nullptr;
    int *d_global_write_count = nullptr;
    int *h_gpu_done = nullptr;
    int *d_gpu_done = nullptr;
    float *h_dynamic_thresholds = nullptr;
    float *d_dynamic_thresholds = nullptr;

    int *d_global_pair_counter = nullptr;

    int last_written_count = 0;
    size_t last_refined_count_ = 0;
    float last_gpu_filter_ms_ = 0.0f;

    IPKernelType ip_kernel_type_ = IPKernelType::LUT_4BIT;

  public:
    IVF_RaBitQ_Raw();
    ~IVF_RaBitQ_Raw();

    size_t max_cluster_size_ = 0;
    float best_rescaling_factor = 0.0f;

    void set_profiling(bool enabled) {
        profiling_enabled_ = enabled;
    }
    void set_candidate_stats(bool enabled) {
        candidate_stats_enabled_ = enabled;
    }
    void set_gpu_filter_timing(bool enabled) {
        gpu_filter_timing_enabled_ = enabled;
    }
    void set_cross_query_reuse(bool enabled) {
        cross_query_reuse_enabled_ = enabled;
    }
    void set_cpu_secondary_filter(bool enabled) {
        cpu_secondary_filter_enabled_ = enabled;
    }

    int get_last_submitted_count() const {
        return last_written_count;
    }
    size_t get_last_refined_count() const {
        return last_refined_count_;
    }
    float get_last_gpu_filter_ms() const {
        return last_gpu_filter_ms_;
    }

    void set_vanguard_size(size_t size) {
        vanguard_size_ = size;
    }
    size_t get_vanguard_size() const {
        return vanguard_size_;
    }

    void set_ip_kernel_type(IPKernelType t) {
        ip_kernel_type_ = t;
    }
    IPKernelType get_ip_kernel_type() const {
        return ip_kernel_type_;
    }

    void prepare_workspace(size_t max_queries, size_t max_topk, size_t max_nprobe);
    void free_workspace();
    void reset_buffer(const float *h_queries, size_t num_queries);

    void enable_exact_refine(const uint8_t *base_mapping, size_t base_rows, size_t base_dim,
                             size_t row_stride_bytes, size_t data_offset_bytes) {
        h_exact_base_mapping = base_mapping;
        exact_base_rows = base_rows;
        exact_base_dim = base_dim;
        exact_base_row_stride_bytes = row_stride_bytes;
        exact_base_data_offset_bytes = data_offset_bytes;
    }

    void cpu_exact_rerank(const float *h_queries, size_t num_queries, size_t topk, int nprobe,
                          float *h_final_dists, uint32_t *h_final_pids);

    void load_from_raw_pointers(size_t num_vecs, size_t d, size_t p_d, size_t n_centroids,
                                size_t ex_b, const float *h_centroids, const float *h_rotator,
                                const uint32_t *h_short_data, const float *h_short_factors,
                                const uint8_t *h_long_code, const float *h_ex_factors,
                                const uint32_t *h_ids, const size_t *h_cluster_sizes);

    void capture_thresholds(float *out, size_t nq) const;
    void seed_thresholds(const float *in, size_t nq);

    void search(const float *h_queries, size_t num_queries, size_t topk, size_t nprobe,
                float *h_final_dists, uint32_t *h_final_pids, int batch_q = BATCH_Q);
};
