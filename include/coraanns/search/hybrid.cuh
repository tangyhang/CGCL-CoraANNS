#pragma once
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "coraanns/queue/candidate_queue.hpp"
#include "coraanns/search/parameters.hpp"

namespace coraanns {

enum class KernelType { Lut4Bit, Bitwise };
namespace detail {
class GPUClusterMeta;
struct ClusterQueryPair;
} // namespace detail

class HybridIndex {
  private:
    cublasHandle_t cublas_handle = nullptr;
    cudaStream_t stream = nullptr;
    cudaStream_t stream_d2h = nullptr;
    cudaStream_t stream_factors = nullptr;

    cudaEvent_t event_rot_done = nullptr;
    cudaEvent_t event_probe_done = nullptr;
    cudaEvent_t event_factors_done = nullptr;

    cudaEvent_t start_event = nullptr, stop_event = nullptr;
    cudaEvent_t filter_start_event = nullptr, filter_stop_event = nullptr;

    float *d_centroids = nullptr;
    float *d_rotator = nullptr;
    uint32_t *d_short_data = nullptr;
    float *d_short_factors = nullptr;
    detail::GPUClusterMeta *d_cluster_meta = nullptr;

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

    size_t max_cluster_size_ = 0;
    size_t ws_vanguard_size_ = 0;
    bool workspace_ready_ = false;
    size_t ws_max_queries = 0;
    size_t ws_max_topk = 0;
    size_t ws_max_nprobe = 0;

    bool profiling_enabled_ = false;
    bool candidate_stats_enabled_ = false;
    bool gpu_filter_timing_enabled_ = false;
    bool cross_query_reuse_enabled_ = true;
    bool cpu_secondary_filter_enabled_ = true;
    size_t vanguard_size_ = default_vanguard_size;

    float *ws_d_queries = nullptr;
    float *ws_d_raw_queries = nullptr;
    float *ws_d_centroid_dists = nullptr;
    float *ws_d_q_norms = nullptr;
    float *ws_d_c_norms = nullptr;
    float *ws_d_probe_vals = nullptr;
    int *ws_d_probe_idx = nullptr;
    detail::ClusterQueryPair *ws_d_sorted_pairs = nullptr;
    float *ws_d_G_k1xSumq = nullptr;
    float *ws_d_G_kbxSumq = nullptr;

    int *ws_d_vanguard_seed_indices = nullptr;
    int *ws_d_vanguard_seed_counts = nullptr;

    half *ws_d_global_luts = nullptr;

    uint32_t *ws_d_q_bits = nullptr;
    float *ws_d_q_scale = nullptr;
    float *ws_d_q_min = nullptr;

    queue::Descriptor *h_global_buffer = nullptr;
    queue::Descriptor d_global_buffer{};

    queue::Position *d_vram_write_count = nullptr;

    queue::Position *h_global_write_count = nullptr;
    int *h_gpu_done = nullptr;
    float *h_dynamic_thresholds = nullptr;
    float *d_dynamic_thresholds = nullptr;

    int *d_global_pair_counter = nullptr;

    size_t queue_capacity_ = queue::default_capacity;
    size_t queue_spill_capacity_ = queue::default_spill_capacity;
    queue::Progress *h_queue_progress = nullptr;
    queue::Position last_written_count = 0;
    queue::Position last_spilled_count_ = 0;
    size_t last_refined_count_ = 0;
    float last_gpu_filter_ms_ = 0.0f;

    KernelType ip_kernel_type_ = KernelType::Lut4Bit;

  public:
    HybridIndex();
    HybridIndex(const HybridIndex &) = delete;
    HybridIndex &operator=(const HybridIndex &) = delete;
    ~HybridIndex();

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

    queue::Position get_last_submitted_count() const {
        return last_written_count;
    }
    queue::Position get_last_spilled_count() const {
        return last_spilled_count_;
    }
    void set_queue_capacity(size_t capacity);
    void set_queue_spill_capacity(size_t capacity);
    size_t get_queue_capacity() const {
        return queue_capacity_;
    }
    size_t get_last_refined_count() const {
        return last_refined_count_;
    }
    float get_last_gpu_filter_ms() const {
        return last_gpu_filter_ms_;
    }

    void set_vanguard_size(size_t size);
    size_t get_vanguard_size() const {
        return vanguard_size_;
    }

    void set_ip_kernel_type(KernelType t) {
        ip_kernel_type_ = t;
    }
    KernelType get_ip_kernel_type() const {
        return ip_kernel_type_;
    }

    void prepare_workspace(size_t max_queries, size_t max_topk, size_t max_nprobe);
    void free_workspace() noexcept;
    void reset_buffer(const float *h_queries, size_t num_queries);

    void enable_exact_refine(const uint8_t *base_mapping, size_t base_rows, size_t base_dim,
                             size_t row_stride_bytes, size_t data_offset_bytes) {
        h_exact_base_mapping = base_mapping;
        exact_base_rows = base_rows;
        exact_base_dim = base_dim;
        exact_base_row_stride_bytes = row_stride_bytes;
        exact_base_data_offset_bytes = data_offset_bytes;
    }

  private:
    void cpu_exact_rerank(const float *h_queries, size_t num_queries, size_t topk, int nprobe,
                          float *h_final_dists, uint32_t *h_final_pids);

  public:
    // Host extended codes, factors and IDs are borrowed until this engine is destroyed.
    void load_from_raw_pointers(size_t num_vecs, size_t d, size_t p_d, size_t n_centroids,
                                size_t ex_b, const float *h_centroids, const float *h_rotator,
                                const uint32_t *h_short_data, const float *h_short_factors,
                                const uint8_t *h_long_code, const float *h_ex_factors,
                                const uint32_t *h_ids, const size_t *h_cluster_sizes);

    void capture_thresholds(float *out, size_t nq) const;
    void seed_thresholds(const float *in, size_t nq);

    void search(const float *h_queries, size_t num_queries, size_t topk, size_t nprobe,
                float *h_final_dists, uint32_t *h_final_pids, int batch_q = default_batch_q);

  private:
    void allocate_candidate_queue();
    void release_candidate_queue() noexcept;
    void release_resources() noexcept;
    void validate_workspace(size_t queries, size_t topk, size_t nprobe) const;
};

} // namespace coraanns
