#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cfloat>
#include <cub/cub.cuh>
#include <omp.h>
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

#include "kernel_types.cuh"
#include "query_kernels.cuh"
#include "filter_kernels.cuh"
#include "cuda_error.hpp"

namespace coraanns {
using namespace detail;

#define CUDA_CHECK(call) ::coraanns::detail::check_cuda((call), #call)
#define CORAANNS_CUBLAS_CHECK(call) ::coraanns::detail::check_cublas((call), #call)

HybridIndex::HybridIndex() {
    try {
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
        CORAANNS_CUBLAS_CHECK(cublasCreate(&cublas_handle));
        CORAANNS_CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
    } catch (...) {
        release_resources();
        throw;
    }
}

HybridIndex::~HybridIndex() {
    release_resources();
}

void HybridIndex::release_resources() noexcept {
    free_workspace();
    if (d_centroids)
        cudaFree(d_centroids);
    if (d_rotator)
        cudaFree(d_rotator);
    if (d_short_data)
        cudaFree(d_short_data);
    if (d_short_factors)
        cudaFree(d_short_factors);
    if (d_cluster_meta)
        cudaFree(d_cluster_meta);
    if (cublas_handle)
        cublasDestroy(cublas_handle);
    if (stream)
        cudaStreamDestroy(stream);
    if (stream_d2h)
        cudaStreamDestroy(stream_d2h);
    if (stream_factors)
        cudaStreamDestroy(stream_factors);
    for (auto event : {event_rot_done, event_probe_done, event_factors_done, start_event,
                       stop_event, filter_start_event, filter_stop_event})
        if (event)
            cudaEventDestroy(event);
}

void HybridIndex::set_vanguard_size(size_t size) {
    if (workspace_ready_ && std::min(size, ws_max_nprobe) > ws_vanguard_size_)
        throw std::logic_error("prepare workspace again before increasing vanguard size");
    vanguard_size_ = size;
}

void HybridIndex::validate_workspace(size_t queries, size_t topk, size_t nprobe) const {
    if (!workspace_ready_)
        throw std::logic_error("prepare workspace before searching");
    if (queries == 0 || queries > ws_max_queries || topk == 0 || topk > ws_max_topk ||
        nprobe == 0 || nprobe > ws_max_nprobe)
        throw std::invalid_argument("search shape exceeds the prepared workspace");
}

void HybridIndex::load_from_raw_pointers(size_t num_vecs, size_t d, size_t p_d, size_t n_centroids,
                                         size_t ex_b, const float *h_centroids,
                                         const float *h_rotator, const uint32_t *h_short_data,
                                         const float *h_short_factors, const uint8_t *h_long_codes,
                                         const float *h_ex_factors, const uint32_t *h_ids,
                                         const size_t *h_cluster_sizes) {
    if (d_cluster_meta)
        throw std::logic_error("index already loaded");
    if (!num_vecs || num_vecs > UINT32_MAX || !d || p_d < d || p_d % 64 || !n_centroids ||
        ex_b > 8 || !h_centroids || !h_rotator || !h_short_data || !h_short_factors || !h_ids ||
        !h_cluster_sizes)
        throw std::invalid_argument("invalid index shape or missing index data");
    size_t count = 0, largest_cluster = 0;
    for (size_t c = 0; c < n_centroids; ++c) {
        if (h_cluster_sizes[c] > num_vecs - count)
            throw std::invalid_argument("cluster sizes exceed index size");
        count += h_cluster_sizes[c];
        largest_cluster = std::max(largest_cluster, h_cluster_sizes[c]);
    }
    if (count != num_vecs)
        throw std::invalid_argument("cluster sizes differ from index size");

    max_cluster_size_ = largest_cluster;
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

void HybridIndex::set_queue_capacity(size_t capacity) {
    if (capacity == 0 || (capacity & (capacity - 1)) != 0 ||
        capacity > SIZE_MAX / sizeof(GlobalTaskItem)) {
        throw std::invalid_argument("queue-capacity must be a positive power of two");
    }
    if (h_global_buffer) {
        throw std::logic_error("set queue capacity before prepare_workspace");
    }
    queue_capacity_ = capacity;
}

void HybridIndex::set_queue_spill_capacity(size_t capacity) {
    if (capacity > SIZE_MAX / sizeof(SpillTaskItem)) {
        throw std::invalid_argument("queue-spill-capacity is too large");
    }
    if (h_global_buffer) {
        throw std::logic_error("set spill capacity before prepare_workspace");
    }
    queue_spill_capacity_ = capacity;
}

void HybridIndex::allocate_candidate_queue() {
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

void HybridIndex::release_candidate_queue() noexcept {
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

void HybridIndex::prepare_workspace(size_t max_queries, size_t max_topk, size_t max_nprobe) {
    if (!d_short_data)
        throw std::logic_error("load an index before preparing workspace");
    validate_search_shape(max_queries, max_topk, max_nprobe, num_centroids, padded_dim);
    free_workspace();
    ws_max_queries = max_queries;
    ws_max_topk = max_topk;
    ws_max_nprobe = max_nprobe;

    CUDA_CHECK(cudaHostAlloc(&h_rotated_queries, max_queries * padded_dim * sizeof(float),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&h_G_kbxSumq, max_queries * sizeof(float), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void **)&h_probe_vals, max_queries * max_nprobe * sizeof(float),
                             cudaHostAllocDefault));
    allocate_candidate_queue();

    CUDA_CHECK(cudaHostAlloc((void **)&h_dynamic_thresholds, max_queries * sizeof(float),
                             cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&d_dynamic_thresholds, h_dynamic_thresholds, 0));

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
    ws_vanguard_size_ = max_vanguard_depth;
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
    workspace_ready_ = true;
}

void HybridIndex::free_workspace() noexcept {
    workspace_ready_ = false;
    ws_max_queries = ws_max_topk = ws_max_nprobe = ws_vanguard_size_ = 0;
    if (ws_d_queries)
        cudaFree(ws_d_queries);
    if (ws_d_raw_queries)
        cudaFree(ws_d_raw_queries);
    if (ws_d_centroid_dists)
        cudaFree(ws_d_centroid_dists);
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
    ws_d_global_luts = nullptr;
    d_vram_write_count = nullptr;
    d_global_pair_counter = nullptr;
    h_rotated_queries = nullptr;
    h_G_kbxSumq = nullptr;
    h_dynamic_thresholds = nullptr;
    d_dynamic_thresholds = nullptr;
    h_probe_vals = nullptr;
}

void HybridIndex::reset_buffer(const float *h_queries, size_t num_queries) {
    validate_workspace(num_queries, 1, 1);
    if (!h_queries)
        throw std::invalid_argument("missing queries");

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

void HybridIndex::search(const float *h_queries, size_t num_queries, size_t topk, size_t nprobe,
                         float *h_final_dists, uint32_t *h_final_pids, int batch_q) {
    validate_workspace(num_queries, topk, nprobe);
    if (!h_queries || !h_final_dists || !h_final_pids || batch_q < 1 || batch_q > 6)
        throw std::invalid_argument("invalid search buffers or batch-q (expected 1..6)");
    if (!h_exact_base_mapping && (!h_long_codes_ref || !h_ex_factors_ref))
        throw std::logic_error("extended codes or an exact-refine base are required");

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
    CORAANNS_CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, padded_dim,
                                      num_queries, dim, &alpha_rot, d_rotator, padded_dim,
                                      d_raw_queries, dim, &beta_rot, d_queries, padded_dim));

    CUDA_CHECK(cudaEventRecord(event_rot_done, stream));

    CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, event_rot_done, 0));
    CUDA_CHECK(cudaMemcpyAsync(h_rotated_queries, d_queries,
                               num_queries * padded_dim * sizeof(float), cudaMemcpyDeviceToHost,
                               stream_d2h));

    CUDA_CHECK(cudaMemsetAsync(d_global_pair_counter, 0, sizeof(int), stream));

    CUDA_CHECK(cudaStreamWaitEvent(stream_factors, event_rot_done, 0));
    computeQueryFactors<float>(d_queries, d_G_k1xSumq, d_G_kbxSumq, num_queries, padded_dim,
                               ex_bits, stream_factors);

    if (ip_kernel_type_ == KernelType::Bitwise) {
        quantize_query_bitwise(d_queries, num_queries, padded_dim, ws_d_q_bits, ws_d_q_scale,
                               ws_d_q_min, stream_factors);
    }

    CUDA_CHECK(cudaEventRecord(event_factors_done, stream_factors));
    CUDA_CHECK(cudaStreamWaitEvent(stream_d2h, event_factors_done, 0));
    CUDA_CHECK(cudaMemcpyAsync(h_G_kbxSumq, d_G_kbxSumq, num_queries * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_d2h));

    const float alpha = -2.0f;
    const float beta = 0.0f;
    CORAANNS_CUBLAS_CHECK(cublasSgemm(
        cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, num_centroids, num_queries, padded_dim, &alpha,
        d_centroids, padded_dim, d_queries, padded_dim, &beta, d_centroid_dists, num_centroids));

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

    if (ip_kernel_type_ == KernelType::Lut4Bit) {

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
                    throw std::invalid_argument("unsupported batch-q");
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
    if (ip_kernel_type_ == KernelType::Lut4Bit) {
        TOCK("7~9. Single-Stage Global Task Pool Pipeline (LUT_4BIT)");
    } else {
        TOCK("7~9. Single-Stage Global Task Pool Pipeline (BITWISE_DECOMP)");
    }
    if (profiling_enabled_) {
        std::cout << "------------------------------------------\n";
    }

#undef TICK
#undef TOCK
}

#undef CUDA_CHECK
#undef CORAANNS_CUBLAS_CHECK

} // namespace coraanns
