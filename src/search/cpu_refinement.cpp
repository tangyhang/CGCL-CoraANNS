// CPU consumer for the GPU candidate queue. Search arithmetic and synchronization
// follow the original implementation; this file contains no command-line entry point.
#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstring>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <queue>
#include <threads.h>
#include <vector>
#include <omp.h>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/utils/tools.hpp"
#include "coraanns/search/hybrid.cuh"

struct alignas(64) PaddedLock {
    std::atomic<bool> flag{false};
};

void IVF_RaBitQ_Raw::capture_thresholds(float *out, size_t nq) const {
    std::memcpy(out, h_dynamic_thresholds, nq * sizeof(float));
}
void IVF_RaBitQ_Raw::seed_thresholds(const float *in, size_t nq) {
    std::memcpy(h_dynamic_thresholds, in, nq * sizeof(float));
}

// Consume published candidates, refine distances, and update per-query top-k.
void IVF_RaBitQ_Raw::cpu_exact_rerank(const float *h_queries, size_t num_queries, size_t topk,
                                      int nprobe, float *h_final_dists, uint32_t *h_final_pids) {
    size_t LC_SIZE = (padded_dim * ex_bits + 7) / 8;
    rabitqlib::ex_ipfunc ip_func = rabitqlib::select_excode_ipfunc(ex_bits);
    const bool use_exact_refine = h_exact_base_mapping != nullptr;
    std::atomic<size_t> global_total_evals(0);
    std::atomic<uint64_t> g_cyc_read(0), g_cyc_ip(0), g_cyc_heap(0);

    std::atomic<double> max_spin_ms(0.0);
    std::atomic<double> max_comp_ms(0.0);

    const size_t kstride = ((topk + 15) / 16) * 16;
    float *topk_dist = (float *)aligned_alloc(64, num_queries * kstride * sizeof(float));
    uint32_t *topk_pid = (uint32_t *)aligned_alloc(64, num_queries * kstride * sizeof(uint32_t));
    struct alignas(64) QSlot {
        std::atomic<bool> lock;
        int cnt;
    };
    QSlot *slots = (QSlot *)aligned_alloc(64, num_queries * sizeof(QSlot));
    for (size_t i = 0; i < num_queries; ++i) {
        slots[i].lock.store(false);
        slots[i].cnt = 0;
    }

    std::atomic<int> global_read_head(0);

    const int thread_count = omp_get_max_threads();
#pragma omp parallel num_threads(thread_count)
    {
        size_t local_evals = 0;
        double local_spin = 0.0;
        double local_comp = 0.0;

        uint64_t cyc_read = 0, cyc_ip = 0, cyc_heap = 0;

        bool isFirst = true;

        while (true) {

            // Retain the original fixed-size work stealing policy.
            int chunk_size = 8192;
            int start_task_idx = global_read_head.fetch_add(chunk_size, std::memory_order_relaxed);

            if (start_task_idx >= GLOBAL_ASYNC_BUFFER_SIZE)
                break;

            for (int i = 0; i < chunk_size; ++i) {
                int task_idx = start_task_idx + i;
                if (task_idx >= GLOBAL_ASYNC_BUFFER_SIZE)
                    break;

                volatile uint32_t *p_gid = &h_global_buffer->items[task_idx].gid;
                uint32_t gid;

                int spin_count = 0;
                // A sentinel denotes an unpublished item; GPU completion bounds the queue.
                while (true) {
                    gid = *p_gid;
                    if (gid != 0xFFFFFFFF)
                        break;

                    if (*(volatile int *)h_gpu_done == 1 &&
                        task_idx >= *(volatile int *)h_global_write_count) {
                        break;
                    }
                }

                if (gid == 0xFFFFFFFF)
                    break;

                uint64_t r0 = profiling_enabled_ ? __rdtsc() : 0;
                int q = h_global_buffer->items[task_idx].query_idx;
                int p = h_global_buffer->items[task_idx].probe_idx;
                float low_dist = h_global_buffer->items[task_idx].low_dist;
                // Recheck the latest threshold before reading extended codes.
                if (cpu_secondary_filter_enabled_ && low_dist >= h_dynamic_thresholds[q]) {
                    if (profiling_enabled_)
                        cyc_read += __rdtsc() - r0;
                    continue;
                }
                uint64_t r1 = profiling_enabled_ ? __rdtsc() : 0;
                if (profiling_enabled_)
                    cyc_read += r1 - r0;

                float refine_dist = 0.0f;
                if (use_exact_refine) {
                    const uint32_t pid = h_pids_ref[gid];
                    if (pid >= exact_base_rows || exact_base_dim != dim) {
                        continue;
                    }
                    const float *base_vec = reinterpret_cast<const float *>(
                        h_exact_base_mapping + exact_base_data_offset_bytes +
                        (size_t)pid * exact_base_row_stride_bytes);
                    const float *query_vec = h_queries + (size_t)q * dim;
#pragma omp simd reduction(+ : refine_dist)
                    for (size_t d = 0; d < dim; ++d) {
                        const float diff = query_vec[d] - base_vec[d];
                        refine_dist += diff * diff;
                    }
                } else {
                    const float exact_ip = h_global_buffer->items[task_idx].exact_ip;
                    const float q_g_add = h_probe_vals[q * nprobe + p];
                    const uint8_t *vec_long_code = h_long_codes_ref + gid * LC_SIZE;
                    const float ip2 =
                        ip_func(h_rotated_queries + q * padded_dim, vec_long_code, padded_dim);
                    const float f_ex_add = h_ex_factors_ref[gid * 2 + 0];
                    const float f_ex_rescale = h_ex_factors_ref[gid * 2 + 1];
                    refine_dist =
                        f_ex_add + q_g_add +
                        f_ex_rescale * ((float)(1 << ex_bits) * exact_ip + ip2 + h_G_kbxSumq[q]);
                }
                uint64_t r2 = profiling_enabled_ ? __rdtsc() : 0;
                if (profiling_enabled_)
                    cyc_ip += r2 - r1;

                if (refine_dist >= h_dynamic_thresholds[q]) {
                    local_evals++;
                    continue;
                }

                // Serialize each query's top-k update and publish its new threshold.
                QSlot &s = slots[q];
                while (s.lock.exchange(true, std::memory_order_acquire)) {
                }
                float *td = topk_dist + (size_t)q * kstride;
                uint32_t *tp = topk_pid + (size_t)q * kstride;
                if (s.cnt < (int)topk) {
                    int i = s.cnt - 1;
                    while (i >= 0 && td[i] > refine_dist) {
                        td[i + 1] = td[i];
                        tp[i + 1] = tp[i];
                        --i;
                    }
                    td[i + 1] = refine_dist;
                    tp[i + 1] = h_pids_ref[gid];
                    ++s.cnt;
                    if (s.cnt == (int)topk)
                        h_dynamic_thresholds[q] = td[topk - 1];
                } else if (refine_dist < td[topk - 1]) {
                    int i = (int)topk - 2;
                    while (i >= 0 && td[i] > refine_dist) {
                        td[i + 1] = td[i];
                        tp[i + 1] = tp[i];
                        --i;
                    }
                    td[i + 1] = refine_dist;
                    tp[i + 1] = h_pids_ref[gid];
                    h_dynamic_thresholds[q] = td[topk - 1];
                }
                s.lock.store(false, std::memory_order_release);

                uint64_t r3 = profiling_enabled_ ? __rdtsc() : 0;
                if (profiling_enabled_)
                    cyc_heap += r3 - r2;
                local_evals++;
            }
        }
        if (profiling_enabled_ || candidate_stats_enabled_) {
            global_total_evals += local_evals;
        }
        if (profiling_enabled_) {
            g_cyc_read += cyc_read;
            g_cyc_ip += cyc_ip;
            g_cyc_heap += cyc_heap;
        }
    }

    last_written_count = *(volatile int *)h_global_write_count;
    last_refined_count_ = global_total_evals.load();

    if (profiling_enabled_) {
        uint64_t tot = g_cyc_read.load() + g_cyc_ip.load() + g_cyc_heap.load();
        printf("[Cora-Profile] read_prune=%.1f%% ip_func=%.1f%% heap_lock=%.1f%%\n",
               tot ? 100.0 * g_cyc_read.load() / tot : 0.0,
               tot ? 100.0 * g_cyc_ip.load() / tot : 0.0,
               tot ? 100.0 * g_cyc_heap.load() / tot : 0.0);
        printf("[Cora-Refine] candidates=%d refined=%zu refinement_ratio=%.6f\n",
               last_written_count, global_total_evals.load(),
               last_written_count ? static_cast<double>(global_total_evals.load()) /
                                        static_cast<double>(last_written_count)
                                  : 0.0);
    }

#pragma omp parallel for
    for (int q = 0; q < (int)num_queries; q++) {
        int base = q * (int)topk;
        float *td = topk_dist + (size_t)q * kstride;
        uint32_t *tp = topk_pid + (size_t)q * kstride;
        int n = slots[q].cnt;
        for (int i = 0; i < n; ++i) {
            h_final_dists[base + i] = td[i];
            h_final_pids[base + i] = tp[i];
        }
        for (int i = n; i < (int)topk; ++i) {
            h_final_dists[base + i] = FLT_MAX;
            h_final_pids[base + i] = 0;
        }
    }
}
