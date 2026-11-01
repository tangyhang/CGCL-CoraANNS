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
#include "coraanns/queue/release_tracker.hpp"

namespace {
inline GlobalTaskItem load_task_relaxed(const GlobalTaskItem *item) {
    GlobalTaskItem result;
#if defined(__x86_64__) && defined(__AVX__)
    __m128i value;
    asm volatile("vmovdqa %1, %0" : "=x"(value) : "m"(*item) : "memory");
    std::memcpy(&result, &value, sizeof(result));
#else
    __atomic_load(item, &result, __ATOMIC_RELAXED);
#endif
    return result;
}

inline void clear_tasks_relaxed(GlobalTaskItem *items, QueueIndex count) {
#if defined(__x86_64__) && defined(__AVX__)
    const __m128i empty = _mm_set1_epi32(-1);
    for (QueueIndex i = 0; i < count; ++i) {
        asm volatile("vmovdqa %1, %0" : "=m"(items[i]) : "x"(empty) : "memory");
    }
#else
    GlobalTaskItem empty;
    std::memset(&empty, 0xFF, sizeof(empty));
    for (QueueIndex i = 0; i < count; ++i) {
        __atomic_store(items + i, &empty, __ATOMIC_RELAXED);
    }
#endif
}

} // namespace

void IVF_RaBitQ_Raw::capture_thresholds(float *out, size_t nq) const {
    std::memcpy(out, h_dynamic_thresholds, nq * sizeof(float));
}
void IVF_RaBitQ_Raw::seed_thresholds(const float *in, size_t nq) {
    std::memcpy(h_dynamic_thresholds, in, nq * sizeof(float));
}

void IVF_RaBitQ_Raw::cpu_exact_rerank(const float *h_queries, size_t num_queries, size_t topk,
                                      int nprobe, float *h_final_dists, uint32_t *h_final_pids) {
    size_t LC_SIZE = (padded_dim * ex_bits + 7) / 8;
    rabitqlib::ex_ipfunc ip_func = rabitqlib::select_excode_ipfunc(ex_bits);
    const bool use_exact_refine = h_exact_base_mapping != nullptr;
    std::atomic<size_t> global_total_evals(0);
    std::atomic<uint64_t> g_cyc_read(0), g_cyc_ip(0), g_cyc_heap(0);

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

    const QueueIndex capacity = queue_capacity_;
    const QueueIndex mask = capacity - 1;
    const QueueIndex chunk_size = std::min<QueueIndex>(QUEUE_CONSUMER_CHUNK, capacity);
    std::atomic<QueueIndex> global_read_head(0);
    coraanns::queue::ReleaseTracker released(*h_queue_progress, capacity, chunk_size);
    auto done = [&]() { return __atomic_load_n(h_gpu_done, __ATOMIC_ACQUIRE) != 0; };
    auto aborted = [&]() {
        return __atomic_load_n(&h_queue_progress->overflow, __ATOMIC_ACQUIRE) != 0;
    };
    const int thread_count = omp_get_max_threads();
#pragma omp parallel num_threads(thread_count)
    {
        size_t local_evals = 0;
        uint64_t cyc_read = 0, cyc_ip = 0, cyc_heap = 0;
        bool stop = false;
        while (!stop && !aborted()) {
            const QueueIndex start_task_idx =
                global_read_head.fetch_add(chunk_size, std::memory_order_relaxed);
            if (done() && start_task_idx >= *h_global_write_count)
                break;

            if (start_task_idx >= capacity) {
                while (!released.readable_generation(start_task_idx)) {
                    if (aborted() || (done() && start_task_idx >= *h_global_write_count)) {
                        stop = true;
                        break;
                    }
                    _mm_pause();
                }
            }
            if (stop)
                break;
            QueueIndex copied = 0;
            for (QueueIndex i = 0; i < chunk_size; ++i) {
                const QueueIndex task_idx = start_task_idx + i;
                const auto *item = &h_global_buffer->items[task_idx & mask];
                GlobalTaskItem record;
                while (true) {
                    record = load_task_relaxed(item);
                    if (record.gid != UINT32_MAX)
                        break;
                    if (aborted() || (done() && task_idx >= *h_global_write_count)) {
                        stop = true;
                        break;
                    }
                    _mm_pause();
                }
                if (stop)
                    break;
                const uint32_t gid = record.gid;
                const int q = record.query_idx;
                const int p = record.probe_idx;
                const float exact_ip = record.exact_ip;
                const float low_dist = record.low_dist;
                ++copied;

                uint64_t r0 = profiling_enabled_ ? __rdtsc() : 0;
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
            if (copied != 0) {
                // Clear before returning storage; claims alone never advance Pr.
                clear_tasks_relaxed(&h_global_buffer->items[start_task_idx & mask], chunk_size);
                released.complete(start_task_idx);
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

    last_written_count = *h_global_write_count;
    last_refined_count_ = global_total_evals.load();

    if (profiling_enabled_) {
        uint64_t tot = g_cyc_read.load() + g_cyc_ip.load() + g_cyc_heap.load();
        printf("[Cora-Profile] read_prune=%.1f%% ip_func=%.1f%% heap_lock=%.1f%%\n",
               tot ? 100.0 * g_cyc_read.load() / tot : 0.0,
               tot ? 100.0 * g_cyc_ip.load() / tot : 0.0,
               tot ? 100.0 * g_cyc_heap.load() / tot : 0.0);
        printf("[Cora-Refine] candidates=%llu refined=%zu refinement_ratio=%.6f\n",
               static_cast<unsigned long long>(last_written_count), global_total_evals.load(),
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
    std::free(slots);
    std::free(topk_pid);
    std::free(topk_dist);
}
