#include <iostream>
#include <vector>
#include <string>
#include <cfloat>
#include <queue>
#include <atomic>
#include <memory>
#include <immintrin.h>
#include <threads.h>
#include <thread>
#include <chrono>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <algorithm>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/io.hpp"
#include "coraanns/benchmark/recall.hpp"
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/utils/tools.hpp"
#include "coraanns/benchmark/mechanism_options.hpp"

#include "coraanns/experimental/backlog_trace.cuh"

#define private public
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/quantization/rabitq_impl.hpp"
#undef private

using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

static size_t topk = 100;


float get_const_scaling_factors(size_t dim, size_t ex_bits) {
    constexpr long kConstNum = 100;
    std::vector<double> h_rand_row_normalized_abs(kConstNum * dim);
    std::mt19937_64 rng(7ULL); 
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    for (long i = 0; i < kConstNum; ++i) {
        double sq_sum = 0.0;
        size_t row_offset = i * dim;
        for (size_t j = 0; j < dim; ++j) {
            double val = normal_dist(rng);
            h_rand_row_normalized_abs[row_offset + j] = val;
            sq_sum += val * val;
        }
        double norm = std::sqrt(sq_sum);
        double inv_norm = (norm > 0.0) ? (1.0 / norm) : 0.0;
        for (size_t j = 0; j < dim; ++j) {
            h_rand_row_normalized_abs[row_offset + j] = std::abs(h_rand_row_normalized_abs[row_offset + j] * inv_norm);
        }
    }
    double sum = 0.0;
    for (long j = 0; j < kConstNum; ++j) {
        sum += rabitqlib::quant::rabitq_impl::ex_bits::best_rescale_factor<double>(&h_rand_row_normalized_abs[j * dim], dim, ex_bits);
    }
    return static_cast<float>(sum / kConstNum);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        rabitq_benchmark::print_experiment_usage(argv[0]);
        return 1;
    }

    try {
    const auto recall_base = recall_eval::take_base_option(argc, argv);
    const rabitq_benchmark::ExperimentOptions experiment(argc, argv, VANGUARD_SIZE);
    if (experiment.vanguard_size != 0 || experiment.cpu_secondary_filter ||
        experiment.profile) {
        throw std::invalid_argument(
            "This trace measures the simple asynchronous pipeline; use "
            "--vanguard-size 0 --cpu-secondary-filter false --profile false");
    }
    topk = experiment.topk;
    const size_t test_round = experiment.rounds;
    std::string index_file = argv[1];
    std::string query_file = argv[2];
    std::string gt_file = argv[3];


    std::cout << "Loading queries and groundtruth..." << std::endl;
    data_type query; gt_type gt;
    rabitqlib::load_auto<float, data_type>(query_file, query);
    rabitqlib::load_auto<uint32_t, gt_type>(gt_file, gt);
    
    size_t nq = query.rows(); size_t dim = query.cols();
    std::cout<< "Loaded " << nq << " queries with dimension " << dim << std::endl;

    std::cout << "Initializing GPU Index..." << std::endl;
    rabitqlib::ivf::IVF cpu_ivf;
    cpu_ivf.load(index_file.c_str());
    recall_eval::Evaluator recall_evaluator(query, gt, topk, recall_base, cpu_ivf.metric_type());
    rabitq_benchmark::validate_benchmark_shape(
        query.rows(), query.cols(), gt.rows(), gt.cols(), cpu_ivf.dim_, topk,
        test_round);

    size_t num_cluster = cpu_ivf.num_cluster_;
    size_t padded_dim = cpu_ivf.padded_dim_;
    size_t num_words = padded_dim / 32;

    std::vector<float> h_centroids(num_cluster * padded_dim);
    std::vector<size_t> cluster_sizes(num_cluster);
    size_t max_cluster_size = 0; 

    for (size_t i = 0; i < num_cluster; ++i) {
        std::memcpy(h_centroids.data() + i * padded_dim, cpu_ivf.initer_->centroid(i), padded_dim * sizeof(float));
        size_t c_size = cpu_ivf.cluster_lst_[i].num();
        cluster_sizes[i] = c_size;
        if (c_size > max_cluster_size) max_cluster_size = c_size; 
    }

    std::vector<uint32_t> host_short_data(cpu_ivf.num_ * num_words, 0);
    std::vector<float> host_short_factors(cpu_ivf.num_ * 3, 0.0f);

    size_t ex_bits = cpu_ivf.ex_bits_;
    std::cout<<"ex_bits: "<<ex_bits<<std::endl;
    std::cout<<"topk: "<<topk<<std::endl;
    size_t long_code_size = (padded_dim * ex_bits + 7) / 8;
    std::vector<uint8_t> host_long_code(cpu_ivf.num_ * long_code_size, 0);
    std::vector<float> host_ex_factors(cpu_ivf.num_ * 2, 0.0f); 

    size_t global_vec_idx = 0;
    const int kPerm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
    int inv_kPerm0[16];
    for (int j = 0; j < 16; ++j) inv_kPerm0[kPerm0[j]] = j;

    for (size_t c = 0; c < num_cluster; ++c) {
        const auto& cluster = cpu_ivf.cluster_lst_[c];
        size_t n = cluster.num();
        size_t cluster_start_index = global_vec_idx;

        const char* ex_ptr = cluster.ex_data();
        if (ex_bits > 0 && ex_ptr != nullptr) { 
            for (size_t v = 0; v < n; ++v) {
                size_t current_global_idx = cluster_start_index + v;
                rabitqlib::ConstExDataMap<float> cur_ex(ex_ptr, padded_dim, ex_bits);
                host_ex_factors[current_global_idx * 2 + 0] = cur_ex.f_add_ex();
                host_ex_factors[current_global_idx * 2 + 1] = cur_ex.f_rescale_ex();
                std::memcpy(&host_long_code[current_global_idx * long_code_size], cur_ex.ex_code(), long_code_size);
                ex_ptr += rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);
            }
        }

        const char* batch_ptr = cluster.batch_data();
        for (size_t i = 0; i < n; i += 32) {
            size_t batch_len = std::min((size_t)32, n - i);
            const uint8_t* bin_code_bytes = reinterpret_cast<const uint8_t*>(batch_ptr);
            const float* batch_add = reinterpret_cast<const float*>(batch_ptr + padded_dim * 4);
            const float* batch_rescale = reinterpret_cast<const float*>(batch_ptr + padded_dim * 4 + 32 * 4);
            const float* batch_error = reinterpret_cast<const float*>(batch_ptr + padded_dim * 4 + 64 * 4);

            for (size_t v = 0; v < batch_len; ++v) {
                size_t current_global_idx = cluster_start_index + i + v;
                for (size_t word = 0; word < num_words; ++word) {
                    uint32_t gpu_word = 0;
                    for (int bit = 0; bit < 32; ++bit) {
                        size_t dim = word * 32 + bit;
                        size_t dim_in_group = dim % 8;
                        size_t offset_half = (dim_in_group < 4) ? 0 : 16;
                        uint8_t byte_val = bin_code_bytes[(dim / 8) * 32 + offset_half + inv_kPerm0[v % 16]];
                        uint8_t nibble = (v < 16) ? (byte_val & 0x0F) : (byte_val >> 4);
                        gpu_word |= (((nibble >> (3 - (dim_in_group % 4))) & 1) << (31 - bit));
                    }
                    host_short_data[cluster_start_index * num_words + word * n + i + v] = gpu_word;
                }
                host_short_factors[current_global_idx * 3 + 0] = batch_add[v];
                host_short_factors[current_global_idx * 3 + 1] = batch_rescale[v];
                host_short_factors[current_global_idx * 3 + 2] = batch_error[v];
            }
            batch_ptr += (padded_dim * 4 + 32 * 12); 
        }
        global_vec_idx += n; 
    }

    std::vector<float> host_rotator(dim * padded_dim);
    cpu_ivf.rotator_->save(reinterpret_cast<char*>(host_rotator.data()));

    IVF_RaBitQ_Raw gpu_ivf;
    experiment.configure(gpu_ivf);
    gpu_ivf.load_from_raw_pointers(
        cpu_ivf.num_, cpu_ivf.dim_, padded_dim, num_cluster, cpu_ivf.ex_bits_,
        h_centroids.data(), host_rotator.data(), host_short_data.data(), host_short_factors.data(), 
        host_long_code.data(), host_ex_factors.data(), cpu_ivf.ids_, cluster_sizes.data()
    );

    gpu_ivf.max_cluster_size_ = max_cluster_size;
    std::cout << "Max cluster size: " << max_cluster_size << std::endl;
    gpu_ivf.best_rescaling_factor = get_const_scaling_factors(padded_dim, cpu_ivf.ex_bits_);
    
    const auto nprobes = rabitq_benchmark::parse_nprobes(
        experiment.cli.get("nprobes", "123"), num_cluster);
    const size_t length = nprobes.size();
    std::cout << "Allocating GPU Workspace pool..." << std::endl;
    gpu_ivf.prepare_workspace(
        nq, topk, *std::max_element(nprobes.begin(), nprobes.end()));
    experiment.print("CoraSearch backlog_trace", nprobes);
    std::cout << "trace_time_origin: filtering phase, before producer launch\n"
              << "trace_sample_interval_us: 250 (actual intervals are in CSV)\n"
              << "trace_gpu_progress: approximate slot watermark, stride=65536\n"
              << "trace_cpu_progress: completed tasks, flush=4096 per worker\n"
              << "trace_QPS_includes_sampling_and_CSV: true; use the normal executable for QPS\n";

    std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(length));
    std::vector<std::vector<float>> all_recall(test_round, std::vector<float>(length));
    rabitqlib::StopW stopw;

    std::vector<float> results_dists(nq * topk);
    std::vector<uint32_t> results_pids(nq * topk);

    std::cout << "Starting batch benchmark..." << std::endl;
    for (size_t l = 0; l < length; ++l) {
        const size_t nprobe = nprobes[l];
        gpu_ivf.set_profiling(false);
        gpu_ivf.set_trace_enabled(false);
        for (size_t warmup = 0; warmup < experiment.warmup; ++warmup) {
            gpu_ivf.reset_buffer(query.data(), nq);
            gpu_ivf.search(query.data(), nq, topk, nprobe,
                           results_dists.data(), results_pids.data(),
                           experiment.batch_q);
        }
        gpu_ivf.set_profiling(experiment.profile);
        for (size_t r = 0; r < test_round; ++r) {
            gpu_ivf.set_trace_enabled(true);
            gpu_ivf.set_trace_output((experiment.output_dir /
                ("backlog_k" + std::to_string(topk) + "_nprobe" +
                 std::to_string(nprobe) + "_round" +
                 std::to_string(r + 1) + ".csv")).string());

        
            gpu_ivf.reset_buffer(query.data(), nq);

            stopw.reset();
            gpu_ivf.search(query.data(), nq, topk, nprobe, results_dists.data(), results_pids.data(), experiment.batch_q);
            float total_time = stopw.get_elapsed_micro();
            
            all_qps[r][l] = static_cast<float>(nq) / (total_time / 1e6F);
            all_recall[r][l] = recall_evaluator.evaluate(results_pids);
            std::cout << "[TraceRun] round=" << (r + 1) << " nprobe=" << nprobe
                      << " instrumented_QPS=" << all_qps[r][l]
                      << " recall=" << all_recall[r][l] << '\n';
        }
    }
    
    auto avg_qps = rabitqlib::horizontal_avg(all_qps);
    auto avg_recall = rabitqlib::horizontal_avg(all_recall);

    std::cout << "nprobe\tinstrumented_QPS\trecall\n";
    for (size_t i = 0; i < length; ++i) {
        std::cout << nprobes[i] << '\t' << avg_qps[i] << '\t'
                  << avg_recall[i] << '\n';
    }
    } catch (const std::exception& error) {
        std::cerr << "backlog_trace failed: " << error.what() << '\n';
        rabitq_benchmark::print_experiment_usage(argv[0]);
        return 1;
    }
    return 0;
}

// 🌟 核心优化 1：将锁对齐到 64 字节，彻底消灭 CPU 伪共享 (False Sharing)！
struct alignas(64) PaddedLock {
    std::atomic<bool> flag{false};
};


void IVF_RaBitQ_Raw::capture_thresholds(float* out, size_t nq) const {
    std::memcpy(out, h_dynamic_thresholds, nq * sizeof(float));
}
void IVF_RaBitQ_Raw::seed_thresholds(const float* in, size_t nq) {
    std::memcpy(h_dynamic_thresholds, in, nq * sizeof(float));
}

// =========================================================================
// 🌟 纯粹的雇佣兵模式：100% 榨干 CPU，无缝负载均衡 (带硬核计时与迟剪枝)
// =========================================================================
void IVF_RaBitQ_Raw::cpu_exact_rerank(size_t num_queries, size_t topk, int nprobe, float* h_final_dists, uint32_t* h_final_pids) {
    size_t LC_SIZE = (padded_dim * ex_bits + 7) / 8;
    rabitqlib::ex_ipfunc ip_func = rabitqlib::select_excode_ipfunc(ex_bits);
    std::atomic<size_t> global_total_evals(0);

    // Backlog trace is a mechanism-only experiment. Warmups disable progress
    // reporting, and performance curves use the uninstrumented executable.
    const bool enable_backlog_trace = trace_enabled_;
    constexpr size_t TRACE_CPU_PROGRESS_FLUSH = 4096;

    // Live counter used by the search-level sampling thread.
    // It counts tasks that have actually been handled by CPU, not tasks
    // pre-claimed by global_read_head.
    if (enable_backlog_trace) {
        trace_cpu_processed_live.store(0, std::memory_order_relaxed);
    }
    std::atomic<uint64_t> g_cyc_read(0), g_cyc_ip(0), g_cyc_heap(0);   // 🌟 新增

    std::atomic<double> max_spin_ms(0.0);
    std::atomic<double> max_comp_ms(0.0);
    
    const size_t kstride = ((topk + 15) / 16) * 16;   // 每 query 的 top-k 段 64B 对齐
    float*    topk_dist = (float*)   aligned_alloc(64, num_queries * kstride * sizeof(float));
    uint32_t* topk_pid  = (uint32_t*)aligned_alloc(64, num_queries * kstride * sizeof(uint32_t));
    struct alignas(64) QSlot { std::atomic<bool> lock; int cnt; };
    QSlot* slots = (QSlot*)aligned_alloc(64, num_queries * sizeof(QSlot));
    for (size_t i = 0; i < num_queries; ++i) { slots[i].lock.store(false); slots[i].cnt = 0; }

    // 🌟 全局读取头：声明在栈上即可，所有 OpenMP 线程共享竞争
    std::atomic<int> global_read_head(0);

    // Backlog sampling is now started in search() before GPU producer kernels launch.
    // cpu_exact_rerank() only updates trace_cpu_processed_live when a task is really handled.

    const int thread_count = omp_get_max_threads();
    #pragma omp parallel num_threads(thread_count)
    {
        size_t local_evals = 0;
        size_t completed_pending = 0;
        double local_spin = 0.0;
        double local_comp = 0.0;

        auto publish_completed = [&]() {
            if (enable_backlog_trace && completed_pending > 0) {
                trace_cpu_processed_live.fetch_add(completed_pending, std::memory_order_relaxed);
                completed_pending = 0;
            }
        };
        auto mark_one_completed = [&]() {
            if (enable_backlog_trace) {
                ++completed_pending;
                // Use a coarse flush granularity to avoid millions of contended
                // atomic updates across CPU worker threads. This is still far
                // finer than the scale of tens of millions of candidates.
                if (completed_pending >= TRACE_CPU_PROGRESS_FLUSH) publish_completed();
            }
        };

        uint64_t cyc_read = 0, cyc_ip = 0, cyc_heap = 0;  // 三段累加器

        bool isFirst = true;
        
        while (true) {
            // 🌟 核心优化 2：批处理窃取 (Chunked Stealing)！
            // 一口气拿多个任务，极大地均摊了原子竞争开销，且避免了连续任务间的微锁冲突
            int chunk_size = 8192;
            int start_task_idx = global_read_head.fetch_add(chunk_size, std::memory_order_relaxed);
            
            // =========================================================
            // 🌟 核心优化 2：基于水位的自适应窃取 (Adaptive Watermark Stealing)
            // =========================================================
            // 1. 窥探当前 CPU 整体读取到了哪里
            // int current_head = global_read_head.load(std::memory_order_relaxed);
            // // 2. 窥探当前 GPU 已经确实写到了哪里
            // int current_write = *(volatile int*)h_global_write_count; 

            // // 3. 计算“积压水位” (有多少任务已经 Ready 但还没被拿走)
            // int available_tasks = current_write - current_head;
            
            // // 4. 动态计算 Chunk Size：
            // // 如果积压如山，我们按比例拿走积压量的一小部分 (除以 线程数*2 保证平滑)
            // int chunk_size = 8192; // 默认极小块收尾
            // if (!isFirst) {
            //     int dynamic_size = available_tasks / thread_count;
            //     chunk_size = std::max(128, std::min(8192, dynamic_size));
            //     isFirst = false;
            //     // if (chunk_size > 2048) printf("[Adaptive Stealing] available_tasks=%d, chunk_size=%d\n", available_tasks, chunk_size);
            // }

            // // 5. 使用计算出的完美粒度发起原子窃取！
            // int start_task_idx = global_read_head.fetch_add(chunk_size, std::memory_order_relaxed);

            if (start_task_idx >= GLOBAL_ASYNC_BUFFER_SIZE) break; 

            for (int i = 0; i < chunk_size; ++i) {
                int task_idx = start_task_idx + i;
                if (task_idx >= GLOBAL_ASYNC_BUFFER_SIZE) break;

                // =========================================================
                // 🌟 终极优化 1：超前缓存预取 (Software Memory Prefetching)
                // 在处理当前任务时，命令硬件去把 4 个迭代之后的向量数据提前拉入 L1 Cache！
                // =========================================================
                // int prefetch_offset = 1;
                // if (i + prefetch_offset < chunk_size) {
                //     int future_task = start_task_idx + i + prefetch_offset;
                //     uint32_t future_gid = h_global_buffer->items[future_task].gid;
                //     // 如果 GPU 已经把未来的 GID 写进来了，立刻预取它的内存
                //     if (future_gid != 0xFFFFFFFF) {
                //         const uint8_t* future_ptr = h_long_codes_ref + future_gid * LC_SIZE;
                //         const float* future_factor = h_ex_factors_ref + future_gid * 2;
                //         // _MM_HINT_T0: 预取到最快的 L1 缓存中
                //         // for (int off = 0; off < LC_SIZE; off += 64)
                //         //     _mm_prefetch((const char*)future_ptr + off, _MM_HINT_T0);
                //         _mm_prefetch((const char*)future_ptr, _MM_HINT_T0);
                //         _mm_prefetch((const char*)future_factor, _MM_HINT_T0);
                //     }
                // }

                volatile uint32_t* p_gid = &h_global_buffer->items[task_idx].gid;
                uint32_t gid;
                
                int spin_count = 0;
                while (true) {
                    gid = *p_gid;
                    if (gid != 0xFFFFFFFF) break; 
                    
                    if (*(volatile int*)h_gpu_done == 1 && task_idx >= *(volatile int*)h_global_write_count) {
                        break;
                    }
                    // _mm_pause();
                }

                if (gid == 0xFFFFFFFF) break; 

                uint64_t r0 = profiling_enabled_ ? __rdtsc() : 0;
                int   q = h_global_buffer->items[task_idx].query_idx;
                int   p = h_global_buffer->items[task_idx].probe_idx;
                float low_dist = h_global_buffer->items[task_idx].low_dist;
                if (cpu_secondary_filter_enabled_ &&
                    low_dist >= h_dynamic_thresholds[q]) {
                    if (profiling_enabled_) cyc_read += __rdtsc() - r0;
                    mark_one_completed();
                    continue;
                }
                float exact_ip = h_global_buffer->items[task_idx].exact_ip;
                float q_g_add  = h_probe_vals[q * nprobe + p];
                uint64_t r1 = profiling_enabled_ ? __rdtsc() : 0;
                if (profiling_enabled_) cyc_read += r1 - r0;

                const uint8_t* vec_long_code = h_long_codes_ref + gid * LC_SIZE;
                float ip2 = ip_func(h_rotated_queries + q * padded_dim, vec_long_code, padded_dim);
                float f_ex_add = h_ex_factors_ref[gid*2+0], f_ex_rescale = h_ex_factors_ref[gid*2+1];
                float ex_dist = f_ex_add + q_g_add + f_ex_rescale * ((float)(1<<ex_bits)*exact_ip + ip2 + h_G_kbxSumq[q]);
                uint64_t r2 = profiling_enabled_ ? __rdtsc() : 0;
                if (profiling_enabled_) cyc_ip += r2 - r1;                                       // 段2: ip_func+ex_dist

                ++local_evals;
                if (ex_dist >= h_dynamic_thresholds[q]) {
                    mark_one_completed();
                    continue;
                }

                QSlot& s = slots[q];
                while (s.lock.exchange(true, std::memory_order_acquire)) { }
                float*    td = topk_dist + (size_t)q * kstride;
                uint32_t* tp = topk_pid  + (size_t)q * kstride;
                if (s.cnt < (int)topk) {                                  // 还没满
                    int i = s.cnt - 1;
                    while (i >= 0 && td[i] > ex_dist) { td[i+1]=td[i]; tp[i+1]=tp[i]; --i; }
                    td[i+1] = ex_dist; tp[i+1] = h_pids_ref[gid]; ++s.cnt;
                    if (s.cnt == (int)topk) h_dynamic_thresholds[q] = td[topk-1];
                } else if (ex_dist < td[topk-1]) {                        // 满了且能进
                    int i = (int)topk - 2;
                    while (i >= 0 && td[i] > ex_dist) { td[i+1]=td[i]; tp[i+1]=tp[i]; --i; }
                    td[i+1] = ex_dist; tp[i+1] = h_pids_ref[gid];
                    h_dynamic_thresholds[q] = td[topk-1];
                }
                s.lock.store(false, std::memory_order_release);
                
                uint64_t r3 = profiling_enabled_ ? __rdtsc() : 0;
                if (profiling_enabled_) cyc_heap += r3 - r2;                                     // 段3: 锁+heap
                mark_one_completed();
            }
        }
        publish_completed();
        global_total_evals += local_evals;
        g_cyc_read += cyc_read;    // 🌟 新增
        g_cyc_ip   += cyc_ip;      // 🌟 新增
        g_cyc_heap += cyc_heap;    // 🌟 新增

        // // 原子更新各线程的最大耗时
        // double cur_max_spin = max_spin_ms.load();
        // while(local_spin > cur_max_spin && !max_spin_ms.compare_exchange_weak(cur_max_spin, local_spin));
        
        // double cur_max_comp = max_comp_ms.load();
        // while(local_comp > cur_max_comp && !max_comp_ms.compare_exchange_weak(cur_max_comp, local_comp));
    }

    if (enable_backlog_trace) {
        trace_cpu_finished_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - trace_epoch_).count();
    }
    last_written_count = *(volatile int*)h_global_write_count;

    if (profiling_enabled_) {
    uint64_t tot = g_cyc_read.load() + g_cyc_ip.load() + g_cyc_heap.load();
    printf("  --> [Breakdown] | read+prune1: %.1f%% | ip_func: %.1f%% | heap+lock: %.1f%%\n",
        tot ? 100.0*g_cyc_read.load()/tot : 0.0,
        tot ? 100.0*g_cyc_ip.load()/tot  : 0.0,
        tot ? 100.0*g_cyc_heap.load()/tot : 0.0);
    printf("  --> [Breakdown] | tot: %llu | read+prune1: %llu | ip_func: %llu | heap+lock: %llu\n",
        static_cast<unsigned long long>(tot),
        static_cast<unsigned long long>(g_cyc_read.load()),
        static_cast<unsigned long long>(g_cyc_ip.load()),
        static_cast<unsigned long long>(g_cyc_heap.load()));
    
    printf("  --> [Debug] GPU Wrote %d items, CPU Evaluated %zu items\n", last_written_count, global_total_evals.load());
    }
    // printf("  --> [CPU Time] Max Spin-Wait Time: %.2f ms, Max Compute Time: %.2f ms\n", max_spin_ms.load(), max_comp_ms.load());

    #pragma omp parallel for
    for (int q = 0; q < (int)num_queries; q++) {
        int base = q * (int)topk;
        float*    td = topk_dist + (size_t)q * kstride;
        uint32_t* tp = topk_pid  + (size_t)q * kstride;
        int n = slots[q].cnt;
        for (int i = 0; i < n; ++i)            { h_final_dists[base+i]=td[i]; h_final_pids[base+i]=tp[i]; }
        for (int i = n; i < (int)topk; ++i)    { h_final_dists[base+i]=FLT_MAX; h_final_pids[base+i]=0; }
    }
    free(slots);
    free(topk_pid);
    free(topk_dist);
}
