#include <iostream>
#include <algorithm>
#include <limits>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <omp.h>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/io.hpp"
#include "coraanns/benchmark/recall.hpp"
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/utils/tools.hpp"
#include "coraanns/benchmark/options.hpp"

#include "coraanns/search/hybrid.cuh"
#include "coraanns/index/search_data.hpp"

#include "rabitqlib/index/ivf/ivf.hpp"

using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

namespace {

void print_usage(const char *program) {
    std::cerr << "Usage: " << program
              << " <index_file> <query_file> <gt_file>"
                 " [--topk 100] [--nprobes 40,50,...] [--rounds 5]"
                 " [--warmup 1] [--batch-q 4] [--query-chunk N] [--threads N]"
                 " [--vanguard-size 5] [--cpu-secondary-filter true]"
                 " [--queue-capacity 67108864] [--queue-spill-capacity 16777216]"
                 " [--cross-query-reuse true] [--candidate-stats false]"
                 " [--gpu-filter-timing false] [--profile false]"
                 " [--exact-base FILE] [--recall-base FILE] [--recall-once false]\n";
}
} // namespace

int main(int argc, char **argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const auto recall_base = recall_eval::take_base_option(argc, argv);
        std::string index_file = argv[1];
        std::string query_file = argv[2];
        std::string gt_file = argv[3];

        rabitq_benchmark::CliOptions options(argc, argv, 4);
        options.require_known({"topk", "nprobes", "rounds", "warmup", "batch-q", "threads",
                               "query-chunk", "vanguard-size", "cpu-secondary-filter", "profile",
                               "cross-query-reuse", "candidate-stats", "gpu-filter-timing",
                               "exact-base", "recall-once", "queue-capacity",
                               "queue-spill-capacity"});
        const size_t topk = options.get_size("topk", 100);
        const size_t test_round = options.get_size("rounds", 5);
        const size_t warmup_rounds = options.get_size("warmup", 1);
        const bool recall_once = options.get_bool("recall-once", false);
        const size_t requested_batch_q = options.get_size("batch-q", coraanns::default_batch_q);
        if (requested_batch_q < 1 || requested_batch_q > 6) {
            throw std::invalid_argument("batch-q must be in [1, 6]");
        }
        const int batch_q = static_cast<int>(requested_batch_q);
        const size_t requested_threads =
            options.get_size("threads", static_cast<size_t>(omp_get_max_threads()));
        if (requested_threads == 0 ||
            requested_threads > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("threads must be a positive int");
        }
        const int cpu_threads = static_cast<int>(requested_threads);
        omp_set_dynamic(0);
        omp_set_num_threads(cpu_threads);
        const size_t vanguard_size =
            options.get_size("vanguard-size", coraanns::default_vanguard_size);
        const bool profile = options.get_bool("profile", false);
        const bool candidate_stats = options.get_bool("candidate-stats", false);
        const bool gpu_filter_timing = options.get_bool("gpu-filter-timing", false);
        const bool cross_query_reuse = options.get_bool("cross-query-reuse", true);
        const bool cpu_secondary_filter = options.get_bool("cpu-secondary-filter", true);
        const size_t queue_capacity =
            options.get_size("queue-capacity", coraanns::queue::default_capacity);
        const size_t queue_spill_capacity =
            options.get_size("queue-spill-capacity", coraanns::queue::default_spill_capacity);
        if (queue_capacity == 0 || (queue_capacity & (queue_capacity - 1)) != 0) {
            throw std::invalid_argument("queue-capacity must be a positive power of two");
        }
        if (queue_capacity > SIZE_MAX / sizeof(coraanns::queue::Candidate) ||
            queue_spill_capacity > SIZE_MAX / sizeof(coraanns::queue::SpillRecord)) {
            throw std::invalid_argument("queue allocation size exceeds address space");
        }
        const auto exact_base_file = options.get("exact-base", "");

        std::cout << "Loading queries and groundtruth..." << std::endl;
        data_type query;
        gt_type gt;

        rabitqlib::load_auto<float, data_type>(query_file, query);
        rabitqlib::load_auto<uint32_t, gt_type>(gt_file, gt);

        size_t nq = query.rows();
        size_t dim = query.cols();
        std::cout << "Loaded " << nq << " queries with dimension " << dim << std::endl;
        const size_t requested_query_chunk = options.get_size("query-chunk", nq);
        if (requested_query_chunk == 0) {
            throw std::invalid_argument("query-chunk must be positive");
        }
        const size_t query_chunk = std::min({requested_query_chunk, nq, coraanns::max_query_chunk});

        std::cout << "Initializing GPU Index..." << std::endl;
        rabitqlib::ivf::IVF cpu_ivf;
        cpu_ivf.load(index_file.c_str());

        // In once mode, base reads for recall start after all timed searches.
        std::unique_ptr<recall_eval::Evaluator> recall_evaluator;
        if (!recall_once) {
            recall_evaluator = std::make_unique<recall_eval::Evaluator>(
                query, gt, topk, recall_base, cpu_ivf.metric_type());
        }

        rabitq_benchmark::validate_benchmark_shape(query.rows(), query.cols(), gt.rows(), gt.cols(),
                                                   cpu_ivf.dim(), topk, test_round);

        std::unique_ptr<coraanns::io::MappedVectors> exact_base;
        if (!exact_base_file.empty()) {
            if (cpu_ivf.metric_type() != rabitqlib::METRIC_L2) {
                std::cerr << "Exact-refine diagnostic currently supports only L2 indexes.\n";
                return 1;
            }
            try {
                exact_base = std::make_unique<coraanns::io::MappedVectors>(exact_base_file);
            } catch (const std::exception &e) {
                std::cerr << e.what() << '\n';
                return 1;
            }
            if (exact_base->rows() != cpu_ivf.num_vectors() || exact_base->dim() != cpu_ivf.dim()) {
                std::cerr << "Exact-refine base shape mismatch: file=" << exact_base->rows() << 'x'
                          << exact_base->dim() << ", index=" << cpu_ivf.num_vectors() << 'x'
                          << cpu_ivf.dim() << '\n';
                return 1;
            }
            std::cout << "Exact FP32 refine enabled with mmap base: " << exact_base_file << " ("
                      << exact_base->rows() << 'x' << exact_base->dim() << ")\n";
        }

        const size_t num_cluster = cpu_ivf.num_clusters();
        const size_t padded_dim = cpu_ivf.padded_dim();
        const auto nprobes =
            rabitq_benchmark::parse_nprobes(options.get("nprobes", ""), num_cluster);

        size_t length = nprobes.size();

        const size_t max_nprobe = *std::max_element(nprobes.begin(), nprobes.end());
        coraanns::validate_search_shape(query_chunk, topk, max_nprobe, num_cluster, padded_dim);
        coraanns::index::SearchData packed(cpu_ivf, !exact_base, cpu_threads);
        coraanns::HybridIndex gpu_ivf;
        gpu_ivf.set_profiling(profile);
        gpu_ivf.set_candidate_stats(candidate_stats);
        gpu_ivf.set_gpu_filter_timing(gpu_filter_timing);
        gpu_ivf.set_cross_query_reuse(cross_query_reuse);
        gpu_ivf.set_cpu_secondary_filter(cpu_secondary_filter);
        gpu_ivf.set_vanguard_size(vanguard_size);
        gpu_ivf.load_from_raw_pointers(
            cpu_ivf.num_vectors(), cpu_ivf.dim(), padded_dim, num_cluster, cpu_ivf.ex_bits(),
            packed.h_centroids.data(), packed.host_rotator.data(), packed.host_short_data.data(),
            packed.host_short_factors.data(), packed.host_long_codes, packed.host_ex_factors.data(),
            cpu_ivf.ids(), packed.cluster_sizes.data());

        if (exact_base) {
            gpu_ivf.enable_exact_refine(exact_base->mapping(), exact_base->rows(),
                                        exact_base->dim(), exact_base->row_stride_bytes(),
                                        exact_base->data_offset_bytes());
        }

        std::cout << "Max cluster size: " << packed.max_cluster_size << std::endl;

        gpu_ivf.set_queue_capacity(queue_capacity);
        gpu_ivf.set_queue_spill_capacity(queue_spill_capacity);
        std::cout << "Allocating GPU workspace...\n";
        gpu_ivf.prepare_workspace(query_chunk, topk, max_nprobe);

        std::cout << "Algorithm: CoraANN (GPU filtering + CPU refinement)\n"
                  << "topk: " << topk << "\n"
                  << "batch_q: " << batch_q << "\n"
                  << "query_chunk: " << query_chunk << "\n"
                  << "cpu_threads: " << cpu_threads << "\n"
                  << "queue_capacity: " << queue_capacity << "\n"
                  << "queue_spill_capacity: " << queue_spill_capacity << "\n"
                  << "vanguard_size: " << vanguard_size << "\n"
                  << "cpu_secondary_filter: " << (cpu_secondary_filter ? "true" : "false") << "\n"
                  << "cross_query_reuse: " << (cross_query_reuse ? "true" : "false") << "\n"
                  << "candidate_stats: " << (candidate_stats ? "true" : "false") << "\n"
                  << "gpu_filter_timing: " << (gpu_filter_timing ? "true" : "false") << "\n"
                  << "rounds: " << test_round << "\n"
                  << "warmup_rounds: " << warmup_rounds << "\n"
                  << "profile: " << (profile ? "true" : "false") << "\n"
                  << "nprobes: " << rabitq_benchmark::join_values(nprobes) << '\n';

        std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(length));
        std::vector<std::vector<float>> all_recall(recall_once ? 1 : test_round,
                                                   std::vector<float>(length));
        std::vector<std::vector<uint32_t>> recall_results(recall_once ? length : 0);
        for (auto &ids : recall_results)
            ids.resize(nq * topk);
        if (recall_once) {
            std::cout << "[RecallSchedule] policy=once-after-benchmark source_round=1"
                      << " evaluations=" << length << '\n';
        }
        rabitqlib::StopW stopw;

        std::vector<float> results_dists(nq * topk);
        std::vector<uint32_t> results_pids(nq * topk);

        struct SearchStats {
            uint64_t submitted = 0;
            uint64_t actual_refined = 0;
            uint64_t spilled = 0;
            uint64_t wraps = 0;
            double gpu_filter_ms = 0.0;
        };
        auto search_all_queries = [&](size_t nprobe) {
            SearchStats stats;
            for (size_t begin = 0; begin < nq; begin += query_chunk) {
                const size_t count = std::min(query_chunk, nq - begin);
                const float *chunk_queries = query.data() + begin * dim;
                float *chunk_dists = results_dists.data() + begin * topk;
                uint32_t *chunk_pids = results_pids.data() + begin * topk;
                gpu_ivf.reset_buffer(chunk_queries, count);
                gpu_ivf.search(chunk_queries, count, topk, nprobe, chunk_dists, chunk_pids,
                               batch_q);
                stats.submitted += static_cast<uint64_t>(gpu_ivf.get_last_submitted_count());
                stats.actual_refined += static_cast<uint64_t>(gpu_ivf.get_last_refined_count());
                stats.gpu_filter_ms += gpu_ivf.get_last_gpu_filter_ms();
                stats.spilled += gpu_ivf.get_last_spilled_count();
                const auto submitted = gpu_ivf.get_last_submitted_count();
                stats.wraps += submitted == 0 ? 0 : (submitted - 1) / queue_capacity;
            }
            return stats;
        };

        std::cout << "Starting batch benchmark..." << std::endl;

        for (size_t l = 0; l < length; ++l) {
            const size_t nprobe = nprobes[l];
            for (size_t warmup = 0; warmup < warmup_rounds; ++warmup) {
                search_all_queries(nprobe);
            }
            for (size_t r = 0; r < test_round; r++) {
                stopw.reset();
                const SearchStats stats = search_all_queries(nprobe);
                float total_time = stopw.get_elapsed_micro();

                all_qps[r][l] = static_cast<float>(nq) / (total_time / 1e6F);
                if (recall_once) {
                    // Save first-round results outside the search timer.
                    if (r == 0) {
                        std::copy(results_pids.begin(), results_pids.end(),
                                  recall_results[l].begin());
                    }
                } else {
                    all_recall[r][l] = recall_evaluator->evaluate(results_pids);
                }
                std::cout << "[Run] round=" << (r + 1) << " nprobe=" << nprobe
                          << " QPS=" << all_qps[r][l];
                if (recall_once) {
                    std::cout << " recall=deferred\n";
                } else {
                    std::cout << " recall=" << all_recall[r][l] << '\n';
                }
                std::printf(
                    "[QueueStats] round=%zu nprobe=%zu submitted=%llu spilled=%llu wraps=%llu\n",
                    r + 1, nprobe, static_cast<unsigned long long>(stats.submitted),
                    static_cast<unsigned long long>(stats.spilled),
                    static_cast<unsigned long long>(stats.wraps));
                if (candidate_stats) {
                    std::printf("[CandidateStats] round=%zu nprobe=%zu queries=%zu "
                                "submitted=%llu actual_refined=%llu "
                                "submitted_per_query=%.6f actual_refined_per_query=%.6f\n",
                                r + 1, nprobe, nq, static_cast<unsigned long long>(stats.submitted),
                                static_cast<unsigned long long>(stats.actual_refined),
                                static_cast<double>(stats.submitted) / nq,
                                static_cast<double>(stats.actual_refined) / nq);
                }
                if (gpu_filter_timing) {
                    std::printf("[GPUFilter] round=%zu nprobe=%zu queries=%zu "
                                "total_ms=%.6f ms_per_query=%.9f\n",
                                r + 1, nprobe, nq, stats.gpu_filter_ms, stats.gpu_filter_ms / nq);
                }
            }
        }

        if (recall_once) {
            std::cout << "[RecallSchedule] QPS rounds complete; evaluating saved results"
                      << std::endl;
            recall_evaluator = std::make_unique<recall_eval::Evaluator>(
                query, gt, topk, recall_base, cpu_ivf.metric_type());
            for (size_t l = 0; l < length; ++l) {
                all_recall[0][l] = recall_evaluator->evaluate(recall_results[l]);
                std::cout << "[RecallPoint] nprobe=" << nprobes[l]
                          << " source_round=1 recall=" << all_recall[0][l] << '\n';
            }
        }

        auto avg_qps = rabitqlib::horizontal_avg(all_qps);
        auto avg_recall = recall_once ? all_recall[0] : rabitqlib::horizontal_avg(all_recall);

        std::cout << "nprobe\tQPS\trecall\n";
        for (size_t i = 0; i < length; ++i) {
            std::cout << nprobes[i] << '\t' << avg_qps[i] << '\t' << avg_recall[i] << '\n';
        }

    } catch (const std::exception &error) {
        std::cerr << "coraanns_search failed: " << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
