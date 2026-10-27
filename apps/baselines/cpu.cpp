#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "coraanns/benchmark/options.hpp"
#include "coraanns/benchmark/recall.hpp"
#include "rabitqlib/defines.hpp"
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/utils/io.hpp"
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/utils/tools.hpp"

using PID = rabitqlib::PID;
using index_type = rabitqlib::ivf::IVF;
using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

namespace {

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " <index_file> <query_file> <gt_file>"
           " [--topk 100] [--nprobes 40,50,...] [--rounds 5]"
           " [--warmup 1] [--threads N] [--hacc true] [--profile false]"
           " [--recall-base FILE]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    try {
        const auto recall_base = recall_eval::take_base_option(argc, argv);
        // Preserve the old fourth positional argument (true/false) so existing
        // scripts keep working. New benchmark scripts should use named options.
        const bool legacy_mode =
            argc > 4 && !rabitq_benchmark::starts_with_option(argv[4]);
        bool use_hacc = true;
        std::size_t option_begin = 4;
        if (legacy_mode) {
            if (argc != 5) {
                throw std::invalid_argument(
                    "The legacy CPU form accepts only one positional hacc value");
            }
            const std::string value(argv[4]);
            if (value == "false") use_hacc = false;
            else if (value != "true") {
                throw std::invalid_argument("hacc must be true or false");
            }
            option_begin = 5;
        }

        rabitq_benchmark::CliOptions options(argc, argv, option_begin);
        options.require_known(
            {"topk", "nprobes", "rounds", "warmup", "threads", "hacc",
             "profile"});

        const std::size_t topk = options.get_size("topk", 100);
        const std::size_t test_rounds = options.get_size("rounds", 5);
        const std::size_t warmup_rounds = options.get_size("warmup", 1);
        const std::size_t requested_threads = options.get_size(
            "threads", static_cast<std::size_t>(omp_get_max_threads()));
        use_hacc = options.get_bool("hacc", use_hacc);
        const bool profile = options.get_bool("profile", false);
        if (requested_threads == 0 ||
            requested_threads > static_cast<std::size_t>(
                                    std::numeric_limits<int>::max())) {
            throw std::invalid_argument("threads must be a positive int");
        }
        const int thread_count = static_cast<int>(requested_threads);
        omp_set_dynamic(0);
        omp_set_num_threads(thread_count);
        g_cpu_stats_enabled = profile;

        data_type query;
        gt_type gt;
        rabitqlib::load_auto<float, data_type>(argv[2], query);
        rabitqlib::load_auto<uint32_t, gt_type>(argv[3], gt);

        index_type ivf;
        ivf.load(argv[1]);

        rabitq_benchmark::validate_benchmark_shape(
            query.rows(), query.cols(), gt.rows(), gt.cols(), ivf.dim(), topk,
            test_rounds);
        const auto nprobes = rabitq_benchmark::parse_nprobes(
            options.get("nprobes", ""), ivf.num_clusters());

        const std::size_t nq = query.rows();
        recall_eval::Evaluator recall_evaluator(query, gt, topk, recall_base, ivf.metric_type());
        std::vector<PID> all_results(nq * topk);
        std::vector<std::vector<float>> all_qps(
            test_rounds, std::vector<float>(nprobes.size()));
        std::vector<std::vector<float>> all_recall(
            test_rounds, std::vector<float>(nprobes.size()));
        rabitqlib::StopW stopw;

        std::cout << "Algorithm: CPU IVF-RaBitQ (parallel)\n"
                  << "topk: " << topk << "\n"
                  << "threads: " << thread_count << "\n"
                  << "rounds: " << test_rounds << "\n"
                  << "warmup_rounds: " << warmup_rounds << "\n"
                  << "hacc: " << (use_hacc ? "true" : "false") << "\n"
                  << "profile: " << (profile ? "true" : "false") << "\n"
                  << "nprobes: " << rabitq_benchmark::join_values(nprobes)
                  << '\n';

        const auto run_search = [&](std::size_t nprobe) {
#pragma omp parallel for schedule(static) num_threads(thread_count)
            for (std::size_t i = 0; i < nq; ++i) {
                ivf.search(
                    &query(i, 0), topk, nprobe,
                    all_results.data() + i * topk, use_hacc);
            }
        };

        for (std::size_t l = 0; l < nprobes.size(); ++l) {
            const std::size_t nprobe = nprobes[l];
            for (std::size_t warmup = 0; warmup < warmup_rounds; ++warmup) {
                run_search(nprobe);
            }

            for (std::size_t round = 0; round < test_rounds; ++round) {
                g_cpu_stats.reset();
                stopw.reset();
                run_search(nprobe);
                const float total_time = stopw.get_elapsed_micro();

                const auto n_scan = g_cpu_stats.total_1bit_scan.load(
                    std::memory_order_relaxed);
                const auto n_refine = g_cpu_stats.total_refine.load(
                    std::memory_order_relaxed);

                all_qps[round][l] =
                    static_cast<float>(nq) / (total_time / 1e6F);
                all_recall[round][l] = recall_evaluator.evaluate(all_results);
                std::cout << "[Run] round=" << (round + 1)
                    << " nprobe=" << nprobe
                    << " QPS=" << all_qps[round][l]
                    << " recall=" << all_recall[round][l];
                if (profile) {
                    std::cout
                        << " scanned=" << n_scan
                        << " refined=" << n_refine
                        << " refinement_ratio="
                        << (n_scan ? static_cast<double>(n_refine) /
                                         static_cast<double>(n_scan)
                                   : 0.0);
                }
                std::cout << '\n';
            }
        }

        const auto avg_qps = rabitqlib::horizontal_avg(all_qps);
        const auto avg_recall = rabitqlib::horizontal_avg(all_recall);
        std::cout << "nprobe\tQPS\trecall\n";
        for (std::size_t i = 0; i < nprobes.size(); ++i) {
            std::cout << nprobes[i] << '\t' << avg_qps[i] << '\t'
                      << avg_recall[i] << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "ivf_rabitq_querying_parallel failed: " << error.what()
                  << '\n';
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
