#pragma once

#include <filesystem>
#include <iostream>
#include <limits>
#include <omp.h>

#include "coraanns/benchmark/options.hpp"

namespace rabitq_benchmark {

// Shared CLI for the two mechanism experiments. Querying baselines keep their
// existing interfaces and defaults.
struct ExperimentOptions {
    CliOptions cli;
    std::size_t topk;
    std::size_t rounds;
    std::size_t warmup;
    std::size_t vanguard_size;
    int batch_q;
    int threads;
    bool cpu_secondary_filter;
    bool profile;
    std::filesystem::path output_dir;

    ExperimentOptions(int argc, char** argv, std::size_t default_vanguard)
        : cli(argc, argv,
              argc > 4 && !starts_with_option(argv[4]) ? 5 : 4) {
        cli.require_known({"topk", "nprobes", "rounds", "warmup", "batch-q",
                           "threads", "vanguard-size", "cpu-secondary-filter",
                           "profile", "output-dir"});
        std::size_t legacy_topk = 100;
        if (argc > 4 && !starts_with_option(argv[4])) {
            if (argc != 5) {
                throw std::invalid_argument("Use either positional topk or named options");
            }
            const std::string value = argv[4];
            std::size_t consumed = 0;
            legacy_topk = std::stoull(value, &consumed);
            if (value.empty() || value[0] == '-' || value[0] == '+' ||
                consumed != value.size()) {
                throw std::invalid_argument("Invalid positional topk");
            }
        }
        topk = cli.get_size("topk", legacy_topk);
        rounds = cli.get_size("rounds", 1);
        warmup = cli.get_size("warmup", 1);
        vanguard_size = cli.get_size("vanguard-size", default_vanguard);
        const auto batch = cli.get_size("batch-q", 4);
        const auto workers = cli.get_size("threads", 72);
        if (topk == 0 || rounds == 0) {
            throw std::invalid_argument("topk and rounds must be positive");
        }
        if (batch < 1 || batch > 6) {
            throw std::invalid_argument("batch-q must be in [1, 6]");
        }
        if (workers == 0 || workers > static_cast<std::size_t>(
                                         std::numeric_limits<int>::max())) {
            throw std::invalid_argument("threads must be a positive int");
        }
        batch_q = static_cast<int>(batch);
        threads = static_cast<int>(workers);
        cpu_secondary_filter = cli.get_bool("cpu-secondary-filter", true);
        profile = cli.get_bool("profile", false);
        output_dir = cli.get("output-dir", ".");
        if (output_dir.empty()) throw std::invalid_argument("output-dir must not be empty");
    }

    template <typename Index>
    void configure(Index& index) const {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
        index.set_vanguard_size(vanguard_size);
        index.set_cpu_secondary_filter(cpu_secondary_filter);
        index.set_profiling(profile);
        std::filesystem::create_directories(output_dir);
    }

    void print(const char* name, const std::vector<std::size_t>& nprobes) const {
        std::cout << "Algorithm: " << name << '\n'
                  << "topk: " << topk << '\n'
                  << "rounds: " << rounds << '\n'
                  << "warmup_rounds: " << warmup << '\n'
                  << "batch_q: " << batch_q << '\n'
                  << "cpu_threads: " << threads << '\n'
                  << "vanguard_size: " << vanguard_size << '\n'
                  << "cpu_secondary_filter: " << (cpu_secondary_filter ? "true" : "false") << '\n'
                  << "profile: " << (profile ? "true" : "false") << '\n'
                  << "nprobes: " << join_values(nprobes) << '\n'
                  << "output_dir: " << output_dir.string() << '\n';
    }
};

inline void print_experiment_usage(const char* program) {
    std::cerr << "Usage: " << program << " <index_file> <query_file> <gt_file>"
              << " [--topk 100] [--nprobes 123] [--rounds 1] [--warmup 1]"
              << " [--batch-q 4] [--threads 72] [--vanguard-size N]"
              << " [--cpu-secondary-filter true] [--profile false]"
              << " [--output-dir .] [--recall-base FILE]\n";
}

}  // namespace rabitq_benchmark
