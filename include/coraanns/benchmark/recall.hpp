#pragma once

// Evaluation only. Reuse the existing read-only, demand-paged FP32 base view;
// no base vectors are copied into RAM or passed to the search implementation.
#include "coraanns/benchmark/distance.hpp"

#include <iostream>
#include <exception>
#include <memory>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace recall_eval {

// Remove only this evaluation option before the existing CLI parses argv.
// This also supports the old positional commands without changing their rules.
inline std::string take_base_option(int& argc, char** argv) {
    std::string base;
    int output = 1;
    bool found = false;
    for (int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if (argument == "--recall-base" || argument.rfind("--recall-base=", 0) == 0) {
            if (found) throw std::invalid_argument("Duplicate --recall-base");
            found = true;
            if (argument == "--recall-base") {
                if (i + 1 == argc || std::string(argv[i + 1]).rfind("--", 0) == 0) {
                    throw std::invalid_argument("Missing file for --recall-base");
                }
                base = argv[++i];
            } else {
                base = argument.substr(14);
            }
            if (base.empty()) throw std::invalid_argument("Empty --recall-base");
        } else {
            argv[output++] = argv[i];
        }
    }
    argc = output;
    argv[argc] = nullptr;
    if (found && argc < 4) {
        throw std::invalid_argument("--recall-base does not replace index/query/GT arguments");
    }
    return base;
}

struct Counts {
    std::size_t total = 0;
    std::size_t id_hits = 0;
    std::size_t distance_hits = 0;  // Additional hits, excluding ID matches.
    std::size_t exact_checks = 0;
    std::size_t cache_hits = 0;
    double milliseconds = 0.0;

    std::size_t correct() const { return id_hits + distance_hits; }
    Counts& operator+=(const Counts& other) {
        total += other.total;
        id_hits += other.id_hits;
        distance_hits += other.distance_hits;
        exact_checks += other.exact_checks;
        cache_hits += other.cache_hits;
        milliseconds += other.milliseconds;
        return *this;
    }
};

class Evaluator {
public:
    template <typename QueryMatrix, typename GtMatrix>
    Evaluator(const QueryMatrix& queries, const GtMatrix& gt, std::size_t topk,
              const std::string& base_file,
              rabitqlib::MetricType metric = rabitqlib::METRIC_L2)
        : Evaluator(queries.data(), queries.rows(), queries.cols(), gt, topk,
                    base_file, metric) {}

    template <typename GtMatrix>
    Evaluator(const float* queries, std::size_t query_count, std::size_t dim,
              const GtMatrix& gt, std::size_t topk, const std::string& base_file,
              rabitqlib::MetricType metric = rabitqlib::METRIC_L2)
        : queries_(queries), nq_(query_count), dim_(dim), topk_(topk), metric_(metric) {
        if (queries == nullptr || nq_ == 0 || dim_ == 0 || topk_ == 0 ||
            static_cast<std::size_t>(gt.rows()) != nq_ ||
            static_cast<std::size_t>(gt.cols()) < topk_) {
            throw std::invalid_argument("Invalid query/GT shape for recall@k");
        }
        if (metric_ != rabitqlib::METRIC_L2 && metric_ != rabitqlib::METRIC_IP) {
            throw std::invalid_argument("Recall evaluation supports only L2 and IP");
        }
        if (!base_file.empty()) {
            base_ = std::make_unique<graph_baseline::FloatVectorMMap>(base_file);
            if (base_->dim() != dim_) {
                throw std::invalid_argument("Recall base and query dimensions differ");
            }
        }
        truth_.resize(nq_ * topk_);
        thresholds_.resize(nq_);
        cache_.resize(nq_);
        for (std::size_t q = 0; q < nq_; ++q) {
            auto begin = truth_.begin() + q * topk_;
            for (std::size_t k = 0; k < topk_; ++k) {
                const auto id = gt(q, k);
                if (id == std::numeric_limits<std::uint32_t>::max() ||
                    (base_ && id >= base_->rows())) {
                    throw std::invalid_argument("Invalid GT ID for recall base");
                }
                begin[k] = id;
            }
            // Use rank K BEFORE sorting IDs for fast membership tests.
            if (base_) {
                thresholds_[q] = exact_score(q, gt(q, topk_ - 1));
                if (!finite(thresholds_[q])) {
                    throw std::invalid_argument("Non-finite query/GT distance");
                }
            }
            std::sort(begin, begin + topk_);
            if (std::adjacent_find(begin, begin + topk_) != begin + topk_) {
                throw std::invalid_argument("GT top-k contains duplicate IDs");
            }
        }
        std::cout << "[Recall] mode=" << (base_ ? "distance-aware" : "id")
                  << " metric=" << (metric_ == rabitqlib::METRIC_L2 ? "l2" : "ip")
                  << " topk=" << topk_ << " base="
                  << (base_ ? base_file : "none (use --recall-base for distance checks)")
                  << "\n";
    }

    // Different queries may be evaluated concurrently, but the same query
    // must not be evaluated by two threads at once (its cache is query-local).
    template <typename Id>
    Counts count_query(std::size_t q, const Id* results, std::size_t count) {
        static_assert(std::is_integral_v<Id>, "Result IDs must be integers");
        if (q >= nq_) throw std::out_of_range("Recall query is out of range");
        const auto started = std::chrono::steady_clock::now();
        Counts counts;
        counts.total = topk_;  // Missing/invalid results remain misses.
        std::vector<std::uint32_t> ids;
        ids.reserve(std::min(count, topk_));
        for (std::size_t k = 0; k < std::min(count, topk_); ++k) {
            if constexpr (std::is_signed_v<Id>) {
                if (results[k] < 0) continue;
            }
            const auto wide_id = static_cast<std::uint64_t>(results[k]);
            if (wide_id >= std::numeric_limits<std::uint32_t>::max() ||
                (base_ && wide_id >= base_->rows())) continue;
            ids.push_back(static_cast<std::uint32_t>(wide_id));
        }
        // Do not give repeated occurrences of the SAME ID multiple credits.
        // Distinct IDs with identical coordinates are still distinct results.
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        const auto truth_begin = truth_.begin() + q * topk_;
        for (const auto id : ids) {
            if (std::binary_search(truth_begin, truth_begin + topk_, id)) {
                ++counts.id_hits;
                continue;  // No vector access or true-distance calculation.
            }
            if (!base_) continue;
            auto& cache = cache_[q];
            const auto cached = cache.find(id);
            bool accepted;
            if (cached != cache.end()) {
                accepted = cached->second;
                ++counts.cache_hits;
            } else {
                const double score = exact_score(q, id);
                // No epsilon: near neighbors outside the GT radius are misses.
                // For IP, a larger exact inner product is better.
                accepted = finite(score) && (metric_ == rabitqlib::METRIC_L2
                    ? score <= thresholds_[q] : score >= thresholds_[q]);
                ++counts.exact_checks;
                // Bound memory in very long parameter sweeps. Cache decisions,
                // not raw vectors; eviction is unnecessary for correctness.
                if (cache.size() < 4096) cache.emplace(id, accepted);
            }
            counts.distance_hits += accepted;
        }
        counts.milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return counts;
    }

    double report(const Counts& counts) const {
        if (counts.total == 0) throw std::invalid_argument("Empty recall results");
        const double denominator = static_cast<double>(counts.total);
        const double recall = counts.correct() / denominator;
        std::cout << "[RecallEval] id_recall=" << counts.id_hits / denominator
                  << " recall=" << recall
                  << " distance_hits=" << counts.distance_hits
                  << " exact_checks=" << counts.exact_checks
                  << " cache_hits=" << counts.cache_hits
                  << " eval_ms=" << counts.milliseconds << '\n';
        return recall;
    }

    template <typename Id>
    double evaluate(const std::vector<Id>& results) {
        if (results.size() < nq_ * topk_) {
            throw std::invalid_argument("Result array is smaller than nq * topk");
        }
        const auto started = std::chrono::steady_clock::now();
        std::vector<Counts> per_query(nq_);
        std::exception_ptr error;
#ifdef _OPENMP
        // Limit concurrent page faults on cold base files; this evaluation
        // pool is entered only AFTER the search timer has stopped.
        const int workers = std::min(8, omp_get_max_threads());
#pragma omp parallel for schedule(static) num_threads(workers) if(base_ && nq_ >= 128)
#endif
        for (std::size_t q = 0; q < nq_; ++q) {
            try {
                per_query[q] = count_query(q, results.data() + q * topk_, topk_);
            } catch (...) {
#ifdef _OPENMP
#pragma omp critical(recall_evaluation_error)
#endif
                { if (!error) error = std::current_exception(); }
            }
        }
        if (error) std::rethrow_exception(error);
        Counts counts;
        for (const auto& query : per_query) counts += query;
        // Report wall time, not the sum of parallel workers' elapsed times.
        counts.milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        return report(counts);
    }

private:
    // Bit test also works when the benchmarks are compiled with -Ofast,
    // which can otherwise optimize away std::isfinite checks.
    static bool finite(double value) {
        std::uint64_t bits;
        std::memcpy(&bits, &value, sizeof(bits));
        return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
    }

    double exact_score(std::size_t q, std::uint32_t id) const {
        const auto* query = queries_ + q * dim_;
        const auto* vector = base_->row(id);
        double score = 0.0;
        if (metric_ == rabitqlib::METRIC_L2) {
            for (std::size_t d = 0; d < dim_; ++d) {
                const double delta = static_cast<double>(query[d]) - vector[d];
                score += delta * delta;
            }
        } else {
            for (std::size_t d = 0; d < dim_; ++d) {
                score += static_cast<double>(query[d]) * vector[d];
            }
        }
        return score;
    }

    const float* queries_;
    std::size_t nq_, dim_, topk_;
    rabitqlib::MetricType metric_;
    std::unique_ptr<graph_baseline::FloatVectorMMap> base_;
    std::vector<std::uint32_t> truth_;
    std::vector<double> thresholds_;
    std::vector<std::unordered_map<std::uint32_t, bool>> cache_;
};

}  // namespace recall_eval
