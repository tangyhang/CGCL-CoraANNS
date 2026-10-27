#include <iostream>
#include <algorithm>
#include <limits>
#include <vector>
#include <string>
#include <cfloat>
#include <queue>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <immintrin.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <threads.h>
#include <unistd.h>
#include <omp.h>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/io.hpp"
#include "coraanns/benchmark/recall.hpp"
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/utils/tools.hpp"
#include "coraanns/benchmark/options.hpp"

#include "coraanns/search/hybrid.cuh"

#define private public
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/quantization/rabitq_impl.hpp"
#undef private

using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

static size_t topk = 100;

namespace {

void print_usage(const char *program) {
    std::cerr << "Usage: " << program
              << " <index_file> <query_file> <gt_file>"
                 " [--topk 100] [--nprobes 40,50,...] [--rounds 5]"
                 " [--warmup 1] [--batch-q 4] [--query-chunk N] [--threads N]"
                 " [--vanguard-size 5] [--cpu-secondary-filter true]"
                 " [--cross-query-reuse true] [--candidate-stats false]"
                 " [--gpu-filter-timing false] [--profile false]"
                 " [--exact-base FILE] [--recall-base FILE] [--recall-once false]\n";
}
class ExactBaseMMap {
  public:
    explicit ExactBaseMMap(const std::string &path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            throw std::runtime_error("Cannot open exact-refine base file " + path + ": " +
                                     std::strerror(errno));
        }

        struct stat st {};
        if (::fstat(fd_, &st) != 0 || st.st_size <= 0) {
            const std::string message =
                "Cannot stat exact-refine base file " + path + ": " + std::strerror(errno);
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(message);
        }
        mapping_bytes_ = static_cast<size_t>(st.st_size);

        void *mapped = ::mmap(nullptr, mapping_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped == MAP_FAILED) {
            const std::string message =
                "Cannot mmap exact-refine base file " + path + ": " + std::strerror(errno);
            ::close(fd_);
            fd_ = -1;
            throw std::runtime_error(message);
        }
        mapping_ = static_cast<const uint8_t *>(mapped);

        const bool is_vecs = path.size() >= 5 && path.compare(path.size() - 5, 5, "fvecs") == 0;
        if (is_vecs) {
            if (mapping_bytes_ < sizeof(uint32_t)) {
                fail_format(path);
            }
            uint32_t file_dim = 0;
            std::memcpy(&file_dim, mapping_, sizeof(file_dim));
            dim_ = file_dim;
            data_offset_bytes_ = sizeof(uint32_t);
            row_stride_bytes_ = sizeof(uint32_t) + dim_ * sizeof(float);
            if (dim_ == 0 || mapping_bytes_ % row_stride_bytes_ != 0) {
                fail_format(path);
            }
            rows_ = mapping_bytes_ / row_stride_bytes_;
        } else {
            if (mapping_bytes_ < 2 * sizeof(uint32_t)) {
                fail_format(path);
            }
            uint32_t file_rows = 0;
            uint32_t file_dim = 0;
            std::memcpy(&file_rows, mapping_, sizeof(file_rows));
            std::memcpy(&file_dim, mapping_ + sizeof(uint32_t), sizeof(file_dim));
            rows_ = file_rows;
            dim_ = file_dim;
            data_offset_bytes_ = 2 * sizeof(uint32_t);
            row_stride_bytes_ = dim_ * sizeof(float);
            const size_t expected = data_offset_bytes_ + rows_ * row_stride_bytes_;
            if (rows_ == 0 || dim_ == 0 || expected != mapping_bytes_) {
                fail_format(path);
            }
        }

        // Candidate pids are effectively random with respect to the original
        // base layout.  Avoid aggressive sequential readahead for the 100M
        // diagnostic dataset.
        ::madvise(const_cast<uint8_t *>(mapping_), mapping_bytes_, MADV_RANDOM);
    }

    ExactBaseMMap(const ExactBaseMMap &) = delete;
    ExactBaseMMap &operator=(const ExactBaseMMap &) = delete;

    ~ExactBaseMMap() {
        if (mapping_ != nullptr) {
            ::munmap(const_cast<uint8_t *>(mapping_), mapping_bytes_);
        }
        if (fd_ >= 0) {
            ::close(fd_);
        }
    }

    const uint8_t *mapping() const {
        return mapping_;
    }
    size_t rows() const {
        return rows_;
    }
    size_t dim() const {
        return dim_;
    }
    size_t row_stride_bytes() const {
        return row_stride_bytes_;
    }
    size_t data_offset_bytes() const {
        return data_offset_bytes_;
    }

  private:
    [[noreturn]] void fail_format(const std::string &path) {
        if (mapping_ != nullptr) {
            ::munmap(const_cast<uint8_t *>(mapping_), mapping_bytes_);
            mapping_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
        throw std::runtime_error("Invalid exact-refine base format or file size: " + path);
    }

    int fd_ = -1;
    const uint8_t *mapping_ = nullptr;
    size_t mapping_bytes_ = 0;
    size_t rows_ = 0;
    size_t dim_ = 0;
    size_t row_stride_bytes_ = 0;
    size_t data_offset_bytes_ = 0;
};

} // namespace

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
            h_rand_row_normalized_abs[row_offset + j] =
                std::abs(h_rand_row_normalized_abs[row_offset + j] * inv_norm);
        }
    }
    double sum = 0.0;
    for (long j = 0; j < kConstNum; ++j) {
        sum += rabitqlib::quant::rabitq_impl::ex_bits::best_rescale_factor<double>(
            &h_rand_row_normalized_abs[j * dim], dim, ex_bits);
    }
    return static_cast<float>(sum / kConstNum);
}

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

        // Preserve the historical positional form: [topk] [batch_q]
        // [exact_base_file]. New experiment scripts should use named options.
        const bool legacy_mode = argc > 4 && !rabitq_benchmark::starts_with_option(argv[4]);
        std::size_t option_begin = 4;
        int batch_q = 4;
        std::string exact_base_file;
        if (legacy_mode) {
            if (argc > 7) {
                throw std::invalid_argument("Too many arguments in the legacy CoraANN form");
            }
            std::size_t consumed = 0;
            topk = std::stoull(argv[4], &consumed);
            if (consumed != std::string(argv[4]).size()) {
                throw std::invalid_argument("Invalid positional topk");
            }
            if (argc > 5)
                batch_q = std::stoi(argv[5]);
            if (argc > 6)
                exact_base_file = argv[6];
            option_begin = argc;
        }

        rabitq_benchmark::CliOptions options(argc, argv, option_begin);
        options.require_known({"topk", "nprobes", "rounds", "warmup", "batch-q", "threads",
                               "query-chunk", "vanguard-size", "cpu-secondary-filter", "profile",
                               "cross-query-reuse", "candidate-stats", "gpu-filter-timing",
                               "exact-base", "recall-once"});
        topk = options.get_size("topk", topk);
        const size_t test_round = options.get_size("rounds", 5);
        const size_t warmup_rounds = options.get_size("warmup", 1);
        const bool recall_once = options.get_bool("recall-once", false);
        const size_t requested_batch_q = options.get_size("batch-q", static_cast<size_t>(batch_q));
        if (requested_batch_q < 1 || requested_batch_q > 6) {
            throw std::invalid_argument("batch-q must be in [1, 6]");
        }
        batch_q = static_cast<int>(requested_batch_q);
        const size_t requested_threads =
            options.get_size("threads", static_cast<size_t>(omp_get_max_threads()));
        if (requested_threads == 0 ||
            requested_threads > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("threads must be a positive int");
        }
        const int cpu_threads = static_cast<int>(requested_threads);
        omp_set_dynamic(0);
        omp_set_num_threads(cpu_threads);
        const size_t vanguard_size = options.get_size("vanguard-size", DEFAULT_VANGUARD_SIZE);
        const bool profile = options.get_bool("profile", false);
        const bool candidate_stats = options.get_bool("candidate-stats", false);
        const bool gpu_filter_timing = options.get_bool("gpu-filter-timing", false);
        const bool cross_query_reuse = options.get_bool("cross-query-reuse", true);
        const bool cpu_secondary_filter = options.get_bool("cpu-secondary-filter", true);
        exact_base_file = options.get("exact-base", exact_base_file);

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
        const size_t query_chunk = std::min(requested_query_chunk, nq);

        std::cout << "Initializing GPU Index..." << std::endl;
        rabitqlib::ivf::IVF cpu_ivf;
        cpu_ivf.load(index_file.c_str());
        // In once mode even evaluator construction is deferred: computing the GT
        // distance thresholds reads base vectors and must not precede QPS rounds.
        std::unique_ptr<recall_eval::Evaluator> recall_evaluator;
        if (!recall_once) {
            recall_evaluator = std::make_unique<recall_eval::Evaluator>(
                query, gt, topk, recall_base, cpu_ivf.metric_type());
        }

        rabitq_benchmark::validate_benchmark_shape(query.rows(), query.cols(), gt.rows(), gt.cols(),
                                                   cpu_ivf.dim_, topk, test_round);

        std::unique_ptr<ExactBaseMMap> exact_base;
        if (!exact_base_file.empty()) {
            if (cpu_ivf.metric_type_ != rabitqlib::METRIC_L2) {
                std::cerr << "Exact-refine diagnostic currently supports only L2 indexes.\n";
                return 1;
            }
            try {
                exact_base = std::make_unique<ExactBaseMMap>(exact_base_file);
            } catch (const std::exception &e) {
                std::cerr << e.what() << '\n';
                return 1;
            }
            if (exact_base->rows() != cpu_ivf.num_ || exact_base->dim() != cpu_ivf.dim_) {
                std::cerr << "Exact-refine base shape mismatch: file=" << exact_base->rows() << 'x'
                          << exact_base->dim() << ", index=" << cpu_ivf.num_ << 'x' << cpu_ivf.dim_
                          << '\n';
                return 1;
            }
            std::cout << "Exact FP32 refine enabled with mmap base: " << exact_base_file << " ("
                      << exact_base->rows() << 'x' << exact_base->dim() << ")\n";
        }

        size_t num_cluster = cpu_ivf.num_cluster_;
        size_t padded_dim = cpu_ivf.padded_dim_;
        size_t num_words = padded_dim / 32;

        std::vector<float> h_centroids(num_cluster * padded_dim);
        std::vector<size_t> cluster_sizes(num_cluster);
        size_t max_cluster_size = 0;

        for (size_t i = 0; i < num_cluster; ++i) {
            std::memcpy(h_centroids.data() + i * padded_dim, cpu_ivf.initer_->centroid(i),
                        padded_dim * sizeof(float));
            size_t c_size = cpu_ivf.cluster_lst_[i].num();
            cluster_sizes[i] = c_size;
            if (c_size > max_cluster_size)
                max_cluster_size = c_size;
        }

        std::vector<uint32_t> host_short_data(cpu_ivf.num_ * num_words, 0);
        std::vector<float> host_short_factors(cpu_ivf.num_ * 3, 0.0f);

        size_t ex_bits = cpu_ivf.ex_bits_;
        std::cout << "ex_bits: " << ex_bits << std::endl;
        std::cout << "topk: " << topk << std::endl;
        std::cout << "batch_q: " << batch_q << std::endl;
        size_t long_code_size = (padded_dim * ex_bits + 7) / 8;
        std::vector<float> host_ex_factors;
        if (!exact_base) {
            host_ex_factors.resize(cpu_ivf.num_ * 2, 0.0f);
        }

        size_t global_vec_idx = 0;
        const int kPerm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
        int inv_kPerm0[16];
        for (int j = 0; j < 16; ++j)
            inv_kPerm0[kPerm0[j]] = j;

        for (size_t c = 0; c < num_cluster; ++c) {
            const auto &cluster = cpu_ivf.cluster_lst_[c];
            size_t n = cluster.num();
            size_t cluster_start_index = global_vec_idx;

            const char *batch_ptr = cluster.batch_data();
            for (size_t i = 0; i < n; i += 32) {
                size_t batch_len = std::min((size_t)32, n - i);
                const uint8_t *bin_code_bytes = reinterpret_cast<const uint8_t *>(batch_ptr);
                const float *batch_add =
                    reinterpret_cast<const float *>(batch_ptr + padded_dim * 4);
                const float *batch_rescale =
                    reinterpret_cast<const float *>(batch_ptr + padded_dim * 4 + 32 * 4);
                const float *batch_error =
                    reinterpret_cast<const float *>(batch_ptr + padded_dim * 4 + 64 * 4);

                for (size_t v = 0; v < batch_len; ++v) {
                    size_t current_global_idx = cluster_start_index + i + v;
                    for (size_t word = 0; word < num_words; ++word) {
                        uint32_t gpu_word = 0;
                        for (int bit = 0; bit < 32; ++bit) {
                            size_t dim = word * 32 + bit;
                            size_t dim_in_group = dim % 8;
                            size_t offset_half = (dim_in_group < 4) ? 0 : 16;
                            uint8_t byte_val =
                                bin_code_bytes[(dim / 8) * 32 + offset_half + inv_kPerm0[v % 16]];
                            uint8_t nibble = (v < 16) ? (byte_val & 0x0F) : (byte_val >> 4);
                            gpu_word |= (((nibble >> (3 - (dim_in_group % 4))) & 1) << (31 - bit));
                        }
                        host_short_data[cluster_start_index * num_words + word * n + i + v] =
                            gpu_word;
                    }
                    host_short_factors[current_global_idx * 3 + 0] = batch_add[v];
                    host_short_factors[current_global_idx * 3 + 1] = batch_rescale[v];
                    host_short_factors[current_global_idx * 3 + 2] = batch_error[v];
                }
                batch_ptr += (padded_dim * 4 + 32 * 12);
            }
            global_vec_idx += n;
        }

        // Preserve the original query path and its contiguous long-code layout,
        // but avoid allocating a second full-size code array.  Save the two
        // factors, then compact the existing interleaved ExDataMap allocation in
        // place using a bounded staging chunk.  For Wiki-88M this removes the
        // roughly 59 GiB duplicate while adding only ~0.66 GiB of factors and a
        // 192 MiB staging buffer during initialization.
        uint8_t *host_long_codes = nullptr;
        if (!exact_base) {
            const size_t ex_stride = rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);
            if (long_code_size == 0 || ex_stride < long_code_size + 2 * sizeof(float) ||
                cpu_ivf.ex_data_ == nullptr) {
                throw std::runtime_error("invalid extended-code layout");
            }

            constexpr size_t kCompactStagingBytes = size_t{192} << 20;
            const size_t compact_chunk_vectors =
                std::max(size_t{1}, std::min(cpu_ivf.num_, kCompactStagingBytes / long_code_size));
            std::vector<uint8_t> compact_codes(compact_chunk_vectors * long_code_size);
            auto *ex_data = reinterpret_cast<uint8_t *>(cpu_ivf.ex_data_);

#pragma omp parallel num_threads(cpu_threads)
            {
                for (size_t begin = 0; begin < cpu_ivf.num_; begin += compact_chunk_vectors) {
                    const size_t count = std::min(compact_chunk_vectors, cpu_ivf.num_ - begin);
#pragma omp for schedule(static)
                    for (long long offset = 0; offset < static_cast<long long>(count); ++offset) {
                        const size_t vector_index = begin + static_cast<size_t>(offset);
                        const uint8_t *record = ex_data + vector_index * ex_stride;
                        std::memcpy(compact_codes.data() +
                                        static_cast<size_t>(offset) * long_code_size,
                                    record, long_code_size);
                        const float *factors =
                            reinterpret_cast<const float *>(record + long_code_size);
                        host_ex_factors[vector_index * 2 + 0] = factors[0];
                        host_ex_factors[vector_index * 2 + 1] = factors[1];
                    }
#pragma omp single
                    std::memcpy(ex_data + begin * long_code_size, compact_codes.data(),
                                count * long_code_size);
                }
            }
            host_long_codes = ex_data;
            std::cout << "Long-code memory: compacted in place; avoided "
                      << (cpu_ivf.num_ * long_code_size / (1024.0 * 1024.0 * 1024.0))
                      << " GiB duplicate\n";
        }

        std::vector<float> host_rotator(dim * padded_dim);
        cpu_ivf.rotator_->save(reinterpret_cast<char *>(host_rotator.data()));

        IVF_RaBitQ_Raw gpu_ivf;
        gpu_ivf.set_profiling(profile);
        gpu_ivf.set_candidate_stats(candidate_stats);
        gpu_ivf.set_gpu_filter_timing(gpu_filter_timing);
        gpu_ivf.set_cross_query_reuse(cross_query_reuse);
        gpu_ivf.set_cpu_secondary_filter(cpu_secondary_filter);
        gpu_ivf.set_vanguard_size(vanguard_size);
        gpu_ivf.load_from_raw_pointers(cpu_ivf.num_, cpu_ivf.dim_, padded_dim, num_cluster,
                                       cpu_ivf.ex_bits_, h_centroids.data(), host_rotator.data(),
                                       host_short_data.data(), host_short_factors.data(),
                                       host_long_codes, host_ex_factors.data(), cpu_ivf.ids_,
                                       cluster_sizes.data());

        if (exact_base) {
            gpu_ivf.enable_exact_refine(exact_base->mapping(), exact_base->rows(),
                                        exact_base->dim(), exact_base->row_stride_bytes(),
                                        exact_base->data_offset_bytes());
        }

        gpu_ivf.max_cluster_size_ = max_cluster_size;
        std::cout << "Max cluster size: " << max_cluster_size << std::endl;
        gpu_ivf.best_rescaling_factor = get_const_scaling_factors(padded_dim, cpu_ivf.ex_bits_);

        const auto nprobes =
            rabitq_benchmark::parse_nprobes(options.get("nprobes", ""), num_cluster);

        size_t length = nprobes.size();

        std::cout << "Allocating GPU Workspace pool..." << std::endl;
        const size_t max_nprobe = *std::max_element(nprobes.begin(), nprobes.end());
        gpu_ivf.prepare_workspace(query_chunk, topk, max_nprobe);

        std::cout << "Algorithm: CoraANN (GPU filtering + CPU refinement)\n"
                  << "topk: " << topk << "\n"
                  << "batch_q: " << batch_q << "\n"
                  << "query_chunk: " << query_chunk << "\n"
                  << "cpu_threads: " << cpu_threads << "\n"
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
                    // Preserve the first measured search, outside its timer. No
                    // base reads or recall worker pool run between QPS rounds.
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
        std::cerr << "ivf_rabitq_querying_gpu_cpu failed: " << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
