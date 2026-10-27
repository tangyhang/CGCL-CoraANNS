#include <iostream>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
#include <string>

// 引入 rabitqlib 的基础工具（用于加载 .fvecs/.ivecs 和计时）
#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/io.hpp"
#include "coraanns/benchmark/recall.hpp"
#include "rabitqlib/utils/stopw.hpp"
#include "rabitqlib/utils/tools.hpp"
#include "coraanns/benchmark/options.hpp"

// 引入我们自己写的 CUDA 包装类头文件
#include "coraanns/baselines/gpu.cuh"

#define private public
#include "rabitqlib/index/ivf/ivf.hpp"
#include "rabitqlib/quantization/rabitq_impl.hpp"
#undef private

using data_type = rabitqlib::RowMajorArray<float>;
using gt_type = rabitqlib::RowMajorArray<uint32_t>;

static size_t topk = 100;

namespace {

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " <index_file> <query_file> <gt_file>"
           " [--topk 100] [--nprobes 40,50,...] [--rounds 5]"
           " [--warmup 1] [--profile false] [--memory-breakdown false]"
           " [--isolated-profile false] [--isolated-profile-rounds 5]"
           " [--index-residency hbm|host-excode|auto-cache]"
           " [--hbm-reserve-mib 768] [--hbm-cache-mib 0]"
           " [--query-chunk N]"
           " [--fine-kernel auto|tiled]"
           " [--cluster-value-output FILE] [--recall-base FILE]"
           " [--recall-once false]\n";
}

double mib(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

void print_hbm_breakdown(
    size_t num_vectors,
    size_t dim,
    size_t padded_dim,
    size_t num_centroids,
    size_t ex_bits,
    size_t max_queries,
    size_t max_topk,
    size_t max_nprobe,
    bool host_backed_ex_codes,
    size_t cached_ex_bytes) {
    const auto old_flags = std::cout.flags();
    const auto old_precision = std::cout.precision();
    const size_t one_bit_codes =
        num_vectors * (padded_dim / 32) * sizeof(uint32_t);
    const size_t ex_codes =
        ex_bits == 0
            ? 0
            : num_vectors * ((padded_dim * ex_bits + 7) / 8);
    const size_t centroids = num_centroids * padded_dim * sizeof(float);
    const size_t rotator = dim * padded_dim * sizeof(float);
    const size_t short_factors = num_vectors * 3 * sizeof(float);
    const size_t ex_factors = ex_bits == 0 ? 0 : num_vectors * 2 * sizeof(float);
    const size_t ids = num_vectors * sizeof(uint32_t);
    const size_t cluster_meta = num_centroids * sizeof(GPUClusterMeta);
    const size_t other_index = centroids + rotator + short_factors +
                               ex_factors + ids + cluster_meta;
    const size_t index_total = one_bit_codes + ex_codes + other_index;
    const size_t host_extended_data =
        host_backed_ex_codes ? ex_codes + ex_factors : 0;
    const size_t hbm_index =
        index_total - host_extended_data + cached_ex_bytes;
    const size_t host_mapped_index = host_extended_data;

    const size_t total_probes = max_queries * max_nprobe;
    const size_t total_candidates = total_probes * max_topk;
    const size_t centroid_matrix_items = max_queries * num_centroids;
    const size_t lut_elements = max_queries * (padded_dim / 4) * 16;
    const size_t runtime_workspace =
        max_queries * sizeof(float) +                         // threshold
        max_queries * padded_dim * sizeof(float) +            // rotated queries
        max_queries * dim * sizeof(float) +                   // raw queries
        centroid_matrix_items * sizeof(float) +               // centroid distances
        max_queries * sizeof(float) +                         // query norms
        num_centroids * sizeof(float) +                       // centroid norms
        total_probes * (sizeof(float) + sizeof(int)) +        // probes
        total_probes * sizeof(ClusterQueryPair) +             // sorted pairs
        2 * max_queries * sizeof(float) +                     // query factors
        total_candidates * (sizeof(float) + sizeof(uint32_t)) +
        max_queries * sizeof(int) +                           // candidate counters
        max_queries * max_topk * (sizeof(float) + sizeof(uint32_t)) +
        lut_elements * sizeof(half);

    const auto print_index_row = [&](const char* name, size_t bytes) {
        const double percent = index_total
            ? 100.0 * static_cast<double>(bytes) /
                  static_cast<double>(index_total)
            : 0.0;
        std::cout << "[HBM-Index]\t" << name << '\t' << bytes << '\t'
                  << mib(bytes) << '\t' << percent << '\n';
    };

    std::cout << std::fixed << std::setprecision(3)
              << "[HBM-Index]\tcomponent\tbytes\tMiB\tpercent_of_index\n";
    print_index_row("1-bit_codes", one_bit_codes);
    print_index_row("ex-codes", ex_codes);
    print_index_row("other_index", other_index);
    print_index_row("total", index_total);
    std::cout << "[HBM-Detail]\tcentroids\t" << centroids << '\t'
              << mib(centroids) << '\n'
              << "[HBM-Detail]\trotator\t" << rotator << '\t'
              << mib(rotator) << '\n'
              << "[HBM-Detail]\t1-bit_factors\t" << short_factors << '\t'
              << mib(short_factors) << '\n'
              << "[HBM-Detail]\tex-code_factors\t" << ex_factors << '\t'
              << mib(ex_factors) << '\n'
              << "[HBM-Detail]\tids\t" << ids << '\t' << mib(ids) << '\n'
              << "[HBM-Detail]\tcluster_metadata\t" << cluster_meta << '\t'
              << mib(cluster_meta) << '\n'
              << "[HBM-Runtime]\tworkspace_requested\t" << runtime_workspace
              << '\t' << mib(runtime_workspace) << '\n'
              << "[HBM-Resident]\tindex\t" << hbm_index << '\t'
              << mib(hbm_index) << '\n'
              << "[HBM-Resident]\tindex_plus_workspace\t"
              << (hbm_index + runtime_workspace) << '\t'
              << mib(hbm_index + runtime_workspace) << '\n'
              << "[Host-Mapped]\textended_codes_and_factors\t"
              << host_mapped_index << '\t'
              << mib(host_mapped_index) << '\n';
    size_t device_free_bytes = 0, device_total_bytes = 0;
    const cudaError_t memory_error = cudaMemGetInfo(
        &device_free_bytes, &device_total_bytes);
    if (memory_error != cudaSuccess) {
        throw std::runtime_error(
            std::string("Cannot inspect GPU memory before search: ") +
            cudaGetErrorString(memory_error));
    }
    std::cout << "[HBM-Device]\tfree_before_search\t" << device_free_bytes
              << '\t' << mib(device_free_bytes) << '\n'
              << "[HBM-Device]\ttotal\t" << device_total_bytes
              << '\t' << mib(device_total_bytes) << '\n';
    std::cout.flags(old_flags);
    std::cout.precision(old_precision);
}

void write_cluster_value_profile(
    const std::filesystem::path& output_path,
    const uint32_t* flattened_ids,
    size_t num_vectors,
    const std::vector<size_t>& cluster_sizes,
    const std::vector<int>& selected_clusters,
    const std::vector<uint32_t>& result_pids,
    size_t num_queries,
    size_t topk,
    size_t nprobe) {
    if (selected_clusters.size() != num_queries * nprobe ||
        result_pids.size() != num_queries * topk) {
        throw std::invalid_argument("Invalid buffers for cluster-value profile");
    }

    // Deep datasets use dense original IDs. This reverse map attributes every
    // returned neighbor to its IVF cluster after the timed GPU search.
    const uint32_t missing = std::numeric_limits<uint32_t>::max();
    std::vector<uint32_t> pid_to_cluster(num_vectors, missing);
    size_t offset = 0;
    for (size_t cluster = 0; cluster < cluster_sizes.size(); ++cluster) {
        for (size_t local = 0; local < cluster_sizes[cluster]; ++local) {
            const uint32_t pid = flattened_ids[offset + local];
            if (pid >= num_vectors) {
                throw std::runtime_error(
                    "Cluster-value profile requires dense PIDs in [0, N)");
            }
            pid_to_cluster[pid] = static_cast<uint32_t>(cluster);
        }
        offset += cluster_sizes[cluster];
    }
    if (offset != num_vectors) {
        throw std::runtime_error("IVF cluster sizes do not sum to index size");
    }

    std::vector<double> result_ratio_sum(nprobe, 0.0);
    std::vector<double> vector_ratio_sum(nprobe, 0.0);
    std::vector<int> cluster_rank(cluster_sizes.size(), -1);
    std::vector<size_t> result_histogram(nprobe, 0);

    for (size_t query_id = 0; query_id < num_queries; ++query_id) {
        size_t selected_vector_count = 0;
        for (size_t rank = 0; rank < nprobe; ++rank) {
            const int cluster = selected_clusters[query_id * nprobe + rank];
            if (cluster < 0 ||
                static_cast<size_t>(cluster) >= cluster_sizes.size()) {
                throw std::runtime_error("GPU returned an invalid cluster ID");
            }
            cluster_rank[cluster] = static_cast<int>(rank);
            selected_vector_count += cluster_sizes[cluster];
            result_histogram[rank] = 0;
        }

        for (size_t result_rank = 0; result_rank < topk; ++result_rank) {
            const uint32_t pid = result_pids[query_id * topk + result_rank];
            if (pid == missing || pid >= num_vectors ||
                pid_to_cluster[pid] == missing) {
                continue;
            }
            const int rank = cluster_rank[pid_to_cluster[pid]];
            if (rank >= 0) ++result_histogram[rank];
        }

        size_t prefix_results = 0;
        size_t prefix_vectors = 0;
        for (size_t rank = 0; rank < nprobe; ++rank) {
            const int cluster = selected_clusters[query_id * nprobe + rank];
            prefix_results += result_histogram[rank];
            prefix_vectors += cluster_sizes[cluster];
            result_ratio_sum[rank] +=
                static_cast<double>(prefix_results) / static_cast<double>(topk);
            vector_ratio_sum[rank] += selected_vector_count
                ? static_cast<double>(prefix_vectors) /
                      static_cast<double>(selected_vector_count)
                : 0.0;
            cluster_rank[cluster] = -1;
        }
    }

    if (!output_path.parent_path().empty()) {
        std::filesystem::create_directories(output_path.parent_path());
    }
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error(
            "Cannot open cluster-value output: " + output_path.string());
    }
    output << "m,mean_topk_contribution_ratio,mean_scanned_vector_ratio\n";
    output << std::setprecision(10);
    for (size_t rank = 0; rank < nprobe; ++rank) {
        output << (rank + 1) << ','
               << result_ratio_sum[rank] / static_cast<double>(num_queries)
               << ','
               << vector_ratio_sum[rank] / static_cast<double>(num_queries)
               << '\n';
    }
    std::cout << "[ClusterValue] wrote " << nprobe << " prefix rows to "
              << output_path << '\n';
}

}  // namespace

float get_const_scaling_factors(size_t dim, size_t ex_bits)
{
    constexpr long kConstNum = 100;
    
    // 使用一维 vector 模拟 100 x dim 的二维矩阵
    std::vector<double> h_rand_row_normalized_abs(kConstNum * dim);

    // 1. 初始化随机数生成器 (完全对齐 raft::random::RngState(7ULL))
    std::mt19937_64 rng(7ULL); 
    // 标准正态分布 N(0, 1)
    std::normal_distribution<double> normal_dist(0.0, 1.0);

    // 2. 生成正态分布矩阵，并进行行级别的 L2 归一化和绝对值操作
    for (long i = 0; i < kConstNum; ++i) {
        double sq_sum = 0.0;
        size_t row_offset = i * dim;
        
        // 步骤 A: 生成高斯噪声，并累计平方和
        for (size_t j = 0; j < dim; ++j) {
            double val = normal_dist(rng);
            h_rand_row_normalized_abs[row_offset + j] = val;
            sq_sum += val * val;
        }

        // 步骤 B: 计算当前行的 L2 范数
        double norm = std::sqrt(sq_sum);
        double inv_norm = (norm > 0.0) ? (1.0 / norm) : 0.0;

        // 步骤 C: 归一化，并取绝对值
        for (size_t j = 0; j < dim; ++j) {
            h_rand_row_normalized_abs[row_offset + j] = 
                std::abs(h_rand_row_normalized_abs[row_offset + j] * inv_norm);
        }
    }

    // 3. 遍历这 100 个超球面样本，计算最优缩放因子并求和
    double sum = 0.0;
    for (long j = 0; j < kConstNum; ++j) {
        sum += rabitqlib::quant::rabitq_impl::ex_bits::best_rescale_factor<double>(&h_rand_row_normalized_abs[j * dim], dim, ex_bits);
    }

    // 4. 求平均值得到统计学常数
    double t_const = sum / kConstNum;

    return static_cast<float>(t_const);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    try {
    const auto recall_base = recall_eval::take_base_option(argc, argv);
    std::string index_file = argv[1];
    std::string query_file = argv[2];
    std::string gt_file = argv[3];

    // Keep the old optional positional topk form for existing scripts.
    const bool legacy_mode =
        argc > 4 && !rabitq_benchmark::starts_with_option(argv[4]);
    std::size_t option_begin = 4;
    if (legacy_mode) {
        if (argc != 5) {
            throw std::invalid_argument(
                "The legacy GPU form accepts only one positional topk value");
        }
        std::size_t consumed = 0;
        topk = std::stoull(argv[4], &consumed);
        if (consumed != std::string(argv[4]).size()) {
            throw std::invalid_argument("Invalid positional topk");
        }
        option_begin = 5;
    }
    rabitq_benchmark::CliOptions options(argc, argv, option_begin);
    options.require_known(
        {"topk", "nprobes", "rounds", "warmup", "profile",
         "memory-breakdown", "isolated-profile",
         "isolated-profile-rounds", "index-residency",
         "query-chunk", "fine-kernel", "cluster-value-output", "recall-once",
         "hbm-reserve-mib", "hbm-cache-mib"});
    topk = options.get_size("topk", topk);
    if (topk > RABITQ_GPU_FINE_CAPACITY) {
        throw std::invalid_argument(
            "GPU-native tiled fine search supports topk <= " +
            std::to_string(RABITQ_GPU_FINE_CAPACITY));
    }
    const size_t test_round = options.get_size("rounds", 5);
    const size_t warmup_rounds = options.get_size("warmup", 1);
    const bool recall_once = options.get_bool("recall-once", false);
    const bool profile = options.get_bool("profile", false);
    const bool isolated_profile =
        options.get_bool("isolated-profile", false);
    const size_t isolated_profile_rounds =
        options.get_size("isolated-profile-rounds", 5);
    if (isolated_profile && !profile) {
        throw std::invalid_argument(
            "--isolated-profile true requires --profile true");
    }
    if (isolated_profile_rounds == 0) {
        throw std::invalid_argument(
            "--isolated-profile-rounds must be positive");
    }
    const bool memory_breakdown =
        options.get_bool("memory-breakdown", false);
    const std::string index_residency =
        options.get("index-residency", "hbm");
    const std::string fine_kernel =
        options.get("fine-kernel", "auto");
    const std::filesystem::path cluster_value_output =
        options.get("cluster-value-output", "");
    if (index_residency != "hbm" && index_residency != "host-excode" &&
        index_residency != "auto-cache") {
        throw std::invalid_argument(
            "--index-residency must be hbm, host-excode or auto-cache");
    }
    if (fine_kernel != "auto" && fine_kernel != "tiled") {
        throw std::invalid_argument(
            "--fine-kernel must be auto or tiled");
    }
    const bool host_backed_ex_codes = index_residency != "hbm";
    const size_t hbm_reserve_mib = options.get_size("hbm-reserve-mib", 768);
    const size_t hbm_cache_mib = options.get_size("hbm-cache-mib", 0);
    if (hbm_reserve_mib > (std::numeric_limits<size_t>::max() >> 20) ||
        hbm_cache_mib > (std::numeric_limits<size_t>::max() >> 20)) {
        throw std::invalid_argument("HBM MiB value is too large");
    }
    if (index_residency != "auto-cache" &&
        (options.contains("hbm-reserve-mib") || options.contains("hbm-cache-mib"))) {
        throw std::invalid_argument("HBM cache options require --index-residency auto-cache");
    }

    if (host_backed_ex_codes) {
        int device = 0;
        cudaDeviceProp properties{};
        const cudaError_t device_error = cudaGetDevice(&device);
        if (device_error != cudaSuccess) {
            throw std::runtime_error(
                std::string("Cannot select CUDA device: ") +
                cudaGetErrorString(device_error));
        }
        const cudaError_t property_error =
            cudaGetDeviceProperties(&properties, device);
        if (property_error != cudaSuccess) {
            throw std::runtime_error(
                std::string("Cannot inspect CUDA device: ") +
                cudaGetErrorString(property_error));
        }
        if (!properties.canMapHostMemory) {
            throw std::runtime_error(
                "Selected GPU cannot directly access mapped host memory");
        }
        std::cout << "Host-backed GPU-native mode: " << properties.name
                  << " will access extended codes and factors in mapped CPU "
                     "DRAM"
                  << std::endl;
    }

    // 1. 加载 Query 和 GroundTruth 数据
    std::cout << "Loading queries and groundtruth..." << std::endl;
    data_type query;
    gt_type gt;
    rabitqlib::load_auto<float, data_type>(query_file, query);
    rabitqlib::load_auto<uint32_t, gt_type>(gt_file, gt);
    
    size_t nq = query.rows();
    size_t dim = query.cols();
    std::cout<< "Loaded " << nq << " queries with dimension " << dim << std::endl;
    const size_t requested_query_chunk = options.get_size("query-chunk", nq);
    if (requested_query_chunk == 0) {
        throw std::invalid_argument("query-chunk must be positive");
    }
    const size_t query_chunk = std::min(requested_query_chunk, nq);
    if (isolated_profile && query_chunk != nq) {
        throw std::invalid_argument(
            "--isolated-profile true requires --query-chunk >= query count");
    }

    // 2. 初始化纯 CUDA 版本索引并加载数据
    std::cout << "Initializing GPU Index..." << std::endl;
    
    rabitqlib::ivf::IVF cpu_ivf;
    cpu_ivf.load(index_file.c_str());
    std::unique_ptr<recall_eval::Evaluator> recall_evaluator;
    if (!recall_once) {
        recall_evaluator = std::make_unique<recall_eval::Evaluator>(
            query, gt, topk, recall_base, cpu_ivf.metric_type());
    }

    rabitq_benchmark::validate_benchmark_shape(
        query.rows(), query.cols(), gt.rows(), gt.cols(), cpu_ivf.dim_, topk,
        test_round);

    size_t num_cluster = cpu_ivf.num_cluster_;
    size_t padded_dim = cpu_ivf.padded_dim_;
    size_t num_words = padded_dim / 32;

    // 3. 提取 Centroids 和 Cluster Sizes，并计算最大簇以保障候选池安全
    std::vector<float> h_centroids(num_cluster * padded_dim);
    std::vector<size_t> cluster_sizes(num_cluster);
    size_t max_cluster_size = 0; // [新增] 追踪最大簇大小

    for (size_t i = 0; i < num_cluster; ++i) {
        std::memcpy(h_centroids.data() + i * padded_dim, 
                    cpu_ivf.initer_->centroid(i), 
                    padded_dim * sizeof(float));
        size_t c_size = cpu_ivf.cluster_lst_[i].num();
        cluster_sizes[i] = c_size;
        if (c_size > max_cluster_size) { max_cluster_size = c_size; }
    }

    std::vector<uint32_t> host_short_data(cpu_ivf.num_ * num_words, 0);
    std::vector<float> host_short_factors(cpu_ivf.num_ * 3, 0.0f);

    // [新增]：分配 Long Code 和 Ex Factors 的内存
    size_t ex_bits = cpu_ivf.ex_bits_;
    std::cout<<"ex_bits: "<<ex_bits<<std::endl;
    std::cout<<"topk: "<<topk<<std::endl;
    size_t long_code_size = (padded_dim * ex_bits + 7) / 8;
    std::vector<uint8_t> host_long_code(cpu_ivf.num_ * long_code_size, 0);
    std::vector<float> host_ex_factors(cpu_ivf.num_ * 2, 0.0f); // 每个向量 2 个 float

    size_t global_vec_idx = 0;
    for (size_t c = 0; c < num_cluster; ++c) {
        const auto& cluster = cpu_ivf.cluster_lst_[c];
        size_t n = cluster.num();
        size_t cluster_start_index = global_vec_idx;

        // ==========================================
        // A. 提取并排布 Ex Data (长码与长因子)
        // ==========================================
        const char* ex_ptr = cluster.ex_data();
        if (ex_bits == 3 && ex_ptr != nullptr) { // 针对 FXU3 (3-bit) 的专用重组
            for (size_t v = 0; v < n; ++v) {
                size_t current_global_idx = cluster_start_index + v;
                rabitqlib::ConstExDataMap<float> cur_ex(ex_ptr, padded_dim, ex_bits);
                
                // 严格对齐 GPU 的 float2 访问
                host_ex_factors[current_global_idx * 2 + 0] = cur_ex.f_add_ex();
                host_ex_factors[current_global_idx * 2 + 1] = cur_ex.f_rescale_ex();
                
                // 🌟 核心魔法：将 CPU 的 FXU3 交织格式逆向解码，打包为线性 3-bit 格式
                const uint8_t* raw_fxu3 = cur_ex.ex_code();
                uint8_t* dest_linear = &host_long_code[current_global_idx * long_code_size];
                
                // 清空目标内存
                std::memset(dest_linear, 0, long_code_size);

                for (size_t d = 0; d < padded_dim; ++d) {
                    // 1. 定位当前维度在哪个 64 维的 Chunk 中 (每个 Chunk 24 字节)
                    size_t chunk_idx = d / 64;
                    size_t d_in_chunk = d % 64;
                    const uint8_t* chunk_ptr = raw_fxu3 + chunk_idx * 24;
                    
                    // 2. 解析低 2 位 (前 16 字节交织)
                    size_t group = d_in_chunk / 16;
                    size_t byte_idx = d_in_chunk % 16;
                    uint32_t low2 = (chunk_ptr[byte_idx] >> (group * 2)) & 0b11;
                    
                    // 3. 解析最高位 (后 8 字节, 8x8 转置)
                    uint64_t top_bit_block = *reinterpret_cast<const uint64_t*>(chunk_ptr + 16);
                    size_t row = d_in_chunk / 8;
                    size_t col = d_in_chunk % 8;
                    size_t top_bit_idx = col * 8 + row; // 逆向转置
                    uint32_t top1 = (top_bit_block >> top_bit_idx) & 1;
                    
                    // 4. 还原出真实的 3-bit 数值 (0 ~ 7)
                    uint32_t code_val = (top1 << 2) | low2;
                    
                    // 5. 按 GPU 期待的顺序 (Little-Endian) 紧凑写入 dest_linear
                    size_t bit_idx = d * ex_bits; 
                    size_t dest_byte_idx = bit_idx / 8;
                    size_t bit_offset = bit_idx % 8;
                    
                    dest_linear[dest_byte_idx] |= (code_val << bit_offset);
                    // 跨字节保护
                    if (bit_offset + ex_bits > 8) {
                        dest_linear[dest_byte_idx + 1] |= (code_val >> (8 - bit_offset));
                    }
                }
                
                ex_ptr += rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);
            }
        } else if (ex_bits == 7 && ex_ptr != nullptr) {
            for (size_t v = 0; v < n; ++v) {
                size_t current_global_idx = cluster_start_index + v;
                rabitqlib::ConstExDataMap<float> cur_ex(ex_ptr, padded_dim, ex_bits);
                
                // 严格对齐 GPU 的 float2 访问
                host_ex_factors[current_global_idx * 2 + 0] = cur_ex.f_add_ex();
                host_ex_factors[current_global_idx * 2 + 1] = cur_ex.f_rescale_ex();
                
                const uint8_t* raw_fxu7 = cur_ex.ex_code();
                uint8_t* dest_linear = &host_long_code[current_global_idx * long_code_size];
                
                // 清空目标内存
                std::memset(dest_linear, 0, long_code_size);

                for (size_t d = 0; d < padded_dim; ++d) {
                    // 每 64 个维度占据 56 字节 (48 字节的低 6 位 + 8 字节的最高位)
                    size_t chunk_idx = d / 64;
                    size_t d_in_chunk = d % 64;
                    const uint8_t* chunk_ptr = raw_fxu7 + chunk_idx * 56;
                    
                    uint32_t code_val = 0;

                    // 1. 拼图还原：提取低 6 位 (0~63 映射关系还原)
                    if (d_in_chunk < 16) {
                        code_val = chunk_ptr[d_in_chunk] & 0x3F;
                    } 
                    else if (d_in_chunk < 32) {
                        code_val = chunk_ptr[16 + (d_in_chunk - 16)] & 0x3F;
                    } 
                    else if (d_in_chunk < 48) {
                        code_val = chunk_ptr[32 + (d_in_chunk - 32)] & 0x3F;
                    } 
                    else {
                        // 维度 48~63 的低 6 位隐藏在前 48 字节各自的最高 2 位中
                        size_t k = d_in_chunk - 48; 
                        uint32_t bits01 = (chunk_ptr[k]      >> 6) & 0x03; // 来自 cpt1
                        uint32_t bits23 = (chunk_ptr[16 + k] >> 6) & 0x03; // 来自 cpt2
                        uint32_t bits45 = (chunk_ptr[32 + k] >> 6) & 0x03; // 来自 cpt3
                        code_val = bits01 | (bits23 << 2) | (bits45 << 4);
                    }

                    // 2. 提取第 7 位 (最高位，8x8 逆向转置)
                    const uint8_t* top_bit_ptr = chunk_ptr + 48;
                    uint64_t top_bit_block = *reinterpret_cast<const uint64_t*>(top_bit_ptr);
                    size_t row = d_in_chunk / 8;
                    size_t col = d_in_chunk % 8;
                    size_t top_bit_idx = col * 8 + row; // 逆向转置公式
                    uint32_t top1 = (top_bit_block >> top_bit_idx) & 1;

                    // 3. 组合出完整的 7-bit 整数 (0 ~ 127)
                    code_val |= (top1 << 6);

                    // 4. 按 GPU 期待的顺序 (Little-Endian) 紧凑写入连续内存
                    size_t bit_idx = d * ex_bits; 
                    size_t dest_byte_idx = bit_idx / 8;
                    size_t bit_offset = bit_idx % 8;
                    
                    dest_linear[dest_byte_idx] |= (code_val << bit_offset);
                    // 跨字节保护 (因为一个数值可能会横跨两个相邻的 Byte)
                    if (bit_offset + ex_bits > 8) {
                        dest_linear[dest_byte_idx + 1] |= (code_val >> (8 - bit_offset));
                    }
                }
                
                // 指针步进
                ex_ptr += rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);
            }
        }

        // 🌟 提前在循环外构建 kPerm0 的反向查询表
        const int kPerm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
        int inv_kPerm0[16];
        for (int j = 0; j < 16; ++j) {
            inv_kPerm0[kPerm0[j]] = j;
        }

        // 🌟 提取 1-bit 短码与短因子 (Batch 数据)
        const char* batch_ptr = cluster.batch_data();
        for (size_t i = 0; i < n; i += 32) {
            size_t batch_len = std::min((size_t)32, n - i);
            
            // 必须使用单字节指针！
            const uint8_t* bin_code_bytes = reinterpret_cast<const uint8_t*>(batch_ptr);
            
            const float* batch_add = reinterpret_cast<const float*>(batch_ptr + padded_dim * 4);
            const float* batch_rescale = reinterpret_cast<const float*>(batch_ptr + padded_dim * 4 + 32 * 4);
            const float* batch_error = reinterpret_cast<const float*>(batch_ptr + padded_dim * 4 + 64 * 4);

            for (size_t v = 0; v < batch_len; ++v) {
                size_t vec_idx_in_cluster = i + v; 
                size_t current_global_idx = cluster_start_index + vec_idx_in_cluster;
                
                // =========================================================================
                // FastScan 变态交织内存 100% 精确逆向转置
                // =========================================================================
                for (size_t word = 0; word < num_words; ++word) {
                    uint32_t gpu_word = 0;
                    
                    for (int bit = 0; bit < 32; ++bit) {
                        size_t dim = word * 32 + bit;
                        
                        // 1. 定位原始字节块 (每 8 个维度切为一个大块，占用 32 字节)
                        size_t byte_group = dim / 8;
                        size_t dim_in_group = dim % 8;
                        
                        // 2. 定位半区：Dim 0~3 在前 16 字节，Dim 4~7 在后 16 字节
                        size_t offset_half = (dim_in_group < 4) ? 0 : 16;
                        
                        // 3. 解除 kPerm0 的空间乱序锁定
                        size_t v_half = v % 16;
                        size_t j = inv_kPerm0[v_half];
                        
                        // 4. 定位到绝对精确的物理字节
                        size_t byte_idx = byte_group * 32 + offset_half + j;
                        uint8_t byte_val = bin_code_bytes[byte_idx];
                        
                        // 5. 提取属于该向量的半字节 (Nibble)
                        // v < 16 在低 4 位 (0x0F)，v >= 16 在高 4 位 (>> 4)
                        uint8_t nibble = (v < 16) ? (byte_val & 0x0F) : (byte_val >> 4);
                        
                        // 6. 提取具体的 bit (大端序：Dim 0 是 Bit 3，Dim 3 是 Bit 0)
                        size_t bit_in_nibble = 3 - (dim_in_group % 4);
                        uint32_t bit_val = (nibble >> bit_in_nibble) & 1;
                        
                        // 7. 压入 GPU 的大端格式中 (高位优先)
                        gpu_word |= (bit_val << (31 - bit));
                    }
                    
                    size_t data_offset = cluster_start_index * num_words + word * n + vec_idx_in_cluster;
                    host_short_data[data_offset] = gpu_word;
                }

                // --- 短因子提取 (保持不变) ---
                size_t factor_offset = current_global_idx * 3;
                host_short_factors[factor_offset + 0] = batch_add[v];
                host_short_factors[factor_offset + 1] = batch_rescale[v];
                host_short_factors[factor_offset + 2] = batch_error[v];
            }
            batch_ptr += (padded_dim * 4 + 32 * 12); 
        }
        global_vec_idx += n; 
    }

    // 1. 提取旋转矩阵
    std::vector<float> host_rotator(dim * padded_dim);
    // MatrixRotator 的 save() 方法会直接将 rand_mat_ 的稠密数据 dump 出来
    cpu_ivf.rotator_->save(reinterpret_cast<char*>(host_rotator.data()));

    IVF_RaBitQ_Raw gpu_ivf;
    gpu_ivf.set_force_tiled_fine_kernel(fine_kernel == "tiled");
    gpu_ivf.set_profiling(profile);
    gpu_ivf.load_from_raw_pointers(
        cpu_ivf.num_, cpu_ivf.dim_, padded_dim, num_cluster, cpu_ivf.ex_bits_,
        h_centroids.data(),
        host_rotator.data(),
        host_short_data.data(),     
        host_short_factors.data(), 
        host_long_code.data(),      // [新增] 传入 Long Code
        host_ex_factors.data(),     // [新增] 传入 Ex Factors
        cpu_ivf.ids_,               
        cluster_sizes.data(),
        host_backed_ex_codes
    );

    gpu_ivf.max_cluster_size_ = max_cluster_size;
    gpu_ivf.best_rescaling_factor = get_const_scaling_factors(padded_dim, cpu_ivf.ex_bits_);
    std::cout<<"best_rescaling_factor: "<<gpu_ivf.best_rescaling_factor<<std::endl;
    
    // std::vector<float> rotated_queries(nq * padded_dim, 0.0f);
    // for(size_t i = 0; i < nq; ++i) {
    //     cpu_ivf.rotator_->rotate(query.data() + i * cpu_ivf.dim_, rotated_queries.data() + i * padded_dim);
    // }

    const auto nprobes = rabitq_benchmark::parse_nprobes(
        options.get("nprobes", ""), num_cluster);
    if (!cluster_value_output.empty() && nprobes.size() != 1) {
        throw std::invalid_argument(
            "--cluster-value-output requires exactly one --nprobes value");
    }
    size_t length = nprobes.size();

    std::cout << "Allocating GPU Workspace pool..." << std::endl;
    const size_t max_nprobe =
        *std::max_element(nprobes.begin(), nprobes.end());
    gpu_ivf.prepare_workspace(query_chunk, topk, max_nprobe);
    if (index_residency == "auto-cache") {
        gpu_ivf.configure_hbm_cache(
            hbm_reserve_mib << 20,
            hbm_cache_mib ? hbm_cache_mib << 20 : std::numeric_limits<size_t>::max());
    }

    std::cout << "Algorithm: GPU-native IVF-RaBitQ\n"
              << "index_residency: " << index_residency << "\n"
              << "hbm_reserve_mib: " << hbm_reserve_mib << "\n"
              << "hbm_cache_mib: " << hbm_cache_mib << "\n"
              << "fine_kernel: "
              << ((fine_kernel == "tiled" || host_backed_ex_codes)
                      ? "tiled"
                      : "original-untiled")
              << "\n"
              << "topk: " << topk << "\n"
              << "query_chunk: " << query_chunk << "\n"
              << "rounds: " << test_round << "\n"
              << "warmup_rounds: " << warmup_rounds << "\n"
              << "profile: " << (profile ? "true" : "false") << "\n"
              << "isolated_profile: "
              << (isolated_profile ? "true" : "false") << "\n"
              << "isolated_profile_rounds: "
              << isolated_profile_rounds << "\n"
              << "cluster_value_output: "
              << (cluster_value_output.empty()
                      ? std::string("disabled")
                      : cluster_value_output.string())
              << "\n"
              << "nprobes: " << rabitq_benchmark::join_values(nprobes) << '\n';
    if (memory_breakdown) {
        print_hbm_breakdown(
            cpu_ivf.num_, dim, padded_dim, num_cluster, ex_bits, query_chunk, topk,
            max_nprobe, host_backed_ex_codes, gpu_ivf.cached_bytes());
    }

    // 记录测试结果
    std::vector<std::vector<float>> all_qps(test_round, std::vector<float>(length));
    std::vector<std::vector<float>> all_recall(
        recall_once ? 1 : test_round, std::vector<float>(length));
    std::vector<std::vector<uint32_t>> recall_results(recall_once ? length : 0);
    for (auto& ids : recall_results) ids.resize(nq * topk);
    if (recall_once) {
        std::cout << "[RecallSchedule] policy=once-after-benchmark source_round=1"
                  << " evaluations=" << length << '\n';
    }

    rabitqlib::StopW stopw;

    // 提前在 Host 端分配好接收批量结果的空间 [nq * topk]
    std::vector<float> results_dists(nq * topk);
    std::vector<uint32_t> results_pids(nq * topk);
    std::vector<int> selected_clusters;
    if (!cluster_value_output.empty()) {
        selected_clusters.resize(nq * nprobes.front());
    }

    auto search_all_queries = [&](size_t nprobe, bool capture_clusters) {
        for (size_t begin = 0; begin < nq; begin += query_chunk) {
            const size_t count = std::min(query_chunk, nq - begin);
            const float* chunk_queries = query.data() + begin * dim;
            float* chunk_dists = results_dists.data() + begin * topk;
            uint32_t* chunk_pids = results_pids.data() + begin * topk;
            int* chunk_clusters = capture_clusters
                ? selected_clusters.data() + begin * nprobe
                : nullptr;
            gpu_ivf.reset_buffer(chunk_queries, count);
            gpu_ivf.search(
                chunk_queries, count, topk, nprobe, chunk_dists, chunk_pids,
                chunk_clusters);
        }
    };

    std::cout << "Starting batch benchmark..." << std::endl;
    for (size_t l = 0; l < length; ++l) {
        const size_t nprobe = nprobes[l];
        gpu_ivf.set_profiling(false);
        gpu_ivf.set_isolated_profiling(false);
        for (size_t warmup = 0; warmup < warmup_rounds; ++warmup) {
            search_all_queries(nprobe, false);
        }
        gpu_ivf.set_profiling(profile);
        gpu_ivf.set_isolated_profiling(
            isolated_profile, isolated_profile_rounds);
        for (size_t r = 0; r < test_round; r++) {
            stopw.reset();
            const bool capture_clusters =
                !cluster_value_output.empty() && r + 1 == test_round;
            search_all_queries(nprobe, capture_clusters);
            float total_time = stopw.get_elapsed_micro();
            float qps = static_cast<float>(nq) / (total_time / 1e6F);
            all_qps[r][l] = qps;
            if (recall_once) {
                if (r == 0) {
                    std::copy(results_pids.begin(), results_pids.end(), recall_results[l].begin());
                }
            } else {
                all_recall[r][l] = recall_evaluator->evaluate(results_pids);
            }
            std::cout << "[Run] round=" << (r + 1)
                      << " nprobe=" << nprobe
                      << " QPS=" << qps;
            if (recall_once) std::cout << " recall=deferred\n";
            else std::cout << " recall=" << all_recall[r][l] << '\n';
            if (isolated_profile && r + 1 == test_round) {
                gpu_ivf.profile_isolated_last_search(nq, topk, nprobe);
            }
        }
    }
    if (!cluster_value_output.empty()) {
        write_cluster_value_profile(
            cluster_value_output, cpu_ivf.ids_, cpu_ivf.num_, cluster_sizes,
            selected_clusters, results_pids, nq, topk, nprobes.front());
    }
    if (recall_once) {
        std::cout << "[RecallSchedule] QPS rounds complete; evaluating saved results" << std::endl;
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

    // std::cout << "nprobe\tQPS\trecall\n";
    // for (size_t i = 0; i < length; ++i) {
    //     std::cout << nprobes[i] << '\n';
    // }
    // for (size_t i = 0; i < length; ++i) {
    //     std::cout << avg_qps[i] << '\n';
    // }
    // for (size_t i = 0; i < length; ++i) {
    //     std::cout << avg_recall[i] << '\n';
    // }

    } catch (const std::exception& error) {
        std::cerr << "ivf_rabitq_querying_gpu failed: " << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
