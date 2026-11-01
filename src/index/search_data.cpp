#include "coraanns/index/search_data.hpp"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace coraanns::index {

SearchData::SearchData(rabitqlib::ivf::IVF &cpu_ivf, bool include_extended, int cpu_threads) {
    const bool exact_base = !include_extended;
    const size_t dim = cpu_ivf.dim();
    size_t num_cluster = cpu_ivf.num_clusters();
    size_t padded_dim = cpu_ivf.padded_dim();
    size_t num_words = padded_dim / 32;

    h_centroids.resize(num_cluster * padded_dim);
    cluster_sizes.resize(num_cluster);

    for (size_t i = 0; i < num_cluster; ++i) {
        std::memcpy(h_centroids.data() + i * padded_dim, cpu_ivf.centroid(i),
                    padded_dim * sizeof(float));
        size_t c_size = cpu_ivf.clusters()[i].num();
        cluster_sizes[i] = c_size;
        if (c_size > max_cluster_size)
            max_cluster_size = c_size;
    }

    host_short_data.resize(cpu_ivf.num_vectors() * num_words, 0);
    host_short_factors.resize(cpu_ivf.num_vectors() * 3, 0.0f);

    size_t ex_bits = cpu_ivf.ex_bits();
    std::cout << "ex_bits: " << ex_bits << std::endl;
    size_t long_code_size = (padded_dim * ex_bits + 7) / 8;

    if (!exact_base) {
        host_ex_factors.resize(cpu_ivf.num_vectors() * 2, 0.0f);
    }

    size_t global_vec_idx = 0;
    const int kPerm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
    int inv_kPerm0[16];
    for (int j = 0; j < 16; ++j)
        inv_kPerm0[kPerm0[j]] = j;

    for (size_t c = 0; c < num_cluster; ++c) {
        const auto &cluster = cpu_ivf.clusters()[c];
        size_t n = cluster.num();
        size_t cluster_start_index = global_vec_idx;

        const char *batch_ptr = cluster.batch_data();
        for (size_t i = 0; i < n; i += 32) {
            size_t batch_len = std::min((size_t)32, n - i);
            const uint8_t *bin_code_bytes = reinterpret_cast<const uint8_t *>(batch_ptr);
            const float *batch_add = reinterpret_cast<const float *>(batch_ptr + padded_dim * 4);
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

    if (!exact_base) {
        // Compact interleaved codes in place with bounded staging storage.
        const size_t ex_stride = rabitqlib::ExDataMap<float>::data_bytes(padded_dim, ex_bits);
        if (long_code_size == 0 || ex_stride < long_code_size + 2 * sizeof(float)) {
            throw std::runtime_error("invalid extended-code layout");
        }

        constexpr size_t kCompactStagingBytes = size_t{192} << 20;
        const size_t compact_chunk_vectors = std::max(
            size_t{1}, std::min(cpu_ivf.num_vectors(), kCompactStagingBytes / long_code_size));
        std::vector<uint8_t> compact_codes(compact_chunk_vectors * long_code_size);
        extended_storage_.reset(cpu_ivf.release_extended_data());
        if (!extended_storage_)
            throw std::runtime_error("missing extended codes");
        auto *ex_data = reinterpret_cast<uint8_t *>(extended_storage_.get());

#pragma omp parallel num_threads(cpu_threads)
        {
            for (size_t begin = 0; begin < cpu_ivf.num_vectors(); begin += compact_chunk_vectors) {
                const size_t count = std::min(compact_chunk_vectors, cpu_ivf.num_vectors() - begin);
#pragma omp for schedule(static)
                for (long long offset = 0; offset < static_cast<long long>(count); ++offset) {
                    const size_t vector_index = begin + static_cast<size_t>(offset);
                    const uint8_t *record = ex_data + vector_index * ex_stride;
                    std::memcpy(compact_codes.data() + static_cast<size_t>(offset) * long_code_size,
                                record, long_code_size);
                    const float *factors = reinterpret_cast<const float *>(record + long_code_size);
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
                  << (cpu_ivf.num_vectors() * long_code_size / (1024.0 * 1024.0 * 1024.0))
                  << " GiB duplicate\n";
    }

    host_rotator.resize(dim * padded_dim);
    cpu_ivf.save_rotator(reinterpret_cast<char *>(host_rotator.data()));
}

} // namespace coraanns::index
