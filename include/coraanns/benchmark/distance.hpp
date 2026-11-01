#pragma once

#include "rabitqlib/defines.hpp"
#include "coraanns/io/mapped_vectors.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace graph_baseline {

using FloatMatrix = rabitqlib::RowMajorArray<float>;
using IdMatrix = rabitqlib::RowMajorArray<std::uint32_t>;

inline double elapsed_seconds(const std::chrono::steady_clock::time_point &start,
                              const std::chrono::steady_clock::time_point &stop) {
    return std::chrono::duration<double>(stop - start).count();
}

using FloatVectorMMap = coraanns::io::MappedVectors;

inline float squared_l2(const float *lhs, const float *rhs, std::size_t dim) {
    float distance = 0.0f;
#pragma omp simd reduction(+ : distance)
    for (std::size_t d = 0; d < dim; ++d) {
        const float delta = lhs[d] - rhs[d];
        distance += delta * delta;
    }
    return distance;
}

inline void exact_rerank_l2(const FloatVectorMMap &base, const float *queries,
                            std::size_t query_count, std::size_t dim,
                            const std::uint32_t *candidates, std::size_t candidate_count,
                            std::size_t topk, std::uint32_t *result_ids, float *result_distances) {
    if (candidate_count < topk) {
        throw std::invalid_argument("candidate_count must be at least topk");
    }

#pragma omp parallel
    {
        std::vector<std::pair<float, std::uint32_t>> scored;
        scored.reserve(candidate_count);

#pragma omp for
        for (std::int64_t q = 0; q < static_cast<std::int64_t>(query_count); ++q) {
            scored.clear();
            std::unordered_set<std::uint32_t> seen;
            seen.reserve(candidate_count * 2);
            const auto *query = queries + static_cast<std::size_t>(q) * dim;
            const auto *row_candidates = candidates + static_cast<std::size_t>(q) * candidate_count;
            for (std::size_t c = 0; c < candidate_count; ++c) {
                const auto id = row_candidates[c];
                if (id >= base.rows() || !seen.insert(id).second)
                    continue;
                scored.emplace_back(squared_l2(query, base.row(id), dim), id);
            }

            const auto keep = std::min(topk, scored.size());
            std::partial_sort(scored.begin(), scored.begin() + keep, scored.end(),
                              [](const auto &lhs, const auto &rhs) {
                                  if (lhs.first != rhs.first)
                                      return lhs.first < rhs.first;
                                  return lhs.second < rhs.second;
                              });
            for (std::size_t k = 0; k < keep; ++k) {
                result_distances[static_cast<std::size_t>(q) * topk + k] = scored[k].first;
                result_ids[static_cast<std::size_t>(q) * topk + k] = scored[k].second;
            }
            for (std::size_t k = keep; k < topk; ++k) {
                result_distances[static_cast<std::size_t>(q) * topk + k] =
                    std::numeric_limits<float>::infinity();
                result_ids[static_cast<std::size_t>(q) * topk + k] =
                    std::numeric_limits<std::uint32_t>::max();
            }
        }
    }
}

} // namespace graph_baseline
