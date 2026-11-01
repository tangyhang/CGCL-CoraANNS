#pragma once

#include <climits>
#include <cstddef>
#include <stdexcept>

namespace coraanns {

inline constexpr std::size_t default_vanguard_size = 5;
inline constexpr int default_batch_q = 4;
// LUT preparation uses grid.y; candidate query/probe IDs occupy 16 bits each.
inline constexpr std::size_t max_query_chunk = 65535;
inline constexpr std::size_t max_probe_count = 65536;

inline void validate_search_shape(std::size_t queries, std::size_t topk, std::size_t nprobe,
                                  std::size_t centroids, std::size_t padded_dim) {
    if (queries == 0 || queries > max_query_chunk)
        throw std::invalid_argument("query count must be in [1, 65535]; split larger batches");
    if (topk == 0 || topk > INT_MAX / queries)
        throw std::invalid_argument("topk must be positive and queries * topk must fit an int");
    if (nprobe == 0 || nprobe > max_probe_count || nprobe > centroids)
        throw std::invalid_argument("nprobe must be in [1, min(nlist, 65536)]");
    if (centroids > INT_MAX / queries || padded_dim == 0 || padded_dim > INT_MAX / (queries * 4))
        throw std::invalid_argument("query workspace exceeds kernel indexing limits");
}

} // namespace coraanns
