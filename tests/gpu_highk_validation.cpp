// Small CPU-oracle check, run by GPU_laion100m_highk.sh before the large index.
// Three queries, 16K vectors and four multi-tile clusters; no external dataset.
#include "coraanns/baselines/gpu.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
constexpr size_t count = 16384, dim = 96, padded = 128;
constexpr size_t clusters = 4, queries = 3, bits = 7;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Fixture {
    std::vector<float> centroids = std::vector<float>(clusters * padded, 0);
    std::vector<float> rotator = std::vector<float>(dim * padded, 0);
    std::vector<uint32_t> short_codes = std::vector<uint32_t>(count * (padded / 32), 0);
    std::vector<float> short_factors = std::vector<float>(count * 3, 0);
    std::vector<uint8_t> long_codes = std::vector<uint8_t>(count * padded * bits / 8, 0);
    std::vector<float> ex_factors = std::vector<float>(count * 2, 0);
    std::vector<uint32_t> ids = std::vector<uint32_t>(count);
    std::vector<size_t> sizes = std::vector<size_t>(clusters, count / clusters);
    std::vector<float> query = std::vector<float>(queries * dim);
    std::vector<std::pair<float, uint32_t>> oracle;

    Fixture() {
        for (size_t d = 0; d < dim; ++d) rotator[d * padded + d] = 1;
        for (size_t i = 0; i < count; ++i) {
            ids[i] = i;
            short_factors[3 * i] = -1e6f; // Every vector passes the lower bound.
            ex_factors[2 * i] = float((i * 7919) % count);
            oracle.emplace_back(ex_factors[2 * i], static_cast<uint32_t>(i));
        }
        std::sort(oracle.begin(), oracle.end());
        for (size_t q = 0; q < queries; ++q)
            for (size_t d = 0; d < dim; ++d)
                query[q * dim + d] = .125f * float(q + 1);
    }
};

void check_result(const Fixture& f, size_t topk, bool host, bool cache, size_t chunk) {
    IVF_RaBitQ_Raw engine;
    engine.set_force_tiled_fine_kernel(true);
    engine.load_from_raw_pointers(count, dim, padded, clusters, bits,
        f.centroids.data(), f.rotator.data(), f.short_codes.data(), f.short_factors.data(),
        f.long_codes.data(), f.ex_factors.data(), f.ids.data(), f.sizes.data(), host);
    engine.max_cluster_size_ = count / clusters;
    engine.best_rescaling_factor = 1;
    engine.prepare_workspace(chunk, topk, clusters);
    if (cache) {
        engine.configure_hbm_cache(size_t(512) << 20, size_t(128) << 10);
        require(engine.cached_vectors() > 0 && engine.cached_vectors() < count,
                "Expected both cached and uncached vectors");
    }
    std::vector<float> distances(queries * topk);
    std::vector<uint32_t> ids(queries * topk);
    for (size_t first = 0; first < queries; first += chunk) {
        const size_t n = std::min(chunk, queries - first);
        engine.reset_buffer(f.query.data() + first * dim, n);
        engine.search(f.query.data() + first * dim, n, topk, clusters,
                      distances.data() + first * topk, ids.data() + first * topk);
    }
    for (size_t q = 0; q < queries; ++q) {
        std::vector<std::pair<float, uint32_t>> actual;
        for (size_t k = 0; k < topk; ++k)
            actual.emplace_back(distances[q * topk + k], ids[q * topk + k]);
        std::sort(actual.begin(), actual.end());
        const float value = .125f * float(q + 1);
        const float norm = float(dim) * value * value;
        for (size_t k = 0; k < topk; ++k) {
            require(actual[k].second == f.oracle[k].second,
                    "Top-k differs from the complete CPU oracle");
            require(std::isfinite(actual[k].first) &&
                    std::abs(actual[k].first - (f.oracle[k].first + norm)) < .001f,
                    "Distance differs from the CPU oracle");
        }
    }
}
} // namespace

int main() {
    try {
        const Fixture f;
        for (size_t topk : {500, 512, 513, 1000, 1024}) {
            check_result(f, topk, false, false, queries);
            check_result(f, topk, true, false, queries);
            check_result(f, topk, true, true, 2);
            std::cout << "PASS: GPU topk=" << topk
                      << " multi-tile HBM/host/partial-cache/chunked vs CPU oracle\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "GPU high-k validation failed: " << error.what() << '\n';
        return 1;
    }
}
