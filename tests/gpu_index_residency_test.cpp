// Link against the production ivf_rabitq_querying_gpu.cu object and CUDA/cuBLAS.
// Exercises both sides of a partially cached cluster, not just allocation.
#include "coraanns/baselines/gpu.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Fixture {
    static constexpr size_t count = 32768, dim = 96, padded = 128;
    static constexpr size_t clusters = 8, queries = 5, bits = 7;
    std::vector<float> centroids = std::vector<float>(clusters * padded);
    std::vector<float> rotator = std::vector<float>(dim * padded);
    std::vector<uint32_t> short_codes = std::vector<uint32_t>(count * (padded / 32));
    std::vector<float> short_factors = std::vector<float>(count * 3);
    std::vector<uint8_t> long_codes = std::vector<uint8_t>(count * padded * bits / 8);
    std::vector<float> ex_factors = std::vector<float>(count * 2);
    std::vector<uint32_t> ids = std::vector<uint32_t>(count);
    std::vector<size_t> sizes = std::vector<size_t>(clusters, count / clusters);
    std::vector<float> query = std::vector<float>(queries * dim);

    Fixture() {
        for (size_t d = 0; d < dim; ++d) rotator[d * padded + d] = 1;
        for (size_t i = 0; i < count; ++i) {
            ids[i] = i;
            short_factors[3 * i] = -1e6f; // Every vector passes the cheap filter.
            ex_factors[2 * i] = float((i * 7919) % count) * 0.01f;
            ex_factors[2 * i + 1] = 0.00001f * (1 + float(i % 7) * .1f);
        }
        for (size_t i = 0; i < long_codes.size(); ++i) long_codes[i] = (i * 37 + i / 11) % 256;
        for (size_t q = 0; q < queries; ++q) {
            for (size_t d = 0; d < dim; ++d) query[q * dim + d] = .01f * (1 + (q * 13 + d * 7) % 31);
        }
    }
};

using Results = std::vector<std::vector<std::pair<uint32_t, float>>>;

static Results run(const Fixture& f, size_t topk, int mode, size_t chunk = Fixture::queries) {
    IVF_RaBitQ_Raw engine;
    engine.set_force_tiled_fine_kernel(true);
    engine.load_from_raw_pointers(f.count, f.dim, f.padded, f.clusters, f.bits,
        f.centroids.data(), f.rotator.data(), f.short_codes.data(), f.short_factors.data(),
        f.long_codes.data(), f.ex_factors.data(), f.ids.data(), f.sizes.data(), mode != 0);
    engine.max_cluster_size_ = f.count / f.clusters;
    engine.best_rescaling_factor = 1;
    engine.prepare_workspace(chunk, topk, f.clusters);
    if (mode == 2) {
        engine.configure_hbm_cache(size_t(512) << 20);
        require(engine.cached_vectors() == f.count, "Full cache must hold all vectors");
    } else if (mode == 3) {
        engine.configure_hbm_cache(size_t(512) << 20, size_t(1) << 20);
        require(engine.cached_vectors() > 0 && engine.cached_vectors() < f.count, "Expected partial cache");
        require(engine.cached_bytes() <= size_t(1) << 20, "Cache exceeds cap");
    } else if (mode == 4) {
        engine.configure_hbm_cache(std::numeric_limits<size_t>::max());
        require(engine.cached_vectors() == 0, "No memory budget must produce an empty cache");
    }
    std::vector<uint32_t> ids(f.queries * topk);
    std::vector<float> distances(f.queries * topk);
    for (size_t begin = 0; begin < f.queries; begin += chunk) {
        const size_t n = std::min(chunk, f.queries - begin);
        engine.reset_buffer(f.query.data() + begin * f.dim, n);
        engine.search(f.query.data() + begin * f.dim, n, topk, f.clusters,
                      distances.data() + begin * topk, ids.data() + begin * topk);
    }
    Results result(f.queries);
    bool hit = false, miss = false;
    for (size_t q = 0; q < f.queries; ++q) {
        for (size_t k = 0; k < topk; ++k) {
            size_t pos = q * topk + k;
            require(ids[pos] < f.count && std::isfinite(distances[pos]), "Invalid result");
            result[q].emplace_back(ids[pos], distances[pos]);
            hit |= ids[pos] < engine.cached_vectors();
            miss |= ids[pos] >= engine.cached_vectors();
        }
        std::sort(result[q].begin(), result[q].end());
    }
    if (mode == 3) require(hit && miss, "Fixture must exercise cached and uncached results");
    return result;
}

int main() {
    try {
        Fixture fixture;
        for (size_t topk : {10, 50, 100, 200}) {
            const auto reference = run(fixture, topk, 0);
            for (int mode : {1, 2, 3, 4}) {
                const auto actual = run(fixture, topk, mode);
                for (size_t q = 0; q < fixture.queries; ++q) {
                    for (size_t k = 0; k < topk; ++k) {
                        require(actual[q][k].first == reference[q][k].first, "Residency changes neighbor IDs");
                        require(std::abs(actual[q][k].second - reference[q][k].second) < 1e-5f,
                                "Residency changes distances");
                    }
                }
            }
            const auto chunks = run(fixture, topk, 3, 2);
            require(chunks == reference, "Chunking changes partial-cache results");
            std::cout << "PASS: topk=" << topk << " HBM/host/full-cache/partial-cache/no-cache/chunked results agree\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
