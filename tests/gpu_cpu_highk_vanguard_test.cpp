// Capacity regression: link the high-k production CUDA/CPU objects.
// Multiple 2048-vector tiles per cluster and unique refinement distances.
#include "coraanns/search/hybrid.cuh"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <utility>
#include <vector>

static void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

struct Fixture {
    static constexpr size_t count = 32768, dim = 96, padded = 128;
    static constexpr size_t clusters = 8, queries = 5, bits = 7;
    std::vector<float> centroids = std::vector<float>(clusters * padded);
    std::vector<float> rotation = std::vector<float>(dim * padded);
    std::vector<uint32_t> short_codes = std::vector<uint32_t>(count * padded / 32);
    std::vector<float> short_factors = std::vector<float>(count * 3);
    std::vector<uint8_t> long_codes = std::vector<uint8_t>(count * padded * bits / 8);
    std::vector<float> long_factors = std::vector<float>(count * 2);
    std::vector<uint32_t> ids = std::vector<uint32_t>(count);
    std::vector<size_t> sizes = std::vector<size_t>(clusters, count / clusters);
    std::vector<float> query = std::vector<float>(queries * dim);

    Fixture() {
        for (size_t d = 0; d < dim; ++d) rotation[d * padded + d] = 1;
        for (size_t i = 0; i < count; ++i) {
            ids[i] = i;
            short_factors[3*i] = -1e6f;
            long_factors[2*i] = 0.01f * float((i * 7919) % count);
        }
        for (size_t q = 0; q < queries; ++q) {
            for (size_t d = 0; d < dim; ++d) query[q*dim+d] = .01f * (1 + (q+d)%7);
        }
    }
};

using Result = std::vector<std::vector<std::pair<uint32_t, float>>>;

static Result run(IVF_RaBitQ_Raw& engine, const Fixture& f,
                  size_t topk, size_t chunk) {
    std::vector<uint32_t> ids(f.queries * topk);
    std::vector<float> distances(f.queries * topk);
    for (size_t begin = 0; begin < f.queries; begin += chunk) {
        const size_t n = std::min(chunk, f.queries-begin);
        const float* query = f.query.data() + begin*f.dim;
        engine.reset_buffer(query, n);
        engine.search(query, n, topk, f.clusters,
                      distances.data()+begin*topk, ids.data()+begin*topk, 4);
    }
    Result result(f.queries);
    for (size_t q = 0; q < f.queries; ++q) {
        for (size_t k = 0; k < topk; ++k) {
            const size_t i = q*topk+k;
            require(ids[i] < f.count && std::isfinite(distances[i]), "Invalid result");
            result[q].emplace_back(ids[i], distances[i]);
        }
        std::sort(result[q].begin(), result[q].end());
    }
    return result;
}

static void initialize(IVF_RaBitQ_Raw& engine, const Fixture& f,
                       size_t k, size_t vanguard) {
    engine.set_vanguard_size(vanguard);
    engine.set_cpu_secondary_filter(true);
    engine.set_cross_query_reuse(true);
    engine.load_from_raw_pointers(f.count, f.dim, f.padded, f.clusters, f.bits,
        f.centroids.data(), f.rotation.data(), f.short_codes.data(),
        f.short_factors.data(), f.long_codes.data(), f.long_factors.data(),
        f.ids.data(), f.sizes.data());
    engine.max_cluster_size_ = f.count / f.clusters;
    engine.best_rescaling_factor = 1;
    engine.prepare_workspace(f.queries, k, f.clusters);
}

int main() {
    try {
        omp_set_dynamic(0);
        omp_set_num_threads(4);
        Fixture f;
        IVF_RaBitQ_Raw engine;
        initialize(engine, f, 1024, 5);
        for (size_t k : {10, 50, 100, 200, 500, 512, 513, 1000, 1024}) {
            engine.set_vanguard_size(0);
            const Result reference = run(engine, f, k, f.queries);
            // With the cheap filter forced open, the unique smallest refinement
            // factors define the exact expected IDs, independently of Vanguard.
            std::vector<std::pair<float,uint32_t>> ordered;
            for (size_t i=0; i<f.count; ++i)
                ordered.emplace_back(f.long_factors[2*i], static_cast<uint32_t>(i));
            std::sort(ordered.begin(), ordered.end());
            std::vector<uint32_t> expected;
            for (size_t i=0; i<k; ++i) expected.push_back(ordered[i].second);
            std::sort(expected.begin(), expected.end());
            for (const auto& result : reference) {
                std::vector<uint32_t> actual;
                for (const auto& pair : result) actual.push_back(pair.first);
                require(actual == expected, "Reference does not match unique exact nearest IDs");
            }
            engine.set_vanguard_size(5);
            for (int round=0; round<2; ++round) {
                require(run(engine, f, k, f.queries) == reference,
                        "Vanguard seed/recovery changes full-batch IDs or distances");
                require(run(engine, f, k, 2) == reference,
                        "Vanguard seed/recovery changes chunked IDs or distances");
            }
            std::cout << "PASS: top" << k
                      << " capacity=1024; multi-tile, exact IDs, no-Vanguard/full/chunked distances agree" << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
