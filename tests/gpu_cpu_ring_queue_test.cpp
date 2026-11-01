#include "coraanns/search/hybrid.cuh"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <omp.h>
#include <stdexcept>
#include <vector>

static void require(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}

template <size_t Count = 32768, size_t TopK = 100> struct Fixture {
    static constexpr size_t count = Count, dim = 96, padded = 128;
    static constexpr size_t clusters = 8, queries = 5, bits = 7, k = TopK;
    std::vector<float> centroids = std::vector<float>(clusters * padded);
    std::vector<float> rotation = std::vector<float>(dim * padded);
    std::vector<uint32_t> short_codes = std::vector<uint32_t>(count * padded / 32);
    std::vector<float> short_factors = std::vector<float>(count * 3);
    std::vector<uint8_t> long_codes = std::vector<uint8_t>(count * padded * bits / 8);
    std::vector<float> long_factors = std::vector<float>(count * 2);
    std::vector<uint32_t> ids = std::vector<uint32_t>(count);
    std::vector<size_t> sizes = std::vector<size_t>(clusters, count / clusters);
    std::vector<float> query = std::vector<float>(queries * dim);
    std::vector<uint32_t> expected;

    Fixture() {
        for (size_t d = 0; d < dim; ++d)
            rotation[d * padded + d] = 1;
        for (size_t i = 0; i < count; ++i) {
            ids[i] = i;

            short_factors[3 * i] = -1e6f;
            long_factors[2 * i] = 0.01f * float((i * 7919) % count);
        }
        for (size_t q = 0; q < queries; ++q)
            for (size_t d = 0; d < dim; ++d)
                query[q * dim + d] = .01f * (1 + (q + d) % 7);
        std::vector<uint32_t> ordered = ids;
        std::sort(ordered.begin(), ordered.end(), [&](uint32_t a, uint32_t b) {
            return long_factors[2 * a] < long_factors[2 * b];
        });
        expected.assign(ordered.begin(), ordered.begin() + k);
        std::sort(expected.begin(), expected.end());
    }
};

template <class F>
static void initialize(IVF_RaBitQ_Raw &engine, const F &f, size_t capacity, size_t spill_capacity,
                       size_t vanguard) {
    engine.set_queue_capacity(capacity);
    engine.set_queue_spill_capacity(spill_capacity);
    engine.set_vanguard_size(vanguard);
    engine.set_candidate_stats(true);
    engine.load_from_raw_pointers(f.count, f.dim, f.padded, f.clusters, f.bits, f.centroids.data(),
                                  f.rotation.data(), f.short_codes.data(), f.short_factors.data(),
                                  f.long_codes.data(), f.long_factors.data(), f.ids.data(),
                                  f.sizes.data());
    engine.max_cluster_size_ = f.count / f.clusters;
    engine.prepare_workspace(f.queries, f.k, f.clusters);
}

template <class F>
static void check(IVF_RaBitQ_Raw &engine, const F &f, size_t chunk, int batch_q = 4,
                  bool expect_all_consumed = false) {
    std::vector<uint32_t> ids(f.queries * f.k);
    std::vector<float> distances(f.queries * f.k);
    for (size_t begin = 0; begin < f.queries; begin += chunk) {
        const size_t n = std::min(chunk, f.queries - begin);
        const float *query = f.query.data() + begin * f.dim;
        engine.reset_buffer(query, n);
        engine.search(query, n, f.k, f.clusters, distances.data() + begin * f.k,
                      ids.data() + begin * f.k, batch_q);
        require(engine.get_last_submitted_count() == n * f.count,
                "candidate count changed: dropped or duplicated candidates");
        if (expect_all_consumed)
            require(engine.get_last_refined_count() == n * f.count,
                    "CPU consumption count differs from the exhaustive candidate count");
    }
    for (size_t q = 0; q < f.queries; ++q) {
        std::vector<uint32_t> actual(ids.begin() + q * f.k, ids.begin() + (q + 1) * f.k);
        std::sort(actual.begin(), actual.end());
        require(actual == f.expected, "ring reuse changed exact nearest IDs");
        for (size_t i = 0; i < f.k; ++i) {
            const size_t pos = q * f.k + i;
            float query_norm = 0;
            for (size_t d = 0; d < f.dim; ++d)
                query_norm += f.query[q * f.dim + d] * f.query[q * f.dim + d];
            const float expected = f.long_factors[2 * ids[pos]] + query_norm;
            require(std::abs(distances[pos] - expected) < 1e-4f,
                    "published task payload was torn or stale");
        }
    }
}

int main() {
    try {
        omp_set_dynamic(0);
        Fixture<> f;
        for (size_t capacity : {size_t{8192}, size_t{262144}}) {
            for (size_t vanguard : {size_t{0}, size_t{5}}) {
                IVF_RaBitQ_Raw engine;
                initialize(engine, f, capacity, 262144, vanguard);
                for (int threads : {1, 4, 16, 72}) {
                    omp_set_num_threads(threads);
                    for (bool secondary : {false, true}) {
                        engine.set_cpu_secondary_filter(secondary);
                        for (int round = 0; round < 3; ++round) {
                            check(engine, f, f.queries, 4, !secondary);
                            if (capacity > f.queries * f.count)
                                require(engine.get_last_spilled_count() == 0, "unexpected spill");
                            check(engine, f, 2, 4, !secondary);
                        }
                    }
                }
                std::cout
                    << "PASS: capacity=" << capacity << " vanguard=" << vanguard
                    << " repeated wraps, exact IDs/distances, CPU threads 1/4/16/72, query chunks\n";
            }
        }
        {
            omp_set_num_threads(4);
            Fixture<1032, 10> partial;
            bool spilled = false;
            for (size_t capacity : {size_t{32}, size_t{8192}}) {
                IVF_RaBitQ_Raw engine;
                initialize(engine, partial, capacity, 8192, 0);
                for (int round = 0; round < 4; ++round) {
                    check(engine, partial, partial.queries);
                    spilled |= engine.get_last_spilled_count() > 0;
                }
            }
            require(spilled, "forced-spill fixture did not exercise HBM refill");
            std::cout << "PASS: final partial chunk, forced spill/refill, repeated reset\n";
        }
        {
            omp_set_num_threads(4);
            IVF_RaBitQ_Raw engine;
            initialize(engine, f, 8192, 262144, 5);
            for (int batch_q : {1, 2, 3, 5, 6})
                check(engine, f, f.queries, batch_q);
            std::cout << "PASS: LUT query batches 1/2/3/5/6\n";
        }
        {
            Fixture<> empty;
            for (size_t i = 0; i < empty.count; ++i)
                empty.short_factors[3 * i] = std::numeric_limits<float>::max();
            IVF_RaBitQ_Raw engine;
            initialize(engine, empty, 8192, 1024, 0);
            std::vector<uint32_t> ids(empty.queries * empty.k);
            std::vector<float> distances(ids.size());
            for (int round = 0; round < 3; ++round) {
                engine.reset_buffer(empty.query.data(), empty.queries);
                engine.search(empty.query.data(), empty.queries, empty.k, empty.clusters,
                              distances.data(), ids.data());
                require(engine.get_last_submitted_count() == 0, "empty queue published records");
                require(std::all_of(
                            distances.begin(), distances.end(),
                            [](float value) { return value == std::numeric_limits<float>::max(); }),
                        "empty queue returned stale results");
            }
            std::cout << "PASS: zero candidates and reset after an empty search\n";
        }

        {
            Fixture<128, 10> small;
            for (size_t capacity : {size_t{32}, size_t{1024}}) {
                IVF_RaBitQ_Raw engine;
                initialize(engine, small, capacity, 1024, 0);
                engine.set_ip_kernel_type(IPKernelType::BITWISE_DECOMP);
                for (int round = 0; round < 3; ++round)
                    check(engine, small, small.queries);
            }
            std::cout << "PASS: bitwise producer, repeated wraps and exact payloads\n";
        }
        {
            omp_set_num_threads(4);
            IVF_RaBitQ_Raw engine;
            initialize(engine, f, 32, 0, 0);
            bool rejected = false;
            try {
                check(engine, f, f.queries);
            } catch (const std::runtime_error &error) {
                rejected =
                    std::string(error.what()).find("spill capacity exhausted") != std::string::npos;
            }
            require(rejected, "exhausted HBM spill returned incomplete results");
            engine.reset_buffer(f.query.data(), f.queries);
            std::vector<float> thresholds(f.queries, -std::numeric_limits<float>::max());
            engine.seed_thresholds(thresholds.data(), f.queries);
            std::vector<uint32_t> ids(f.queries * f.k);
            std::vector<float> distances(ids.size());
            engine.search(f.query.data(), f.queries, f.k, f.clusters, distances.data(), ids.data());
            require(engine.get_last_submitted_count() == 0,
                    "reset after overflow retained stale queue progress");
            std::cout
                << "PASS: HBM exhaustion is reported without hanging or returning partial results\n";
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
