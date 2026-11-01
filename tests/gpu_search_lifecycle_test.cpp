#include "coraanns/search/hybrid.cuh"
#include <functional>
#include <stdexcept>
#include <iostream>
#include <vector>

static void rejects(const std::function<void()> &action) {
    try {
        action();
    } catch (const std::logic_error &) {
        return;
    }
    throw std::runtime_error("invalid lifecycle or shape was accepted");
}
int main() {
    try {
        coraanns::HybridIndex engine;
        constexpr size_t dim = 128, count = 32, queries = 2;
        std::vector<float> q(queries * dim), centers(dim), rotation(dim * dim);
        std::vector<float> short_factors(count * 3, 0), factors(count * 2, 0);
        std::vector<uint32_t> codes(count * dim / 32), pids(count);
        std::vector<uint8_t> extended(count * dim * 7 / 8);
        for (size_t d = 0; d < dim; ++d)
            rotation[d * dim + d] = 1;
        for (size_t i = 0; i < count; ++i) {
            pids[i] = i;
            short_factors[3 * i] = 1e6f;
        }
        const size_t sizes[] = {count};
        float distances[20]{};
        uint32_t ids[20]{};
        rejects([&] { engine.prepare_workspace(queries, 10, 1); });
        rejects([&] { engine.reset_buffer(q.data(), queries); });
        rejects([&] { engine.search(q.data(), queries, 10, 1, distances, ids); });
        engine.load_from_raw_pointers(count, dim, dim, 1, 7, centers.data(), rotation.data(),
                                      codes.data(), short_factors.data(), extended.data(),
                                      factors.data(), pids.data(), sizes);
        engine.set_queue_capacity(32);
        engine.set_queue_spill_capacity(1024);
        engine.set_vanguard_size(0);
        rejects([&] { engine.prepare_workspace(65536, 10, 1); });
        engine.prepare_workspace(queries, 10, 1);
        rejects([&] { engine.reset_buffer(q.data(), queries + 1); });
        rejects([&] { engine.search(q.data(), queries + 1, 10, 1, distances, ids); });
        rejects([&] { engine.search(q.data(), queries, 11, 1, distances, ids); });
        rejects([&] { engine.search(q.data(), queries, 10, 2, distances, ids); });
        rejects([&] { engine.search(q.data(), queries, 10, 1, distances, ids, 7); });
        rejects([&] { engine.set_vanguard_size(1); });
        for (int iteration = 0; iteration < 3; ++iteration) {
            engine.reset_buffer(q.data(), queries);
            std::vector<float> threshold(queries, -1);
            engine.seed_thresholds(threshold.data(), queries);
            engine.search(q.data(), queries, 10, 1, distances, ids);
            engine.free_workspace();
            engine.free_workspace();
            rejects([&] { engine.reset_buffer(q.data(), queries); });
            engine.set_vanguard_size(iteration ? 0 : 1);
            engine.prepare_workspace(queries, 10, 1);
        }
        std::cout << "PASS: workspace bounds, unique ownership, repeated allocation and release\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
