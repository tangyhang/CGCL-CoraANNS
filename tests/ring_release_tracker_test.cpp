#include "coraanns/queue/release_tracker.hpp"

#include <iostream>
#include <atomic>
#include <stdexcept>
#include <thread>
#include <vector>

static void require(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}

int main() {
    try {
        coraanns::queue::Progress progress;
        coraanns::queue::ReleaseTracker tracker(progress, 16, 4);
        tracker.complete(4);
        tracker.complete(12);
        require(tracker.released() == 0, "out-of-order completion released a live slot");
        require(!tracker.readable_generation(16), "next generation became readable early");
        tracker.complete(0);
        require(tracker.released() == 8, "contiguous prefix did not advance");
        tracker.complete(8);
        require(tracker.released() == 16, "completed prefix was lost");

        {
            coraanns::queue::Progress delayed;
            coraanns::queue::ReleaseTracker release(delayed, 16, 4);
            std::atomic<bool> finish_first{false};
            std::thread slow([&] {
                while (!finish_first.load(std::memory_order_acquire))
                    std::this_thread::yield();
                release.complete(0);
            });
            std::vector<std::thread> later;
            for (auto start : {4ULL, 8ULL, 12ULL})
                later.emplace_back([&, start] { release.complete(start); });
            for (auto &worker : later)
                worker.join();
            const bool held = release.released() == 0 && !release.readable_generation(16);
            finish_first.store(true, std::memory_order_release);
            slow.join();
            require(held, "a slow first consumer lost ownership of its live slots");
            require(release.released() == 16, "slow completion did not release the prefix");
        }
        for (unsigned long long base = 16; base < 160000; base += 16) {
            tracker.complete(base + 12);
            tracker.complete(base + 8);
            tracker.complete(base + 4);
            require(tracker.released() == base, "old completion tag was reused");
            tracker.complete(base);
            require(tracker.released() == base + 16, "wraparound prefix failed");
        }

        progress.released = 1ULL << 40;
        coraanns::queue::ReleaseTracker wide(progress, 16, 4);
        wide.complete((1ULL << 40) + 4);
        wide.complete(1ULL << 40);
        require(wide.released() == (1ULL << 40) + 8, "64-bit positions were truncated");

        progress.released = 0;
        coraanns::queue::ReleaseTracker concurrent(progress, 64, 4);
        for (unsigned long long base = 0; base < 64000; base += 64) {
            std::vector<std::thread> workers;
            for (unsigned i = 0; i < 8; ++i) {
                workers.emplace_back([&, i, base] {
                    concurrent.complete(base + (15 - i) * 4);
                    concurrent.complete(base + i * 4);
                });
            }
            for (auto &worker : workers)
                worker.join();
            require(concurrent.released() == base + 64, "concurrent completion lost progress");
        }
        bool rejected = false;
        try {
            coraanns::queue::ReleaseTracker invalid(progress, 16, 0);
        } catch (const std::invalid_argument &) {
            rejected = true;
        }
        require(rejected, "invalid chunk size was accepted");
        std::cout
            << "PASS: out-of-order completion, repeated wraps, 64-bit counters, concurrent release\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
