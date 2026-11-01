#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <vector>
#include "coraanns/queue/candidate_queue.hpp"

namespace coraanns::queue {

// A claim distributes work. Only a contiguous prefix of cleared chunks
// releases storage to the GPU. Full logical end tags distinguish generations.
class ReleaseTracker {
  public:
    ReleaseTracker(Progress &progress, std::size_t capacity, std::size_t chunk_size)
        : progress_(progress), capacity_(capacity), chunk_size_(chunk_size) {
        if (!is_power_of_two(capacity) || !is_power_of_two(chunk_size) || chunk_size > capacity) {
            throw std::invalid_argument("invalid ring capacity/chunk size");
        }
        completed_.resize(capacity / chunk_size, 0);
    }

    Position released() const {
        return __atomic_load_n(&progress_.released, __ATOMIC_ACQUIRE);
    }

    bool readable_generation(Position start) const {
        return start < released() + capacity_;
    }

    // The caller clears the entire physical chunk first. The mutex imports
    // other workers' clears before this thread publishes their released prefix.
    void complete(Position start) {
        std::lock_guard<std::mutex> guard(mutex_);
        completed_[(start / chunk_size_) & (completed_.size() - 1)] = start + chunk_size_;
        const Position old = __atomic_load_n(&progress_.released, __ATOMIC_RELAXED);
        Position next = old;
        while (completed_[(next / chunk_size_) & (completed_.size() - 1)] == next + chunk_size_) {
            next += chunk_size_;
        }
        if (next != old) {
            __atomic_store_n(&progress_.released, next, __ATOMIC_RELEASE);
        }
    }

  private:
    static bool is_power_of_two(std::size_t value) {
        return value != 0 && (value & (value - 1)) == 0;
    }

    Progress &progress_;
    Position capacity_;
    Position chunk_size_;
    std::vector<Position> completed_;
    std::mutex mutex_;
};

} // namespace coraanns::queue
