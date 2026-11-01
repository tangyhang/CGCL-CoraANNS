#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace coraanns::queue {

// Match CUDA's unsigned-long-long atomicAdd overload.
using Position = unsigned long long;
static_assert(sizeof(Position) == 8);

struct alignas(64) Progress {
    alignas(64) Position released = 0;
    alignas(64) Position produced = 0;
    alignas(64) Position spilled = 0;
    alignas(64) int done = 0;
    alignas(64) int overflow = 0;
};

inline constexpr std::size_t default_capacity = std::size_t{1} << 26;
inline constexpr std::size_t default_spill_capacity = std::size_t{1} << 24;
inline constexpr std::size_t consumer_chunk = 8192;

// The entire publication unit fits in one naturally aligned vector access.
struct alignas(16) Candidate {
    std::uint16_t query_idx;
    std::uint16_t probe_idx;
    std::uint32_t gid;
    float exact_ip;
    float low_dist;
};
static_assert(sizeof(Candidate) == 16 && alignof(Candidate) == 16);
static_assert(std::is_trivially_copyable_v<Candidate>);

struct SpillRecord {
    Candidate item;
    Position position;
};
static_assert(sizeof(SpillRecord) == 32);

// Passed to kernels by value. items/progress are mapped-host device addresses;
// spill storage, reservation counters and cached watermarks reside in HBM.
struct Descriptor {
    Candidate *items = nullptr;
    Progress *progress = nullptr;
    Position capacity = 0;
    Position mask = 0;
    SpillRecord *spill_items = nullptr;
    Position *spill_count = nullptr;
    Position spill_capacity = 0;
    int *spill_overflow = nullptr;
    Position *cached_write_limit = nullptr;
    int *refresh_lock = nullptr;
};
static_assert(std::is_trivially_copyable_v<Descriptor>);

} // namespace coraanns::queue
