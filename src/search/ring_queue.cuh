#pragma once

#include <cuda/atomic>
#include <cuda_runtime.h>
#include "coraanns/queue/candidate_queue.hpp"

namespace coraanns::queue::device {

// These words are allocated in HBM. Explicit global-space operations avoid
// generic-address atomic fallbacks and their local CAS temporary in CCCL.
__device__ __forceinline__ Position load_cached_limit(Position *address) {
    Position value;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];" : "=l"(value) : "l"(address) : "memory");
    return value;
}

__device__ __forceinline__ void store_cached_limit(Position *address, Position value) {
    asm volatile("st.release.gpu.global.u64 [%0], %1;" ::"l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ bool try_refresh_lock(int *address) {
    int previous;
    asm volatile("atom.acquire.gpu.global.cas.b32 %0, [%1], 0, 1;"
                 : "=r"(previous)
                 : "l"(address)
                 : "memory");
    return previous == 0;
}

__device__ __forceinline__ void release_refresh_lock(int *address) {
    asm volatile("st.release.gpu.global.u32 [%0], 0;" ::"l"(address) : "memory");
}

// Cache the writable limit in HBM across blocks. Only its refresh owner reads Pr.
__device__ __forceinline__ Position queue_released(const Descriptor &queue) {
    Position released;
    asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                 : "=l"(released)
                 : "l"(&queue.progress->released)
                 : "memory");
    return released;
}

// Keep the PCIe refresh cold path out of the producer's register live ranges.
__device__ __noinline__ Position refresh_cached_limit(Position *cached_limit, int *refresh_lock,
                                                      Progress *progress, Position capacity,
                                                      Position position) {
    Position limit = load_cached_limit(cached_limit);
    bool acquired = false;
    while (!acquired && position >= limit) {
        acquired = try_refresh_lock(refresh_lock);
        if (!acquired) {
            __nanosleep(64);
            limit = load_cached_limit(cached_limit);
        }
    }
    if (acquired) {
        limit = load_cached_limit(cached_limit);
        if (position >= limit) {
            Position released;
            asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
                         : "=l"(released)
                         : "l"(&progress->released)
                         : "memory");
            limit = released + capacity;
            store_cached_limit(cached_limit, limit);
        }
        release_refresh_lock(refresh_lock);
    }
    return limit;
}

__device__ __forceinline__ Position refresh_write_limit(const Descriptor &queue,
                                                        Position position) {
    const unsigned participants = __activemask();
    const int leader = __ffs(participants) - 1;
    Position limit = 0;
    if ((threadIdx.x & 31) == leader) {
        limit = load_cached_limit(queue.cached_write_limit);
        if (position >= limit) {
            limit = refresh_cached_limit(queue.cached_write_limit, queue.refresh_lock,
                                         queue.progress, queue.capacity, position);
        }
    }
    const unsigned lo = __shfl_sync(participants, unsigned(limit), leader);
    const unsigned hi = __shfl_sync(participants, unsigned(limit >> 32), leader);
    __syncwarp(participants);
    return (Position(hi) << 32) | lo;
}

__device__ __forceinline__ void publish_ring_item(Candidate *item, uint4 payload) {
    // Preserve one publication transaction for the CPU's aligned snapshot.
    asm volatile("st.global.v4.u32 [%0], {%1, %2, %3, %4};" ::"l"(item), "r"(payload.x),
                 "r"(payload.y), "r"(payload.z), "r"(payload.w)
                 : "memory");
}

__device__ __forceinline__ void publish_task(const Descriptor &queue, Position position,
                                             uint4 payload, Position &write_limit) {
    if (position >= write_limit) {
        write_limit = refresh_write_limit(queue, position);
        if (position >= write_limit) {
            const Position spill = atomicAdd(queue.spill_count, 1ULL);
            if (spill < queue.spill_capacity) {
                auto &task = queue.spill_items[spill];
                *reinterpret_cast<uint4 *>(&task.item) = payload;
                task.position = position;
            } else {
                atomicExch(queue.spill_overflow, 1);
            }
            return;
        }
    }
    publish_ring_item(&queue.items[position & queue.mask], payload);
}

__global__ void reset_queue(const Descriptor queue, Position *produced) {
    *produced = 0;
    *queue.spill_count = 0;
    *queue.spill_overflow = 0;
    *queue.cached_write_limit = queue.capacity;
    *queue.refresh_lock = 0;
}

__global__ void finish_queue(const Descriptor queue, const Position *produced) {
    __shared__ Position remaining;
    __shared__ Position write_limit;
    const Position spill_count = *queue.spill_count;
    if (threadIdx.x == 0) {
        queue.progress->produced = *produced;
        queue.progress->spilled = spill_count;
    }
    __syncthreads();
    if (*queue.spill_overflow) {
        if (threadIdx.x == 0) {
            cuda::atomic_ref<int, cuda::thread_scope_system>(queue.progress->overflow)
                .store(1, cuda::memory_order_release);
        }
    } else {
        // Spill append order can differ from logical position order. Scan all
        // pending tickets so a later generation cannot block an earlier hole.
        do {
            if (threadIdx.x == 0) {
                remaining = 0;
                write_limit = queue_released(queue) + queue.capacity;
            }
            __syncthreads();
            Position local_remaining = 0;
            for (Position i = threadIdx.x; i < spill_count; i += blockDim.x) {
                auto &task = queue.spill_items[i];
                const Position position = task.position;
                if (position == ~Position{0})
                    continue;
                if (position < write_limit) {
                    publish_ring_item(&queue.items[position & queue.mask],
                                      *reinterpret_cast<const uint4 *>(&task.item));
                    task.position = ~Position{0};
                } else {
                    ++local_remaining;
                }
            }
            atomicAdd(&remaining, local_remaining);
            __syncthreads();
        } while (remaining != 0);
    }
    if (threadIdx.x == 0) {
        cuda::atomic_ref<int, cuda::thread_scope_system>(queue.progress->done)
            .store(1, cuda::memory_order_release);
    }
}

} // namespace coraanns::queue::device
