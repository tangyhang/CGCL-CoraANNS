#pragma once

#include "coraanns/search/hybrid.cuh"

namespace coraanns::detail {

static constexpr int WARP_SIZE = 32;
constexpr size_t FAST_SIZE = 32;

static constexpr int BITS_PER_CHUNK = 4;
static constexpr int LUT_SIZE = (1 << BITS_PER_CHUNK);

constexpr int RABITQ_BQ = 8;

using T = float;
using IdxT = uint32_t;
typedef uint32_t PID;

// Schedule vanguard pairs first, then ordinary pairs by cluster/probe/query.
struct ClusterQueryPair {
    int cluster_idx;
    int query_idx;
    int probe_idx;
};

using GlobalTaskItem = coraanns::queue::Candidate;
using QueueIndex = coraanns::queue::Position;
using SpillTaskItem = coraanns::queue::SpillRecord;
using GlobalAsyncBuffer = coraanns::queue::Descriptor;
constexpr size_t DEFAULT_QUEUE_CAPACITY = coraanns::queue::default_capacity;
constexpr size_t DEFAULT_QUEUE_SPILL_CAPACITY = coraanns::queue::default_spill_capacity;
constexpr size_t QUEUE_CONSUMER_CHUNK = coraanns::queue::consumer_chunk;

struct PairSortCompare {
    size_t vanguard_size;

    __host__ __device__ explicit PairSortCompare(size_t size = default_vanguard_size)
        : vanguard_size(size) {}

    __host__ __device__ bool operator()(const ClusterQueryPair &a,
                                        const ClusterQueryPair &b) const {

        bool a_is_vanguard = (static_cast<size_t>(a.probe_idx) < vanguard_size);
        bool b_is_vanguard = (static_cast<size_t>(b.probe_idx) < vanguard_size);

        if (a_is_vanguard != b_is_vanguard) {
            return a_is_vanguard;
        }

        if (a.cluster_idx != b.cluster_idx) {
            return a.cluster_idx < b.cluster_idx;
        }

        if (a.probe_idx != b.probe_idx) {
            return a.probe_idx < b.probe_idx;
        }

        return a.query_idx < b.query_idx;
    }
};

class GPUClusterMeta {
  public:
    size_t num;
    size_t iter;
    size_t remain;
    size_t start_index;

    __host__ __device__ GPUClusterMeta(size_t num, size_t start_idx)
        : num(num), start_index(start_idx) {
        iter = num / FAST_SIZE;
        remain = num - iter * FAST_SIZE;
    }
    GPUClusterMeta() : num(0), iter(0), remain(0), start_index(0) {}
    GPUClusterMeta(const GPUClusterMeta &other) = default;
    GPUClusterMeta &operator=(const GPUClusterMeta &other) = default;
    ~GPUClusterMeta() = default;
};

struct ComputeBitwiseKernelParams {
    const ClusterQueryPair *d_sorted_pairs;
    const GPUClusterMeta *d_cluster_meta;
    const float *d_query;
    const float *d_G_k1xSumq;
    const float *d_centroid_distances;
    const uint32_t *d_short_data;
    const float *d_short_factors;
    const half *d_global_luts;

    const uint32_t *d_q_bits;
    const float *d_q_scale;
    const float *d_q_min;

    int num_queries;
    int num_centroids;
    int D;
    int max_candidates_per_pair;
    int num_pairs;
    int topk;
};

} // namespace coraanns::detail
