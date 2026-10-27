#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "rabitqlib/defines.hpp"
#include "rabitqlib/utils/space.hpp"
#include "rabitqlib/third/hnswlib/hnswlib.h"

namespace rabitqlib::ivf {
inline constexpr size_t kHnswInitializerThreshold = 20000;

[[nodiscard]] inline constexpr bool use_hnsw_initializer(size_t num_clusters) {
    return num_clusters >= kHnswInitializerThreshold;
}

enum class InitializerStorage {
    EmbeddedCentroids,
    HnswSidecar,
};

template <class Function>
inline void parallel_for(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr last_exception = nullptr;
        std::mutex last_except_mutex;

        threads.reserve(numThreads);
        for (size_t thread_id = 0; thread_id < numThreads; ++thread_id) {
            threads.push_back(std::thread([&, thread_id] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, thread_id);
                    } catch (...) {
                        std::unique_lock<std::mutex> last_except_lock(last_except_mutex);
                        last_exception = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto& thread : threads) {
            thread.join();
        }
        if (last_exception) {
            std::rethrow_exception(last_exception);
        }
    }
}

/**
 * @brief For ivf centroids, we need an intializer to get the candidate clusters.
 */
class Initializer {
   protected:
    size_t dim_;
    size_t num_cluster_;

   public:
    explicit Initializer(size_t d, size_t k) : dim_(d), num_cluster_(k) {}
    virtual ~Initializer() = 0;
    [[nodiscard]] virtual const float* centroid(PID) const = 0;
    virtual void add_vectors(const float*) = 0;
    virtual void centroids_distances(
        const float*, size_t, std::vector<AnnCandidate<float>>&
    ) const = 0;
    virtual void load(std::ifstream&, const char*, InitializerStorage) = 0;
    virtual void save(std::ofstream&, const char*) const = 0;
};
inline Initializer::~Initializer() {}

class FlatInitializer : public Initializer {
   private:
    std::vector<float> centroids_;

   public:
    explicit FlatInitializer(size_t d, size_t k)
        : Initializer(d, k), centroids_(num_cluster_ * dim_) {}

    ~FlatInitializer() override = default;

    [[nodiscard]] const float* centroid(PID id) const override {
        return &centroids_[id * dim_];
    }

    void add_vectors(const float* cent) override {
        std::memcpy(centroids_.data(), cent, sizeof(float) * num_cluster_ * dim_);
    }

    void centroids_distances(
        const float* query, size_t nprobe, std::vector<AnnCandidate<float>>& candidates
    ) const override {
        std::vector<AnnCandidate<float>> centroid_dist(this->num_cluster_);
        for (PID i = 0; i < num_cluster_; ++i) {
            centroid_dist[i].id = i;
            centroid_dist[i].distance = std::sqrt(euclidean_sqr(query, centroid(i), dim_));
        }
        std::partial_sort(
            centroid_dist.begin(),
            centroid_dist.begin() + static_cast<long>(nprobe),
            centroid_dist.end()
        );

        std::memcpy(
            candidates.data(), centroid_dist.data(), sizeof(AnnCandidate<float>) * nprobe
        );
    }

    // for flat initer, we save & load into the ifstream
    void save(std::ofstream& output, const char*) const override {
        output.write(
            reinterpret_cast<const char*>(centroids_.data()),
            static_cast<long>(sizeof(float) * dim_ * num_cluster_)
        );
    }

    void load(
        std::ifstream& input, const char*, InitializerStorage storage
    ) override {
        if (storage != InitializerStorage::EmbeddedCentroids) {
            throw std::runtime_error(
                "Cannot load an HNSW-only centroid index with the flat initializer"
            );
        }
        input.read(
            reinterpret_cast<char*>(centroids_.data()),
            static_cast<long>(sizeof(float) * dim_ * num_cluster_)
        );
        if (!input) {
            throw std::runtime_error("Failed to read IVF centroids");
        }
    }
};

class HNSWInitializer : public Initializer {
   private:
    int M_ = 16;
    int ef_construction_ = 400;
    hnswlib::HierarchicalNSW<float>* alg_hnsw_ = nullptr;
    hnswlib::L2Space space_;
    std::vector<hnswlib::tableint> label_to_internal_;

    void rebuild_label_to_internal() {
        const auto count = alg_hnsw_->getCurrentElementCount();
        if (count != num_cluster_) {
            throw std::runtime_error(
                "HNSW centroid count does not match the IVF cluster count"
            );
        }

        const auto missing = std::numeric_limits<hnswlib::tableint>::max();
        label_to_internal_.assign(num_cluster_, missing);
        for (hnswlib::tableint internal_id = 0; internal_id < count; ++internal_id) {
            const auto label = alg_hnsw_->getExternalLabel(internal_id);
            if (label >= num_cluster_ || label_to_internal_[label] != missing) {
                throw std::runtime_error("Invalid or duplicate HNSW centroid label");
            }
            label_to_internal_[label] = internal_id;
        }
    }

   public:
    explicit HNSWInitializer(size_t d, size_t k) : Initializer(d, k), space_(d) {
        alg_hnsw_ = new hnswlib::HierarchicalNSW<float>(
            &space_, num_cluster_, M_, ef_construction_
        );
    }

    void add_vectors(const float* cent) override {
        std::cout << "Inserting vectors into hnsw...\n";
        size_t start = 0;
        size_t rows = num_cluster_;
        size_t num_threads = 0;
        parallel_for(start, rows, num_threads, [&](size_t row, size_t /*thread_id*/) {
            alg_hnsw_->addPoint(cent + (row * dim_), row);
        });
        rebuild_label_to_internal();
        std::cout << "Inserted vectors into hnsw...\n" << std::flush;
    }

    [[nodiscard]] const float* centroid(PID id) const override {
        if (id >= label_to_internal_.size()) {
            throw std::out_of_range("IVF centroid label is out of range");
        }
        return reinterpret_cast<const float*>(
            alg_hnsw_->getDataByInternalId(label_to_internal_[id])
        );
    }

    void centroids_distances(
        const float* query, size_t nprobe, std::vector<AnnCandidate<float>>& candidates
    ) const override {
        const size_t doubled_nprobe =
            nprobe > std::numeric_limits<size_t>::max() / 2 ? nprobe : 2 * nprobe;
        const size_t search_ef = std::max<size_t>(768, doubled_nprobe);
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            alg_hnsw_->searchKnnWithEf(query, nprobe, search_ef);

        if (result.size() != nprobe) {
            throw std::runtime_error("HNSW returned fewer IVF centroids than requested");
        }

        // hnswlib's priority queue exposes the farthest result first.  Fill the
        // output backwards so IVF scans candidate clusters nearest first, just
        // like the flat initializer.
        for (size_t i = nprobe; i > 0; --i) {
            candidates[i - 1].distance = std::sqrt(result.top().first);
            candidates[i - 1].id = result.top().second;
            result.pop();
        }
    }

    // for hnsw initer, we save & load into a separate file by hnswlib
    void save(std::ofstream&, const char* filename) const override {
        std::string hnsw(filename);
        hnsw += ".hnsw";
        alg_hnsw_->saveIndex(hnsw);
    }

    void load(
        std::ifstream& input, const char* filename, InitializerStorage storage
    ) override {
        if (storage == InitializerStorage::EmbeddedCentroids) {
            std::vector<float> centroids(num_cluster_ * dim_);
            input.read(
                reinterpret_cast<char*>(centroids.data()),
                static_cast<long>(sizeof(float) * centroids.size())
            );
            if (!input) {
                throw std::runtime_error("Failed to read IVF centroids");
            }
            std::cout
                << "Building HNSW centroid index from embedded centroids...\n";
            add_vectors(centroids.data());
        } else {
            std::string hnsw(filename);
            hnsw += ".hnsw";
            alg_hnsw_->loadIndex(hnsw, &space_, num_cluster_);
            rebuild_label_to_internal();
        }
    }

    ~HNSWInitializer() override { delete alg_hnsw_; }
};
}  // namespace rabitqlib::ivf
