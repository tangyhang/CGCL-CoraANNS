#pragma once

#include "rabitqlib/index/ivf/ivf.hpp"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

namespace coraanns::index {

// Packed arrays borrowed by HybridIndex; keep both this adapter and IVF alive
// for the lifetime of the engine. Extended-code ownership transfers from IVF.
class SearchData {
  public:
    SearchData(rabitqlib::ivf::IVF &source, bool include_extended, int cpu_threads);
    std::vector<float> h_centroids, host_rotator, host_short_factors, host_ex_factors;
    std::vector<uint32_t> host_short_data;
    std::vector<size_t> cluster_sizes;
    uint8_t *host_long_codes = nullptr;
    size_t max_cluster_size = 0;

  private:
    struct FreeAllocation {
        void operator()(char *data) const noexcept {
            std::free(data);
        }
    };
    std::unique_ptr<char, FreeAllocation> extended_storage_;
};

} // namespace coraanns::index
