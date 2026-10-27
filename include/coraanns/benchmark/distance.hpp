#pragma once

#include "rabitqlib/defines.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace graph_baseline {

using FloatMatrix = rabitqlib::RowMajorArray<float>;
using IdMatrix = rabitqlib::RowMajorArray<std::uint32_t>;

inline bool starts_with_option(const std::string &value) {
  return value.size() > 2 && value[0] == '-' && value[1] == '-';
}

class CliOptions {
public:
  CliOptions(int argc, char **argv, int start) {
    for (int i = start; i < argc; ++i) {
      std::string token(argv[i]);
      if (!starts_with_option(token)) {
        throw std::invalid_argument("Unexpected positional argument: " + token);
      }
      token.erase(0, 2);
      const auto equal = token.find('=');
      if (equal != std::string::npos) {
        values_[token.substr(0, equal)] = token.substr(equal + 1);
      } else {
        if (i + 1 >= argc || starts_with_option(argv[i + 1])) {
          throw std::invalid_argument("Missing value for --" + token);
        }
        values_[token] = argv[++i];
      }
    }
  }

  void require_known(std::initializer_list<const char *> names) const {
    for (const auto &entry : values_) {
      const bool known =
          std::any_of(names.begin(), names.end(),
                      [&](const char *name) { return entry.first == name; });
      if (!known)
        throw std::invalid_argument("Unknown option --" + entry.first);
    }
  }

  std::string get(const std::string &name, const std::string &fallback) const {
    const auto it = values_.find(name);
    return it == values_.end() ? fallback : it->second;
  }

  std::size_t get_size(const std::string &name, std::size_t fallback) const {
    const auto it = values_.find(name);
    if (it == values_.end())
      return fallback;
    if (it->second.empty() || it->second.front() == '-') {
      throw std::invalid_argument("Invalid non-negative integer for --" + name +
                                  ": " + it->second);
    }
    std::size_t consumed = 0;
    const auto value = std::stoull(it->second, &consumed);
    if (consumed != it->second.size()) {
      throw std::invalid_argument("Invalid integer for --" + name + ": " +
                                  it->second);
    }
    return static_cast<std::size_t>(value);
  }

private:
  std::unordered_map<std::string, std::string> values_;
};

inline double
elapsed_seconds(const std::chrono::steady_clock::time_point &start,
                const std::chrono::steady_clock::time_point &stop) {
  return std::chrono::duration<double>(stop - start).count();
}

inline bool has_suffix(const std::string &value, const std::string &suffix) {
  return value.size() >= suffix.size() &&
         value.compare(value.size() - suffix.size(), suffix.size(), suffix) ==
             0;
}

// A read-only view over the two FP32 layouts used by this project: fvecs and
// [uint32 rows][uint32 dim][row-major floats].  The latter covers .bin/.fbin.
class FloatVectorMMap {
public:
  explicit FloatVectorMMap(const std::string &path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0)
      fail("Cannot open " + path, errno);

    struct stat st {};
    if (::fstat(fd_, &st) != 0)
      fail("Cannot stat " + path, errno);
    if (st.st_size <= 0)
      fail("Empty vector file " + path);
    mapping_bytes_ = static_cast<std::size_t>(st.st_size);
    void *mapped =
        ::mmap(nullptr, mapping_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped == MAP_FAILED) {
      mapping_ = nullptr;
      fail("Cannot mmap " + path, errno);
    }
    mapping_ = static_cast<const std::uint8_t *>(mapped);

    if (has_suffix(path, "fvecs")) {
      if (mapping_bytes_ < sizeof(std::uint32_t))
        fail("Invalid fvecs file " + path);
      std::uint32_t dim = 0;
      std::memcpy(&dim, mapping_, sizeof(dim));
      dim_ = dim;
      data_offset_bytes_ = sizeof(std::uint32_t);
      row_stride_bytes_ = sizeof(std::uint32_t) + dim_ * sizeof(float);
      if (dim_ == 0 || mapping_bytes_ % row_stride_bytes_ != 0) {
        fail("Invalid fvecs shape in " + path);
      }
      rows_ = mapping_bytes_ / row_stride_bytes_;
      std::uint32_t last_dim = 0;
      std::memcpy(&last_dim, mapping_ + (rows_ - 1) * row_stride_bytes_,
                  sizeof(last_dim));
      if (last_dim != dim_)
        fail("Inconsistent fvecs dimensions in " + path);
    } else {
      if (mapping_bytes_ < 2 * sizeof(std::uint32_t)) {
        fail("Invalid FP32 bin file " + path);
      }
      std::uint32_t rows = 0;
      std::uint32_t dim = 0;
      std::memcpy(&rows, mapping_, sizeof(rows));
      std::memcpy(&dim, mapping_ + sizeof(rows), sizeof(dim));
      rows_ = rows;
      dim_ = dim;
      data_offset_bytes_ = 2 * sizeof(std::uint32_t);
      row_stride_bytes_ = dim_ * sizeof(float);
      const auto payload_bytes = mapping_bytes_ - data_offset_bytes_;
      if (rows_ == 0 || dim_ == 0 || payload_bytes % rows_ != 0 ||
          payload_bytes / rows_ != row_stride_bytes_) {
        fail("Invalid FP32 bin shape in " + path);
      }
    }
    ::madvise(const_cast<std::uint8_t *>(mapping_), mapping_bytes_,
              MADV_RANDOM);
  }

  FloatVectorMMap(const FloatVectorMMap &) = delete;
  FloatVectorMMap &operator=(const FloatVectorMMap &) = delete;

  ~FloatVectorMMap() {
    if (mapping_ != nullptr) {
      ::munmap(const_cast<std::uint8_t *>(mapping_), mapping_bytes_);
    }
    if (fd_ >= 0)
      ::close(fd_);
  }

  std::size_t rows() const { return rows_; }
  std::size_t dim() const { return dim_; }

  const float *row(std::size_t index) const {
    if (index >= rows_)
      throw std::out_of_range("Base vector ID is out of range");
    return reinterpret_cast<const float *>(mapping_ + data_offset_bytes_ +
                                           index * row_stride_bytes_);
  }

private:
  [[noreturn]] void fail(const std::string &prefix, int error_number = 0) {
    auto message = prefix;
    if (error_number != 0)
      message += ": " + std::string(std::strerror(error_number));
    if (mapping_ != nullptr) {
      ::munmap(const_cast<std::uint8_t *>(mapping_), mapping_bytes_);
      mapping_ = nullptr;
    }
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
    throw std::runtime_error(message);
  }

  int fd_ = -1;
  const std::uint8_t *mapping_ = nullptr;
  std::size_t mapping_bytes_ = 0;
  std::size_t rows_ = 0;
  std::size_t dim_ = 0;
  std::size_t data_offset_bytes_ = 0;
  std::size_t row_stride_bytes_ = 0;
};

inline float squared_l2(const float *lhs, const float *rhs, std::size_t dim) {
  float distance = 0.0f;
#pragma omp simd reduction(+ : distance)
  for (std::size_t d = 0; d < dim; ++d) {
    const float delta = lhs[d] - rhs[d];
    distance += delta * delta;
  }
  return distance;
}

inline void exact_rerank_l2(const FloatVectorMMap &base, const float *queries,
                            std::size_t query_count, std::size_t dim,
                            const std::uint32_t *candidates,
                            std::size_t candidate_count, std::size_t topk,
                            std::uint32_t *result_ids,
                            float *result_distances) {
  if (candidate_count < topk) {
    throw std::invalid_argument("candidate_count must be at least topk");
  }

#pragma omp parallel
  {
    std::vector<std::pair<float, std::uint32_t>> scored;
    scored.reserve(candidate_count);

#pragma omp for
    for (std::int64_t q = 0; q < static_cast<std::int64_t>(query_count); ++q) {
      scored.clear();
      std::unordered_set<std::uint32_t> seen;
      seen.reserve(candidate_count * 2);
      const auto *query = queries + static_cast<std::size_t>(q) * dim;
      const auto *row_candidates =
          candidates + static_cast<std::size_t>(q) * candidate_count;
      for (std::size_t c = 0; c < candidate_count; ++c) {
        const auto id = row_candidates[c];
        if (id >= base.rows() || !seen.insert(id).second)
          continue;
        scored.emplace_back(squared_l2(query, base.row(id), dim), id);
      }

      const auto keep = std::min(topk, scored.size());
      std::partial_sort(scored.begin(), scored.begin() + keep, scored.end(),
                        [](const auto &lhs, const auto &rhs) {
                          if (lhs.first != rhs.first)
                            return lhs.first < rhs.first;
                          return lhs.second < rhs.second;
                        });
      for (std::size_t k = 0; k < keep; ++k) {
        result_distances[static_cast<std::size_t>(q) * topk + k] =
            scored[k].first;
        result_ids[static_cast<std::size_t>(q) * topk + k] = scored[k].second;
      }
      for (std::size_t k = keep; k < topk; ++k) {
        result_distances[static_cast<std::size_t>(q) * topk + k] =
            std::numeric_limits<float>::infinity();
        result_ids[static_cast<std::size_t>(q) * topk + k] =
            std::numeric_limits<std::uint32_t>::max();
      }
    }
  }
}

} // namespace graph_baseline
