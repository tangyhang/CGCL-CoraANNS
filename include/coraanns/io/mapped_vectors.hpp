#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace coraanns::io {

inline bool has_suffix(const std::string &value, const std::string &suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// A read-only view over the two FP32 layouts used by this project: fvecs and
// [uint32 rows][uint32 dim][row-major floats].  The latter covers .bin/.fbin.
class MappedVectors {
  public:
    explicit MappedVectors(const std::string &path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0)
            fail("Cannot open " + path, errno);

        struct stat st {};
        if (::fstat(fd_, &st) != 0)
            fail("Cannot stat " + path, errno);
        if (st.st_size <= 0)
            fail("Empty vector file " + path);
        mapping_bytes_ = static_cast<std::size_t>(st.st_size);
        void *mapped = ::mmap(nullptr, mapping_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
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
            std::memcpy(&last_dim, mapping_ + (rows_ - 1) * row_stride_bytes_, sizeof(last_dim));
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
        ::madvise(const_cast<std::uint8_t *>(mapping_), mapping_bytes_, MADV_RANDOM);
    }

    MappedVectors(const MappedVectors &) = delete;
    MappedVectors &operator=(const MappedVectors &) = delete;

    ~MappedVectors() {
        if (mapping_ != nullptr) {
            ::munmap(const_cast<std::uint8_t *>(mapping_), mapping_bytes_);
        }
        if (fd_ >= 0)
            ::close(fd_);
    }

    std::size_t rows() const {
        return rows_;
    }
    std::size_t dim() const {
        return dim_;
    }
    const std::uint8_t *mapping() const {
        return mapping_;
    }
    std::size_t row_stride_bytes() const {
        return row_stride_bytes_;
    }
    std::size_t data_offset_bytes() const {
        return data_offset_bytes_;
    }

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

} // namespace coraanns::io
