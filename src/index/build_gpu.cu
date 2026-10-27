#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/system/cuda/execution_policy.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "rabitqlib/index/ivf/ivf.hpp"

namespace {

constexpr unsigned int kQuantizeBlockSize = 256;
constexpr size_t kDefaultBatchSize = 32768;
using TimingClock = std::chrono::steady_clock;

double elapsed_seconds(TimingClock::time_point begin) {
    return std::chrono::duration<double>(TimingClock::now() - begin).count();
}

[[noreturn]] void throw_cuda_error(cudaError_t status, const char* operation) {
    throw std::runtime_error(
        std::string(operation) + ": " + cudaGetErrorString(status)
    );
}

void cuda_check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw_cuda_error(status, operation);
    }
}

void cublas_check(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(operation) + " failed with cuBLAS status " +
            std::to_string(static_cast<int>(status))
        );
    }
}

class CudaStream {
   public:
    CudaStream() { cuda_check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
                              "cudaStreamCreateWithFlags"); }
    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    operator cudaStream_t() const { return stream_; }

   private:
    cudaStream_t stream_ = nullptr;
};

class CublasHandle {
   public:
    explicit CublasHandle(cudaStream_t stream) {
        cublas_check(cublasCreate(&handle_), "cublasCreate");
        cublas_check(cublasSetStream(handle_, stream), "cublasSetStream");
    }
    ~CublasHandle() {
        if (handle_ != nullptr) {
            cublasDestroy(handle_);
        }
    }
    CublasHandle(const CublasHandle&) = delete;
    CublasHandle& operator=(const CublasHandle&) = delete;
    operator cublasHandle_t() const { return handle_; }

   private:
    cublasHandle_t handle_ = nullptr;
};

template <typename T>
class DeviceBuffer {
   public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { allocate(count); }
    ~DeviceBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void allocate(size_t count) {
        if (count == 0) {
            return;
        }
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::overflow_error("CUDA allocation size overflow");
        }
        cuda_check(
            cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
            "cudaMalloc"
        );
        count_ = count;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return count_; }

   private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

template <typename T>
class PinnedBuffer {
   public:
    explicit PinnedBuffer(size_t count) : count_(count) {
        if (count == 0) {
            return;
        }
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::overflow_error("pinned allocation size overflow");
        }
        cuda_check(
            cudaHostAlloc(
                reinterpret_cast<void**>(&data_), count * sizeof(T), cudaHostAllocDefault
            ),
            "cudaHostAlloc"
        );
    }
    ~PinnedBuffer() {
        if (data_ != nullptr) {
            cudaFreeHost(data_);
        }
    }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return count_; }

   private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

size_t checked_product(size_t lhs, size_t rhs, const char* description) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw std::overflow_error(std::string(description) + " size overflow");
    }
    return lhs * rhs;
}

int checked_cublas_int(size_t value, const char* description) {
    if (value > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(description) + " exceeds cuBLAS int range");
    }
    return static_cast<int>(value);
}

size_t gpu_batch_size() {
    const char* value = std::getenv("RABITQ_GPU_BATCH_SIZE");
    if (value == nullptr || *value == '\0') {
        return kDefaultBatchSize;
    }

    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed == 0) {
        throw std::runtime_error(
            "RABITQ_GPU_BATCH_SIZE must be a positive integer"
        );
    }
    if (parsed > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("RABITQ_GPU_BATCH_SIZE is too large");
    }

    size_t result = static_cast<size_t>(parsed);
    result = std::max(result, rabitqlib::fastscan::kBatchSize);
    result -= result % rabitqlib::fastscan::kBatchSize;
    return result;
}

__global__ void histogram_cluster_ids_kernel(
    const rabitqlib::PID* cluster_ids,
    size_t num_points,
    size_t num_clusters,
    unsigned long long* counts,
    int* invalid
) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         i < num_points;
         i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const auto cid = static_cast<size_t>(cluster_ids[i]);
        if (cid < num_clusters) {
            atomicAdd(counts + cid, 1ULL);
        } else {
            atomicExch(invalid, 1);
        }
    }
}

template <unsigned int BlockSize>
__device__ float block_sum(float value) {
    __shared__ float warp_sums[32];
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int warp = threadIdx.x >> 5U;

    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    value = threadIdx.x < (BlockSize + 31) / 32 ? warp_sums[lane] : 0.0F;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) {
            warp_sums[0] = value;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

template <unsigned int BlockSize>
__device__ double block_sum(double value) {
    __shared__ double warp_sums[32];
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int warp = threadIdx.x >> 5U;

    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset);
    }
    if (lane == 0) {
        warp_sums[warp] = value;
    }
    __syncthreads();

    value = threadIdx.x < (BlockSize + 31) / 32 ? warp_sums[lane] : 0.0;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffU, value, offset);
        }
        if (lane == 0) {
            warp_sums[0] = value;
        }
    }
    __syncthreads();
    return warp_sums[0];
}

template <unsigned int BlockSize>
__device__ float approximate_rescale_factor(
    const float* normalized_abs, int dim, int ex_bits, float* scratch
) {
    constexpr float kEpsilon = 1e-5F;
    constexpr int kNEnum = 10;
    constexpr float kTightStart[9] = {
        0.0F, 0.15F, 0.20F, 0.52F, 0.59F, 0.71F, 0.75F, 0.77F, 0.81F
    };

    float local_max = 0.0F;
    for (int i = static_cast<int>(threadIdx.x); i < dim; i += BlockSize) {
        local_max = fmaxf(local_max, normalized_abs[i]);
    }
    scratch[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned int stride = BlockSize / 2; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] =
                fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    const float max_value = scratch[0];
    if (max_value < kEpsilon) {
        return 1.0F;
    }

    const int max_code = (1 << ex_bits) - 1;
    const float t_end = static_cast<float>(max_code + kNEnum) / max_value;
    const float t_start = t_end * kTightStart[ex_bits];

    constexpr int kCoarseSamples = 64;
    float best_ip = 0.0F;
    float best_t = t_start;
    for (int sample = static_cast<int>(threadIdx.x);
         sample < kCoarseSamples;
         sample += BlockSize) {
        const float t = t_start +
                        (t_end - t_start) * static_cast<float>(sample) /
                            static_cast<float>(kCoarseSamples - 1);
        float numerator = 0.0F;
        float denominator_sqr = static_cast<float>(dim) * 0.25F;
        for (int j = 0; j < dim; ++j) {
            int q = static_cast<int>(t * normalized_abs[j] + kEpsilon);
            q = q < max_code ? q : max_code;
            numerator += (static_cast<float>(q) + 0.5F) * normalized_abs[j];
            denominator_sqr += static_cast<float>(q * q + q);
        }
        const float ip = numerator / sqrtf(denominator_sqr);
        if (ip > best_ip) {
            best_ip = ip;
            best_t = t;
        }
    }

    float* ips = scratch + BlockSize;
    float* ts = ips + BlockSize;
    ips[threadIdx.x] = best_ip;
    ts[threadIdx.x] = best_t;
    __syncthreads();
    for (unsigned int stride = BlockSize / 2; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride && ips[threadIdx.x + stride] > ips[threadIdx.x]) {
            ips[threadIdx.x] = ips[threadIdx.x + stride];
            ts[threadIdx.x] = ts[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float coarse_t = ts[0];
    const float radius = (t_end - t_start) / static_cast<float>(kCoarseSamples);
    const float fine_start = fmaxf(t_start, coarse_t - radius);
    const float fine_end = fminf(t_end, coarse_t + radius);

    constexpr int kFineSamples = 32;
    best_ip = 0.0F;
    best_t = coarse_t;
    for (int sample = static_cast<int>(threadIdx.x);
         sample < kFineSamples;
         sample += BlockSize) {
        const float t = fine_start +
                        (fine_end - fine_start) * static_cast<float>(sample) /
                            static_cast<float>(kFineSamples - 1);
        float numerator = 0.0F;
        float denominator_sqr = static_cast<float>(dim) * 0.25F;
        for (int j = 0; j < dim; ++j) {
            int q = static_cast<int>(t * normalized_abs[j] + kEpsilon);
            q = q < max_code ? q : max_code;
            numerator += (static_cast<float>(q) + 0.5F) * normalized_abs[j];
            denominator_sqr += static_cast<float>(q * q + q);
        }
        const float ip = numerator / sqrtf(denominator_sqr);
        if (ip > best_ip) {
            best_ip = ip;
            best_t = t;
        }
    }

    ips[threadIdx.x] = best_ip;
    ts[threadIdx.x] = best_t;
    __syncthreads();
    for (unsigned int stride = BlockSize / 2; stride > 0; stride >>= 1U) {
        if (threadIdx.x < stride && ips[threadIdx.x + stride] > ips[threadIdx.x]) {
            ips[threadIdx.x] = ips[threadIdx.x + stride];
            ts[threadIdx.x] = ts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return ts[0];
}

template <unsigned int BlockSize>
__global__ void quantize_rows_kernel(
    const float* rotated_data,
    const float* rotated_centroids,
    const rabitqlib::PID* cluster_ids,
    size_t num_rows,
    int padded_dim,
    int ex_bits,
    int metric_type,
    bool faster,
    double constant_rescale,
    uint8_t* binary_codes,
    float* short_factors,
    uint8_t* raw_ex_codes,
    float* ex_factors
) {
    const size_t row = blockIdx.x;
    if (row >= num_rows) {
        return;
    }

    const float* vector = rotated_data + row * static_cast<size_t>(padded_dim);
    const float* centroid =
        rotated_centroids +
        static_cast<size_t>(cluster_ids[row]) * static_cast<size_t>(padded_dim);

    float l2_sqr = 0.0F;
    float ip_residual_xu = 0.0F;
    float ip_centroid_xu = 0.0F;
    float residual_centroid_ip = 0.0F;
    for (int j = static_cast<int>(threadIdx.x); j < padded_dim; j += BlockSize) {
        const float residual = vector[j] - centroid[j];
        const float xu = residual > 0.0F ? 0.5F : -0.5F;
        l2_sqr += residual * residual;
        ip_residual_xu += residual * xu;
        ip_centroid_xu += centroid[j] * xu;
        residual_centroid_ip += residual * centroid[j];
    }
    l2_sqr = block_sum<BlockSize>(l2_sqr);
    ip_residual_xu = block_sum<BlockSize>(ip_residual_xu);
    ip_centroid_xu = block_sum<BlockSize>(ip_centroid_xu);
    residual_centroid_ip = block_sum<BlockSize>(residual_centroid_ip);

    if (threadIdx.x == 0) {
        float denominator = ip_residual_xu;
        if (denominator == 0.0F) {
            denominator = INFINITY;
        }
        const float l2_norm = sqrtf(fmaxf(l2_sqr, 0.0F));
        const float xu_sqr = static_cast<float>(padded_dim) * 0.25F;
        const float ratio = (l2_sqr * xu_sqr) / (denominator * denominator);
        const float inner =
            (ratio - 1.0F) / fmaxf(static_cast<float>(padded_dim - 1), 1.0F);
        const float error = l2_norm * 1.9F * sqrtf(fmaxf(inner, 0.0F));

        float f_add = 0.0F;
        float f_rescale = 0.0F;
        float f_error = 0.0F;
        if (metric_type == static_cast<int>(rabitqlib::METRIC_L2)) {
            f_add = l2_sqr + 2.0F * l2_sqr * ip_centroid_xu / denominator;
            f_rescale = -2.0F * l2_sqr / denominator;
            f_error = 2.0F * error;
        } else {
            f_add = 1.0F - residual_centroid_ip +
                    l2_sqr * ip_centroid_xu / denominator;
            f_rescale = -l2_sqr / denominator;
            f_error = error;
        }
        short_factors[row * 3 + 0] = f_add;
        short_factors[row * 3 + 1] = f_rescale;
        short_factors[row * 3 + 2] = f_error;
    }

    const int binary_bytes = padded_dim / 8;
    for (int byte = static_cast<int>(threadIdx.x);
         byte < binary_bytes;
         byte += BlockSize) {
        uint8_t packed = 0;
#pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            const int j = byte * 8 + bit;
            const bool positive = vector[j] - centroid[j] > 0.0F;
            packed |= static_cast<uint8_t>(positive) << (7 - bit);
        }
        binary_codes[row * static_cast<size_t>(binary_bytes) + byte] = packed;
    }

    if (ex_bits == 0) {
        return;
    }

    extern __shared__ float shared[];
    float* normalized_abs = shared;
    float* scratch = normalized_abs + padded_dim;
    const float l2_norm = sqrtf(fmaxf(l2_sqr, 0.0F));
    for (int j = static_cast<int>(threadIdx.x); j < padded_dim; j += BlockSize) {
        const float residual = vector[j] - centroid[j];
        normalized_abs[j] = l2_norm > 0.0F ? fabsf(residual) / l2_norm : 0.0F;
    }
    __syncthreads();

    const double rescale = faster
                               ? constant_rescale
                               : static_cast<double>(approximate_rescale_factor<BlockSize>(
                                     normalized_abs, padded_dim, ex_bits, scratch
                                 ));
    const int mask = (1 << ex_bits) - 1;
    double local_ipnorm = 0.0;
    for (int j = static_cast<int>(threadIdx.x); j < padded_dim; j += BlockSize) {
        int code = static_cast<int>(rescale * normalized_abs[j] + 1e-5);
        code = code < mask ? code : mask;
        local_ipnorm += (static_cast<double>(code) + 0.5) * normalized_abs[j];

        const float residual = vector[j] - centroid[j];
        if (residual < 0.0F) {
            code = (~code) & mask;
        }
        raw_ex_codes[row * static_cast<size_t>(padded_dim) + j] =
            static_cast<uint8_t>(code);
    }
    const double ipnorm = block_sum<BlockSize>(local_ipnorm);

    float ex_ip_residual_xu = 0.0F;
    float ex_ip_centroid_xu = 0.0F;
    for (int j = static_cast<int>(threadIdx.x); j < padded_dim; j += BlockSize) {
        const float residual = vector[j] - centroid[j];
        int total_code =
            static_cast<int>(raw_ex_codes[row * static_cast<size_t>(padded_dim) + j]);
        total_code += static_cast<int>(residual >= 0.0F) << ex_bits;
        const float xu = static_cast<float>(total_code) -
                         (static_cast<float>(1 << ex_bits) - 0.5F);
        ex_ip_residual_xu += residual * xu;
        ex_ip_centroid_xu += centroid[j] * xu;
    }
    ex_ip_residual_xu = block_sum<BlockSize>(ex_ip_residual_xu);
    ex_ip_centroid_xu = block_sum<BlockSize>(ex_ip_centroid_xu);

    if (threadIdx.x == 0) {
        float denominator = ex_ip_residual_xu;
        if (denominator == 0.0F) {
            denominator = INFINITY;
        }
        float ipnorm_inverse = static_cast<float>(1.0 / ipnorm);
        if (!isfinite(ipnorm_inverse) || ipnorm_inverse == 0.0F) {
            ipnorm_inverse = 1.0F;
        }

        float f_add = 0.0F;
        float f_rescale = 0.0F;
        if (metric_type == static_cast<int>(rabitqlib::METRIC_L2)) {
            f_add = l2_sqr + 2.0F * l2_sqr * ex_ip_centroid_xu / denominator;
            f_rescale = ipnorm_inverse * -2.0F * l2_norm;
        } else {
            f_add = 1.0F - residual_centroid_ip +
                    l2_sqr * ex_ip_centroid_xu / denominator;
            f_rescale = ipnorm_inverse * -l2_norm;
        }
        ex_factors[row * 2 + 0] = f_add;
        ex_factors[row * 2 + 1] = f_rescale;
    }
}

void rotate_rows(
    cublasHandle_t handle,
    const float* rotation,
    const float* input,
    float* output,
    size_t rows,
    size_t dim,
    size_t padded_dim
) {
    if (rows == 0) {
        return;
    }
    const float alpha = 1.0F;
    const float beta = 0.0F;
    cublas_check(
        cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            checked_cublas_int(padded_dim, "padded dimension"),
            checked_cublas_int(rows, "row count"),
            checked_cublas_int(dim, "dimension"),
            &alpha,
            rotation,
            checked_cublas_int(padded_dim, "rotation leading dimension"),
            input,
            checked_cublas_int(dim, "input leading dimension"),
            &beta,
            output,
            checked_cublas_int(padded_dim, "output leading dimension")
        ),
        "cublasSgemm(rotation)"
    );
}

struct BatchSegment {
    size_t cluster = 0;
    size_t local_start = 0;
    size_t grouped_start = 0;
    size_t batch_start = 0;
    size_t count = 0;
};

}  // namespace

namespace rabitqlib::ivf {

void IVF::construct_gpu(
    const float* data,
    const float* centroids,
    const PID* cluster_ids,
    bool faster
) {
    const auto construction_begin = TimingClock::now();
    std::cout << "Start GPU IVF construction...\n";
    if (type_ != RotatorType::MatrixRotator) {
        throw std::runtime_error(
            "GPU indexing currently supports MatrixRotator only"
        );
    }
    if (data == nullptr || centroids == nullptr || cluster_ids == nullptr) {
        throw std::invalid_argument("GPU indexing received a null input pointer");
    }

    const auto cuda_setup_begin = TimingClock::now();
    int device = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    if (properties.major < 7) {
        throw std::runtime_error(
            "GPU indexing requires compute capability 7.0 or newer"
        );
    }
    std::cout << "\tCUDA device: " << properties.name << " (sm_" << properties.major
              << properties.minor << ")\n";

    const size_t batch_capacity = gpu_batch_size();
    if (batch_capacity > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("RABITQ_GPU_BATCH_SIZE exceeds CUDA launch range");
    }
    std::cout << "\tGPU batch size: " << batch_capacity << " vectors\n";

    CudaStream stream;
    CublasHandle cublas(stream);
    const double cuda_setup_seconds = elapsed_seconds(cuda_setup_begin);

    const auto grouping_begin = TimingClock::now();
    std::cout << "\tGrouping vectors by cluster on GPU...\n";
    std::vector<size_t> counts(num_cluster_, 0);
    {
        DeviceBuffer<PID> sorted_cluster_ids(num_);
        DeviceBuffer<PID> sorted_ids(num_);
        DeviceBuffer<unsigned long long> device_counts(num_cluster_);
        DeviceBuffer<int> invalid(1);

        cuda_check(
            cudaMemsetAsync(
                device_counts.data(), 0, num_cluster_ * sizeof(unsigned long long), stream
            ),
            "cudaMemsetAsync(cluster counts)"
        );
        cuda_check(cudaMemsetAsync(invalid.data(), 0, sizeof(int), stream),
                   "cudaMemsetAsync(invalid cluster flag)");
        cuda_check(
            cudaMemcpyAsync(
                sorted_cluster_ids.data(),
                cluster_ids,
                num_ * sizeof(PID),
                cudaMemcpyHostToDevice,
                stream
            ),
            "copy cluster ids to GPU"
        );

        if (num_ > 0) {
            constexpr int block_size = 256;
            const size_t wanted_blocks = (num_ + block_size - 1) / block_size;
            const int grid_size = static_cast<int>(std::min<size_t>(wanted_blocks, 65535));
            histogram_cluster_ids_kernel<<<grid_size, block_size, 0, stream>>>(
                sorted_cluster_ids.data(),
                num_,
                num_cluster_,
                device_counts.data(),
                invalid.data()
            );
            cuda_check(cudaGetLastError(), "histogram_cluster_ids_kernel launch");
        }

        std::vector<unsigned long long> wide_counts(num_cluster_);
        int invalid_host = 0;
        cuda_check(
            cudaMemcpyAsync(
                wide_counts.data(),
                device_counts.data(),
                num_cluster_ * sizeof(unsigned long long),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "copy cluster counts to host"
        );
        cuda_check(
            cudaMemcpyAsync(
                &invalid_host,
                invalid.data(),
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "copy invalid cluster flag to host"
        );
        cuda_check(cudaStreamSynchronize(stream), "synchronize cluster histogram");
        if (invalid_host != 0) {
            throw std::runtime_error("cluster-id file contains an out-of-range id");
        }
        for (size_t cluster = 0; cluster < num_cluster_; ++cluster) {
            if (wide_counts[cluster] > std::numeric_limits<size_t>::max()) {
                throw std::overflow_error("cluster size exceeds host size_t range");
            }
            counts[cluster] = static_cast<size_t>(wide_counts[cluster]);
        }

        auto cluster_begin = thrust::device_pointer_cast(sorted_cluster_ids.data());
        auto id_begin = thrust::device_pointer_cast(sorted_ids.data());
        thrust::sequence(thrust::cuda::par.on(stream), id_begin, id_begin + num_, PID{0});
        thrust::stable_sort_by_key(
            thrust::cuda::par.on(stream),
            cluster_begin,
            cluster_begin + num_,
            id_begin
        );
        cuda_check(cudaStreamSynchronize(stream), "sort vectors by cluster");

        allocate_memory(counts);
        init_clusters(counts);
        if (num_ > 0) {
            cuda_check(
                cudaMemcpy(
                    ids_,
                    sorted_ids.data(),
                    num_ * sizeof(PID),
                    cudaMemcpyDeviceToHost
                ),
                "copy sorted vector ids to host"
            );
        }
    }
    const double grouping_seconds = elapsed_seconds(grouping_begin);

    std::vector<size_t> cluster_offsets(num_cluster_ + 1, 0);
    for (size_t cluster = 0; cluster < num_cluster_; ++cluster) {
        cluster_offsets[cluster + 1] = cluster_offsets[cluster] + counts[cluster];
    }
    if (cluster_offsets.back() != num_) {
        throw std::runtime_error("GPU cluster histogram does not sum to the data size");
    }

    const auto centroid_begin = TimingClock::now();
    std::cout << "\tUploading rotation and rotating centroids on GPU...\n";
    const size_t rotation_elements = checked_product(dim_, padded_dim_, "rotation matrix");
    if (rotator_->dump_bytes() != rotation_elements * sizeof(float)) {
        throw std::runtime_error("unexpected MatrixRotator serialization size");
    }
    std::vector<float> host_rotation(rotation_elements);
    rotator_->save(reinterpret_cast<char*>(host_rotation.data()));

    DeviceBuffer<float> device_rotation(rotation_elements);
    DeviceBuffer<float> device_rotated_centroids(
        checked_product(num_cluster_, padded_dim_, "rotated centroids")
    );
    cuda_check(
        cudaMemcpyAsync(
            device_rotation.data(),
            host_rotation.data(),
            rotation_elements * sizeof(float),
            cudaMemcpyHostToDevice,
            stream
        ),
        "copy rotation matrix to GPU"
    );
    {
        DeviceBuffer<float> device_centroids(
            checked_product(num_cluster_, dim_, "centroids")
        );
        cuda_check(
            cudaMemcpyAsync(
                device_centroids.data(),
                centroids,
                num_cluster_ * dim_ * sizeof(float),
                cudaMemcpyHostToDevice,
                stream
            ),
            "copy centroids to GPU"
        );
        rotate_rows(
            cublas,
            device_rotation.data(),
            device_centroids.data(),
            device_rotated_centroids.data(),
            num_cluster_,
            dim_,
            padded_dim_
        );
        cuda_check(cudaStreamSynchronize(stream), "rotate centroids");
    }
    std::vector<float> rotated_centroids(
        checked_product(num_cluster_, padded_dim_, "host rotated centroids")
    );
    cuda_check(
        cudaMemcpy(
            rotated_centroids.data(),
            device_rotated_centroids.data(),
            rotated_centroids.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "copy rotated centroids to host"
    );
    const double centroid_seconds = elapsed_seconds(centroid_begin);

    const auto batch_setup_begin = TimingClock::now();
    quant::RabitqConfig config;
    if (faster) {
        config = quant::faster_config(padded_dim_, ex_bits_ + 1);
    }

    const size_t input_elements = checked_product(batch_capacity, dim_, "input batch");
    const size_t rotated_elements =
        checked_product(batch_capacity, padded_dim_, "rotated batch");
    const size_t binary_bytes_per_vector = padded_dim_ / 8;
    const size_t binary_elements =
        checked_product(batch_capacity, binary_bytes_per_vector, "binary codes");
    const size_t raw_ex_elements = ex_bits_ > 0
                                       ? checked_product(
                                             batch_capacity, padded_dim_, "raw ex codes"
                                         )
                                       : 0;

    PinnedBuffer<float> host_input(input_elements);
    PinnedBuffer<PID> host_cluster_ids(batch_capacity);
    PinnedBuffer<uint8_t> host_binary(binary_elements);
    PinnedBuffer<float> host_short_factors(batch_capacity * 3);
    PinnedBuffer<uint8_t> host_raw_ex(raw_ex_elements);
    PinnedBuffer<float> host_ex_factors(ex_bits_ > 0 ? batch_capacity * 2 : 0);

    DeviceBuffer<float> device_input(input_elements);
    DeviceBuffer<float> device_rotated(rotated_elements);
    DeviceBuffer<PID> device_batch_cluster_ids(batch_capacity);
    DeviceBuffer<uint8_t> device_binary(binary_elements);
    DeviceBuffer<float> device_short_factors(batch_capacity * 3);
    DeviceBuffer<uint8_t> device_raw_ex(raw_ex_elements);
    DeviceBuffer<float> device_ex_factors(ex_bits_ > 0 ? batch_capacity * 2 : 0);

    const size_t shared_bytes =
        (padded_dim_ + 3 * static_cast<size_t>(kQuantizeBlockSize)) * sizeof(float);
    if (ex_bits_ > 0 && shared_bytes > properties.sharedMemPerBlock) {
        throw std::runtime_error(
            "padded dimension requires more shared memory than this GPU provides"
        );
    }
    const double batch_setup_seconds = elapsed_seconds(batch_setup_begin);

    std::cout << "\tRotating and quantizing vectors on GPU...\n";
    size_t current_cluster = 0;
    size_t current_local = 0;
    size_t processed = 0;
    double staging_seconds = 0.0;
    double gpu_work_seconds = 0.0;
    double layout_packing_seconds = 0.0;
    while (current_cluster < num_cluster_) {
        const auto staging_begin = TimingClock::now();
        std::vector<BatchSegment> segments;
        size_t rows = 0;
        while (rows < batch_capacity && current_cluster < num_cluster_) {
            if (current_local == counts[current_cluster]) {
                ++current_cluster;
                current_local = 0;
                continue;
            }

            const size_t remaining = counts[current_cluster] - current_local;
            const size_t space = batch_capacity - rows;
            size_t take = std::min(remaining, space);
            if (take < remaining) {
                take -= take % fastscan::kBatchSize;
            }
            if (take == 0) {
                break;
            }
            segments.push_back(BatchSegment{
                current_cluster,
                current_local,
                cluster_offsets[current_cluster] + current_local,
                rows,
                take,
            });
            rows += take;
            current_local += take;
        }
        if (rows == 0) {
            throw std::runtime_error("failed to form a non-empty GPU indexing batch");
        }

        for (const BatchSegment& segment : segments) {
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < segment.count; ++i) {
                const PID id = ids_[segment.grouped_start + i];
                std::memcpy(
                    host_input.data() + (segment.batch_start + i) * dim_,
                    data + static_cast<size_t>(id) * dim_,
                    dim_ * sizeof(float)
                );
                host_cluster_ids.data()[segment.batch_start + i] =
                    static_cast<PID>(segment.cluster);
            }
        }
        staging_seconds += elapsed_seconds(staging_begin);

        const auto gpu_work_begin = TimingClock::now();
        cuda_check(
            cudaMemcpyAsync(
                device_input.data(),
                host_input.data(),
                rows * dim_ * sizeof(float),
                cudaMemcpyHostToDevice,
                stream
            ),
            "copy input batch to GPU"
        );
        cuda_check(
            cudaMemcpyAsync(
                device_batch_cluster_ids.data(),
                host_cluster_ids.data(),
                rows * sizeof(PID),
                cudaMemcpyHostToDevice,
                stream
            ),
            "copy batch cluster ids to GPU"
        );
        rotate_rows(
            cublas,
            device_rotation.data(),
            device_input.data(),
            device_rotated.data(),
            rows,
            dim_,
            padded_dim_
        );

        quantize_rows_kernel<kQuantizeBlockSize>
            <<<static_cast<unsigned int>(rows),
               kQuantizeBlockSize,
               ex_bits_ > 0 ? shared_bytes : 0,
               stream>>>(
                device_rotated.data(),
                device_rotated_centroids.data(),
                device_batch_cluster_ids.data(),
                rows,
                checked_cublas_int(padded_dim_, "padded dimension"),
                static_cast<int>(ex_bits_),
                static_cast<int>(metric_type_),
                faster,
                config.t_const,
                device_binary.data(),
                device_short_factors.data(),
                device_raw_ex.data(),
                device_ex_factors.data()
            );
        cuda_check(cudaGetLastError(), "quantize_rows_kernel launch");

        cuda_check(
            cudaMemcpyAsync(
                host_binary.data(),
                device_binary.data(),
                rows * binary_bytes_per_vector * sizeof(uint8_t),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "copy binary codes to host"
        );
        cuda_check(
            cudaMemcpyAsync(
                host_short_factors.data(),
                device_short_factors.data(),
                rows * 3 * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream
            ),
            "copy short-code factors to host"
        );
        if (ex_bits_ > 0) {
            cuda_check(
                cudaMemcpyAsync(
                    host_raw_ex.data(),
                    device_raw_ex.data(),
                    rows * padded_dim_ * sizeof(uint8_t),
                    cudaMemcpyDeviceToHost,
                    stream
                ),
                "copy ex codes to host"
            );
            cuda_check(
                cudaMemcpyAsync(
                    host_ex_factors.data(),
                    device_ex_factors.data(),
                    rows * 2 * sizeof(float),
                    cudaMemcpyDeviceToHost,
                    stream
                ),
                "copy ex-code factors to host"
            );
        }
        cuda_check(cudaStreamSynchronize(stream), "complete GPU quantization batch");
        gpu_work_seconds += elapsed_seconds(gpu_work_begin);

        const auto layout_packing_begin = TimingClock::now();
        for (const BatchSegment& segment : segments) {
            const size_t num_blocks =
                (segment.count + fastscan::kBatchSize - 1) / fastscan::kBatchSize;
#pragma omp parallel for schedule(static)
            for (size_t block = 0; block < num_blocks; ++block) {
                const size_t segment_offset = block * fastscan::kBatchSize;
                const size_t block_count = std::min(
                    fastscan::kBatchSize, segment.count - segment_offset
                );
                char* output =
                    cluster_lst_[segment.cluster].batch_data() +
                    ((segment.local_start + segment_offset) / fastscan::kBatchSize) *
                        BatchDataMap<float>::data_bytes(padded_dim_);
                BatchDataMap<float> output_map(output, padded_dim_);
                const size_t batch_row = segment.batch_start + segment_offset;
                fastscan::pack_codes(
                    padded_dim_,
                    host_binary.data() + batch_row * binary_bytes_per_vector,
                    block_count,
                    output_map.bin_code()
                );
                for (size_t i = 0; i < block_count; ++i) {
                    const float* factors =
                        host_short_factors.data() + (batch_row + i) * 3;
                    output_map.f_add()[i] = factors[0];
                    output_map.f_rescale()[i] = factors[1];
                    output_map.f_error()[i] = factors[2];
                }
            }

            if (ex_bits_ > 0) {
                const size_t ex_stride =
                    ExDataMap<float>::data_bytes(padded_dim_, ex_bits_);
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < segment.count; ++i) {
                    char* output =
                        cluster_lst_[segment.cluster].ex_data() +
                        (segment.local_start + i) * ex_stride;
                    ExDataMap<float> output_map(output, padded_dim_, ex_bits_);
                    const size_t batch_row = segment.batch_start + i;
                    quant::rabitq_impl::ex_bits::packing_rabitqplus_code(
                        host_raw_ex.data() + batch_row * padded_dim_,
                        output_map.ex_code(),
                        padded_dim_,
                        ex_bits_
                    );
                    const float* factors =
                        host_ex_factors.data() + batch_row * 2;
                    output_map.f_add_ex() = factors[0];
                    output_map.f_rescale_ex() = factors[1];
                }
            }
        }
        layout_packing_seconds += elapsed_seconds(layout_packing_begin);

        processed += rows;
        if (processed == num_ || processed % (batch_capacity * 32) == 0) {
            std::cout << "\t  processed " << processed << " / " << num_
                      << " vectors\n";
        }
    }

    const auto initializer_begin = TimingClock::now();
    initer_->add_vectors(rotated_centroids.data());
    const double initializer_seconds = elapsed_seconds(initializer_begin);
    const double construction_seconds = elapsed_seconds(construction_begin);

    std::cout << "GPU IVF construction complete\n"
              << "\tGPU construction timing (seconds)\n"
              << "\t  CUDA context/stream/cuBLAS setup: " << cuda_setup_seconds << '\n'
              << "\t  cluster grouping/sorting/allocation: " << grouping_seconds << '\n'
              << "\t  centroid upload/rotation/download: " << centroid_seconds << '\n'
              << "\t  quantizer config/batch-buffer allocation: "
              << batch_setup_seconds << '\n'
              << "\t  batch planning and host staging: " << staging_seconds << '\n'
              << "\t  GPU transfers/rotation/quantization: " << gpu_work_seconds << '\n'
              << "\t  CPU legacy-layout packing: " << layout_packing_seconds << '\n'
              << "\t  centroid search initializer: " << initializer_seconds << '\n'
              << "\t  construct_gpu total: " << construction_seconds << '\n';
}

}  // namespace rabitqlib::ivf
