#pragma once
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdexcept>
#include <string>

namespace coraanns::detail {
inline void check_cuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
inline void check_cublas(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(status));
}
} // namespace coraanns::detail
