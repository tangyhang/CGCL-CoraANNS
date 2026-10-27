cmake_minimum_required(VERSION 3.24)
project(CoraANNS VERSION 0.1.0 LANGUAGES CXX)

option(CORAANNS_ENABLE_CUDA "Build CoraANNS and the CUDA index builder" ON)
option(CORAANNS_BUILD_BASELINES "Build CPU and GPU RaBitQ baselines" ON)
option(CORAANNS_BUILD_EXPERIMENTAL "Build mechanism and tracing variants" OFF)
option(CORAANNS_BUILD_GPU_TESTS "Build and register tests that require a GPU" OFF)
option(CORAANNS_NATIVE_ARCH "Use the build host's CPU instruction set" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

find_package(OpenMP REQUIRED COMPONENTS CXX)
add_library(coraanns_host INTERFACE)
target_include_directories(coraanns_host INTERFACE "${PROJECT_SOURCE_DIR}/include")
target_include_directories(coraanns_host SYSTEM INTERFACE
    "${PROJECT_SOURCE_DIR}/third_party/rabitq/include")
target_link_libraries(coraanns_host INTERFACE OpenMP::OpenMP_CXX)
# Retain the original numerical and CPU ISA settings. Keep flags target-local.
target_compile_options(coraanns_host INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-mavx2;-mfma;-mbmi2;-fPIC;-fexceptions;-ftree-vectorize>
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CONFIG:Release>>:-Ofast>)
if(CORAANNS_NATIVE_ARCH)
    target_compile_options(coraanns_host INTERFACE
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>)
endif()

add_executable(coraanns_build_index apps/build_index.cpp)
target_link_libraries(coraanns_build_index PRIVATE coraanns_host)
if(CORAANNS_BUILD_BASELINES)
    add_executable(rabitq_cpu apps/baselines/cpu.cpp)
    target_link_libraries(rabitq_cpu PRIVATE coraanns_host)
endif()

if(CORAANNS_ENABLE_CUDA)
    # Set before enabling CUDA so CMake cannot silently choose a different GPU.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING "CUDA architectures (70 = V100)")
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    find_package(CUDAToolkit REQUIRED)
    if(DEFINED ENV{CONDA_PREFIX})
        list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    endif()
    find_package(raft CONFIG REQUIRED)

    add_library(coraanns_cuda_base INTERFACE)
    target_link_libraries(coraanns_cuda_base INTERFACE
        coraanns_host CUDA::cudart CUDA::cublas)
    target_compile_options(coraanns_cuda_base INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo;--expt-extended-lambda;--expt-relaxed-constexpr;-Xcompiler=-fopenmp,-fno-stack-protector,-mavx2,-mfma,-mbmi2>)
    if(CORAANNS_NATIVE_ARCH)
        target_compile_options(coraanns_cuda_base INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>)
    endif()
    add_library(coraanns_cuda INTERFACE)
    target_link_libraries(coraanns_cuda INTERFACE coraanns_cuda_base raft::raft)

    add_library(coraanns_index_gpu STATIC src/index/build_gpu.cu)
    target_link_libraries(coraanns_index_gpu PUBLIC coraanns_cuda_base)
    add_executable(coraanns_build_index_gpu apps/build_index_gpu.cpp)
    target_link_libraries(coraanns_build_index_gpu PRIVATE coraanns_index_gpu)

    function(coraanns_add_search library command)
        add_library(${library} STATIC ${ARGN}
            src/search/hybrid.cu src/search/cpu_refinement.cpp)
        target_link_libraries(${library} PUBLIC coraanns_cuda)
        add_executable(${command} ${ARGN} apps/search.cpp)
        target_link_libraries(${command} PRIVATE ${library})
    endfunction()
    coraanns_add_search(coraanns coraanns_search)
    # Capacity-only variant; leave the original 512-entry build unchanged.
    coraanns_add_search(coraanns_highk coraanns_search_highk EXCLUDE_FROM_ALL)
    target_compile_definitions(coraanns_highk PUBLIC RABITQ_VANGUARD_CAPACITY=1024)

    if(CORAANNS_BUILD_BASELINES)
        add_library(rabitq_gpu_kernels STATIC src/baselines/gpu.cu)
        target_link_libraries(rabitq_gpu_kernels PUBLIC coraanns_cuda)
        add_executable(rabitq_gpu apps/baselines/gpu.cpp)
        target_link_libraries(rabitq_gpu PRIVATE rabitq_gpu_kernels)
        add_library(rabitq_gpu_highk_kernels STATIC EXCLUDE_FROM_ALL src/baselines/gpu.cu)
        target_compile_definitions(rabitq_gpu_highk_kernels PUBLIC RABITQ_GPU_FINE_CAPACITY=1024)
        target_link_libraries(rabitq_gpu_highk_kernels PUBLIC coraanns_cuda)
        add_executable(rabitq_gpu_highk EXCLUDE_FROM_ALL apps/baselines/gpu.cpp)
        target_link_libraries(rabitq_gpu_highk PRIVATE rabitq_gpu_highk_kernels)
    endif()

    if(CORAANNS_BUILD_EXPERIMENTAL)
        foreach(variant cpu_gpu gpu_global asymmetric backlog_trace)
            add_executable(coraanns_${variant}
                apps/experimental/${variant}.cpp src/experimental/${variant}.cu)
            if(variant STREQUAL "cpu_gpu")
                target_link_libraries(coraanns_${variant} PRIVATE coraanns_cuda_base)
            else()
                target_link_libraries(coraanns_${variant} PRIVATE coraanns_cuda)
            endif()
        endforeach()
    endif()
elseif(CORAANNS_BUILD_GPU_TESTS OR CORAANNS_BUILD_EXPERIMENTAL)
    message(FATAL_ERROR "GPU tests and experimental variants require CORAANNS_ENABLE_CUDA=ON")
endif()

include(CTest)
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
