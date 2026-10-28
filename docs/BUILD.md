# Building and validation

## Dependencies

The code targets Linux and x86-64 CPUs with AVX2, FMA, and BMI2. RaBitQ kernels
use the build host's instruction set by default. Set
`-DCORAANNS_NATIVE_ARCH=OFF` when building for another compatible CPU; the
explicit AVX2/FMA/BMI2 requirement still applies.

The hybrid engine uses a fixed candidate queue with one billion 16-byte
items, allocated with `cudaHostAllocMapped`: 16 GB (about 14.9 GiB) of pinned
host memory before index/query workspace. This applies to small GPU examples
and the vanguard GPU test as well. GPU filtering also requires device memory
for the one-bit index and workspace.

| Dependency | Use | Reference environment |
| --- | --- | --- |
| CMake | Build configuration | Minimum 3.24; used 3.31.10 |
| GCC/G++ and OpenMP | CPU code | GCC 11.4 for CPU-only; G++ 13.4 for CUDA build C++ sources |
| CUDA toolkit, cuBLAS | GPU search and indexing | CUDA 12.6.85; cuBLAS 12.6.4.1 |
| RAFT headers, RMM, CCCL | GPU selection primitives | RAFT/RMM 26.06; CCCL 3.4.0 |
| Python 3 | Experiment driver and smoke test | Standard library suffices |
| NumPy and Faiss | Demo generation and IVF training | NumPy 1.26.4, Faiss 1.8.0; optional |
| NumPy and PyTorch | Exhaustive ground-truth tool | PyTorch 2.7.0+cu126; optional |

Activate an environment containing the CUDA toolkit,
`libraft-headers`, and their dependencies. CMake automatically searches the
active `CONDA_PREFIX`; otherwise pass `-DCMAKE_PREFIX_PATH=/path/to/prefix`.
RAFT's CMake target supplies RMM and CCCL. Python Faiss is used for IVF training.

## Full system

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build -j2
ctest --test-dir build -L cpu --output-on-failure
```

If compiler detection selects another toolkit or compiler, set
`CMAKE_CUDA_COMPILER`, `CMAKE_CXX_COMPILER`, and, if needed,
`CMAKE_CUDA_HOST_COMPILER` explicitly when creating a fresh build directory.
An activated Conda environment may provide its own C++ compiler and `nvcc`.
Use a CUDA-supported host compiler. The CUDA default is SM 70 to match the
V100 configuration.

Use the environment's C++ runtime as well as its compiler. If a partially
activated Conda setup fails to link RMM with `CXXABI_1.3.15` missing, first
activate the complete environment. For an explicit prefix-based configuration,
set `-DCMAKE_EXE_LINKER_FLAGS="-L${CONDA_PREFIX}/lib -Wl,-rpath,${CONDA_PREFIX}/lib"`.
This selects the environment's C++ runtime.

Release host code uses `-Ofast`, AVX2/FMA/BMI2, OpenMP, and native CPU selection.
CUDA Release settings come from CMake, with relaxed constexpr, extended lambda,
and line-info options. Compiler flags are configured per target.

CPU-only:

```bash
cmake -S . -B build-cpu -DCORAANNS_ENABLE_CUDA=OFF
cmake --build build-cpu -j2
ctest --test-dir build-cpu --output-on-failure
```

## Targets and options

| Target | Purpose |
| --- | --- |
| `coraanns_build_index` | CPU index construction and serialization |
| `coraanns_build_index_gpu` | GPU index construction |
| `coraanns_search` | Main system; vanguard capacity 512 |
| `coraanns_search_highk` | Capacity 1024, built explicitly |
| `rabitq_cpu` | Parallel CPU IVF-RaBitQ baseline |
| `rabitq_gpu` | GPU baseline with HBM/host-excode/cache residency options |
| `rabitq_gpu_highk` | GPU baseline with fine capacity 1024, built explicitly |

`CORAANNS_BUILD_BASELINES` defaults to ON. Disable it to build only the system
and index builders. `BUILD_TESTING` defaults to ON.

For large top-k:

```bash
cmake --build build --target coraanns_search_highk rabitq_gpu_highk -j2
```

`-DCORAANNS_BUILD_EXPERIMENTAL=ON` additionally builds `coraanns_cpu_gpu`,
`coraanns_gpu_global`, `coraanns_asymmetric`, and `coraanns_backlog_trace`.
Run these variants through their individual executables. The
CPU-filter/GPU-refinement variant and GPU index builder link the CUDA toolkit
directly; the other search targets also link RAFT.

## Tests

The CPU tests exercise option validation, ID and distance-aware recall
(including duplicates, ties, invalid IDs, L2/IP, and binary formats), and a
synthetic index round trip. Release tests explicitly retain assertions.

For GPU tests:

```bash
cmake -S . -B build -DCORAANNS_BUILD_GPU_TESTS=ON
cmake --build build -j2
ctest --test-dir build -L gpu --output-on-failure
```

The GPU tests cover vanguard top-k capacity through 1024, result consistency
with and without vanguard and query chunking, GPU index residency modes, and
high-k results against a CPU oracle. They link the production kernels. With
GPU tests enabled, their high-k libraries are built as dependencies.

Optional ground-truth tool tests, using NumPy/PyTorch:

```bash
python3 -m unittest discover -s tests -p test_generate_gt_gpu.py -v
GT_TEST_DEVICE=cuda:0 python3 -m unittest discover -s tests -p test_generate_gt_gpu.py -v
```

See [validation results](VALIDATION.md).
