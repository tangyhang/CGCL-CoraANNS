# Validation results

Validation date: 2026-10-27.

## Passed

- CPU-only Release build with GCC 11.4 and CMake 3.31.10.
- CPU application build with `CORAANNS_NATIVE_ARCH=OFF`, retaining the explicit
  AVX2/FMA/BMI2 requirements without selecting the local CPU's extra features.
- Full SM 70 CUDA build with CUDA 12.6.85, RAFT/RMM 26.06, and the environment's
  C++ runtime. G++ 13.4 compiled C++ sources; nvcc selected GCC 11.4 as its host
  compiler. Explicit prefix library search/rpath was used to select the Conda
  runtime required by RMM.
- All 11 application executables link, load, and display usage, including
  the four optional experimental variants and both high-k applications.
- All three GPU correctness executables compile and are registered with CTest.
- Four CPU CTest entries pass in both the CPU-only and CUDA build directories:
  options, recall evaluation, experiment-runner integration, and a synthetic
  index-build/serialization/load/search round trip.
- The runner integration entry contains eight subprocess cases. These exercise
  output preservation, failure metadata, exact probe tables, recall labels,
  missing inputs, paths containing spaces, and dry-run behavior.
- Ten ground-truth-generator tests pass on CPU using an independent
  FP64 oracle.
- The documented demo completes data generation, Faiss IVF
  training, 8-bit CPU index construction, loading, and CPU search on 4,096
  32-dimensional vectors, 32 queries, and 16 clusters.
- Every supplied configuration expands successfully: 47 jobs across the three
  datasets. Ablation groups have matching nprobe grids and top-1000 selects
  the high-k executable. Python tools compile and local documentation links
  resolve.

## Runtime validation status

| Check | Status |
| --- | --- |
| CPU search and recall | Passed |
| GPU correctness tests | Pending |
| GPU index construction and search | Pending |
| Large-dataset throughput and recall | Pending |

## Local logs

- `build/compile-verified.log`
- `build/cpu-tests.log`
- `build-cpu/groundtruth-tests.log`
- `build-cpu/demo-*.log`
