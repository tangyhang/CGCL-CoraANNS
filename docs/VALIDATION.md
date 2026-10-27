# Migration validation

Validation date: 2026-10-27. This records local checks of the repository
organization, not a reproduction of the paper's large-dataset performance.

## Passed

- Imported-file audit: 413 destination files match their recorded SHA-256
  hashes and the source working-tree file hashes. All 27 imported first-party
  C++/CUDA implementation, driver, and shared-interface files match the source
  token sequences after the recorded include moves and CPU file split.
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
- Ten retained ground-truth-generator tests pass on CPU using an independent
  FP64 oracle.
- The documented demo completes data generation, original-recipe Faiss IVF
  training, 8-bit CPU index construction, loading, and CPU search on 4,096
  32-dimensional vectors, 32 queries, and 16 clusters. This validates the tool
  chain; its recall is not a paper benchmark result.
- Every supplied configuration expands successfully: 47 jobs across the three
  datasets. Ablation groups have matching nprobe grids and top-1000 selects
  the high-k executable. Python tools compile and local documentation links
  resolve.

## Not run

`nvidia-smi` cannot communicate with the NVIDIA driver in this environment.
GPU correctness tests, GPU index construction/search execution, and large-data
throughput/recall experiments were therefore not run. CUDA compilation and
command-line loading are not substitutes for those runtime checks.

The GitHub Actions workflow has been prepared but has not run on GitHub.
No remote repository, release, public artifact URL, dataset hosting, or paper
citation was created as part of this local task.

The code retains inherited unused-variable and CUDA qualifier warnings; no
computational code was removed to silence them. Main-engine queue size and
memory requirements are documented in `BUILD.md`.

Local build and demo outputs are ignored by `.gitignore`. Relevant logs are
`build/compile-verified.log`, `build/cpu-tests.log`,
`build-cpu/groundtruth-tests.log`, and `build-cpu/demo-*.log`; these are local
verification evidence and are not bundled as published experimental results.
