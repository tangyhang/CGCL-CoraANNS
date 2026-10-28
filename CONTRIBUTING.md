# Contributing

Include correctness and performance results for changes to thresholds, queues,
numerical precision, timing scopes, and kernel selection.

Use C++17, the repository's `.clang-format`, and target-local CMake settings.
Keep command-line handling in `apps/`, computational implementations in `src/`,
and shared interfaces in `include/coraanns/`. Follow the upstream formatting
for files under `third_party/`.

Before submitting a change, build the affected targets and run `ctest -L cpu
--output-on-failure` in the build directory. For CUDA changes, also build the
GPU tests and run `ctest -L gpu --output-on-failure` on a GPU. Include the
compiler, CUDA/RAFT versions, GPU, and test results. Large
benchmark runs should retain the configuration, raw logs, and actual recall
mode produced by `tools/run_experiment.py`.

Use small, focused pull requests. Describe the concrete behavior and validation.
Keep datasets and generated artifacts in `data/`, `build*/`, and `results/`.
