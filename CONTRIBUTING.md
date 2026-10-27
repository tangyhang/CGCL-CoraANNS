# Contributing

Keep changes to organization separate from changes to search behavior. In
particular, changes to thresholds, queues, numerical precision, timing scopes,
and kernel selection need independent correctness and performance evidence.

Use C++17, the repository's `.clang-format`, and target-local CMake settings.
Keep command-line handling in `apps/`, computational implementations in `src/`,
and shared interfaces in `include/coraanns/`. Avoid reformatting `third_party/`.

Before submitting a change, build the affected targets and run `ctest -L cpu
--output-on-failure` in the build directory. For CUDA changes, also build the
GPU tests and run `ctest -L gpu --output-on-failure` on a working GPU. Report the
compiler, CUDA/RAFT versions, GPU, and any tests you could not run. Large
benchmark runs should retain the configuration, raw logs, and actual recall
mode produced by `tools/run_experiment.py`.

Use small, focused pull requests. Describe the concrete behavior and validation.
Do not commit datasets, compiled binaries, profiler dumps, or private paths.
