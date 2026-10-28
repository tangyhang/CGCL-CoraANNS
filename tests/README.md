# Checks

The CMake suite includes standalone C++ checks and Python integration tests.

- `experiment_options_test`: option validation.
- `recall_evaluator_test`: ID/distance-aware recall, tie, invalid-ID,
  metric, and file-format cases.
- `cpu_pipeline`: independent synthetic vectors with known nearest IDs; CPU
  index construction, serialization, loading, search, and recall evaluation.
- `experiment_runner`: eight subprocess integration cases covering successful
  results, failed jobs, incomplete tables, recall-mode mismatches, missing
  inputs, paths with spaces, dry runs, and existing-result preservation.

Run `ctest --test-dir build-cpu --output-on-failure` after a CPU-only build.
The GitHub Actions workflow exercises this path.

With `CORAANNS_BUILD_GPU_TESTS=ON`, three additional CTest entries link the
production libraries: `gpu_cpu_highk_vanguard_test`,
`gpu_index_residency_test`, and `gpu_highk_validation`. Run them on a working
GPU with `ctest --test-dir build -L gpu --output-on-failure`.

`test_generate_gt_gpu.py` is a separate set of ten NumPy/PyTorch tests with
an independent FP64 exhaustive-search oracle. The default device is CPU;
set `GT_TEST_DEVICE=cuda:0` to exercise CUDA. See `docs/BUILD.md` for commands
and the hybrid engine's pinned-memory requirements.
