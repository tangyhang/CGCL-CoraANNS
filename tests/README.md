# Testing

The CMake suite includes standalone C++ checks and Python integration tests.
See [the build guide](../docs/BUILD.md) for dependencies and compiler options.

- `experiment_options_test`: option validation.
- `search_parameters_test`: batch, launch-grid and integer indexing limits.
- `public_interfaces_test` (CUDA build): all engine headers included together;
  owning engines cannot be copied.
- `recall_evaluator_test`: ID/distance-aware recall, tie, invalid-ID,
  metric, and file-format cases.
- `cpu_pipeline`: independent synthetic vectors with known nearest IDs; CPU
  index construction, serialization, loading, search, and recall evaluation.
- `experiment_runner`: eight subprocess integration cases covering successful
  results, failed jobs, incomplete tables, recall-mode mismatches, missing
  inputs, paths with spaces, dry runs, and existing-result preservation.
- Performance-log checks validate repeated rounds, matched configurations,
  and QPS/recall regression detection.

Run CPU checks with:

```bash
cmake -S . -B build-cpu -DCORAANNS_ENABLE_CUDA=OFF
cmake --build build-cpu -j2
ctest --test-dir build-cpu --output-on-failure
```

The GitHub Actions workflow exercises this path.

With `CORAANNS_BUILD_GPU_TESTS=ON`, six GPU CTest entries exercise production
code. `gpu_pipeline` builds both CPU and GPU indexes and checks hybrid search
with extended-code and exact FP32 refinement; `gpu_search_lifecycle_test` checks
workspace limits and repeated allocation/release. GPU checks also cover
vanguard, index residency and high top-k results. In the CUDA/RAFT environment,
run:

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=70 -DCORAANNS_BUILD_GPU_TESTS=ON
cmake --build build -j2
ctest --test-dir build -L gpu --output-on-failure
```

`test_generate_gt_gpu.py` is a separate set of ten NumPy/PyTorch tests with
an independent FP64 exhaustive-search oracle. The default device is CPU;
set `GT_TEST_DEVICE=cuda:0` to exercise CUDA. See `docs/BUILD.md` for commands
and the hybrid engine's pinned-memory requirements.
