# CoraANNS

CoraANNS performs approximate nearest neighbor search with GPU filtering and
CPU refinement over an IVF-RaBitQ index. It combines vanguard-cluster scheduling,
shared thresholds, CPU secondary filtering, and cross-query reuse.

[中文说明](README.zh-CN.md) · [Build](docs/BUILD.md) ·
[Reproduce experiments](docs/REPRODUCIBILITY.md) ·
[Implementation map](docs/ARCHITECTURE.md) · [Source map](docs/MIGRATION.md)

## Layout

```text
apps/                    Command-line index builders and search drivers
include/coraanns/        Search interfaces and shared benchmark utilities
src/search/              Production GPU search and CPU refinement
src/index/               GPU index construction
src/baselines/           GPU RaBitQ baseline
src/experimental/        Optional mechanism and backlog-tracing variants
configs/                 Portable LAION100M, Deep1B, and Wiki88M configurations
tools/data/              Data generation, IVF training, and ground-truth tools
tools/run_experiment.py  Experiment commands, raw logs, metadata, and CSV output
tests/                   CPU checks and optional GPU correctness checks
third_party/rabitq/      Vendored RaBitQ dependency with original notices
docs/                    Build, data, evaluation, provenance, and release notes
```

## Build

The full system requires Linux on an AVX2/FMA/BMI2-capable x86-64 CPU, OpenMP,
CUDA/cuBLAS, and RAFT with its RMM/CCCL dependencies. The default CUDA target is
SM 70 (V100); set `CMAKE_CUDA_ARCHITECTURES` for another GPU. See
[the build guide](docs/BUILD.md) for dependency versions
and optional targets.

With a CUDA/RAFT environment activated:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build -j2
ctest --test-dir build -L cpu --output-on-failure
```

For a CPU-only build:

```bash
cmake -S . -B build-cpu -DCORAANNS_ENABLE_CUDA=OFF
cmake --build build-cpu -j2
ctest --test-dir build-cpu --output-on-failure
```

Executables are written to the selected build directory's `bin/`. The default
build includes `coraanns_build_index`, `coraanns_build_index_gpu`,
`coraanns_search`, `rabitq_cpu`, and `rabitq_gpu`.

## Small end-to-end example

The demo requires NumPy and Faiss in addition to the C++ build. It generates
4,096 vectors, trains IVF with Faiss, builds an 8-bit RaBitQ index, and searches it.

The hybrid engine allocates a fixed mapped candidate queue of 16 GB
(about 14.9 GiB) of pinned host memory. The CPU example below uses the CPU
RaBitQ search path.

```bash
python3 tools/data/make_demo.py --output data/demo
python3 tools/data/train_ivf.py \
  --base data/demo/base.fvecs --nlist 16 \
  --centroids data/demo/centroids.fvecs --assignments data/demo/assignments.ivecs
build/bin/coraanns_build_index \
  data/demo/base.fvecs data/demo/centroids.fvecs data/demo/assignments.ivecs \
  8 data/demo/ivf.index l2 false
build/bin/coraanns_search \
  data/demo/ivf.index data/demo/queries.fvecs data/demo/groundtruth.ivecs \
  --topk 10 --nprobes 4,8,16 --threads 4 --batch-q 4 --rounds 5 --warmup 1 \
  --vanguard-size 5 --cpu-secondary-filter true \
  --recall-base data/demo/base.fvecs
```

For a CPU-only demonstration, use `build-cpu/bin/coraanns_build_index` above
and search with:

```bash
build-cpu/bin/rabitq_cpu \
  data/demo/ivf.index data/demo/queries.fvecs data/demo/groundtruth.ivecs \
  --topk 10 --nprobes 4,8,16 --threads 4 --rounds 5 --warmup 1 \
  --recall-base data/demo/base.fvecs
```

## Dataset experiments

Use the supplied configurations with your dataset directory. First inspect
the expanded commands:

```bash
python3 tools/run_experiment.py \
  --config configs/laion100m.json --data-root /data/laion100m \
  --experiment ablation --dry-run
```

Remove `--dry-run` to execute. The runner creates a new results directory with
raw logs, `commands.sh`, `metadata.json`, and `summary.csv`. Available sweeps
are `overall`, `ablation`, `vanguard`, `threads`, and LAION100M `topk`.
Use `--numa-interleave` to reproduce `numactl --interleave=all` explicitly.

The runner defaults to distance-aware recall and supplies the base vectors
through `--recall-base`. Use `--recall-mode id` for ID recall. The chosen mode
is recorded with the results. The `coraanns_search` option `--exact-base`
enables FP32 refinement for diagnostic runs.

Prepare datasets and indexes following
[the reproducibility guide](docs/REPRODUCIBILITY.md), which describes data
layouts, training commands, evaluation, and experiment parameters.

## License and attribution

This project uses the [Apache License 2.0](LICENSE). See the
[third-party notices](THIRD_PARTY_NOTICES.md) for dependency licenses and
attribution. RaBitQ's README is at `third_party/rabitq/UPSTREAM_README.md`. The
[source map](docs/MIGRATION.md) lists the corresponding source files.
