# Implementation map

CoraANNS combines GPU filtering and CPU refinement, vanguard-cluster scheduling,
shared thresholds, CPU secondary filtering, and cross-query reuse.

| Responsibility | Main location |
| --- | --- |
| Inputs, options, warmups, measurement and recall | `apps/search.cpp` |
| Packed IVF adaptation and extended-code ownership | `include/coraanns/index/search_data.hpp`, `src/index/search_data.cpp` |
| Search API and parameter limits | `include/coraanns/search/` |
| Workspace, streams and CPU/GPU orchestration | `src/search/hybrid.cu` |
| Rotation factors, norms, query quantization and LUT preparation | `src/search/query_kernels.cuh` |
| Vanguard, recovery, bitwise and cross-query filtering | `src/search/filter_kernels.cuh` |
| CPU secondary filtering, refinement and per-query top-k | `src/search/cpu_refinement.cpp` |
| CPU/GPU index construction | `apps/build_index*.cpp`, `src/index/build_gpu.cu` |
| IVF serialization, quantization, FastScan and rotations | `third_party/rabitq/include/rabitqlib/` |
| Vector file mapping | `include/coraanns/io/mapped_vectors.hpp` |
| Benchmark options and recall evaluation | `include/coraanns/benchmark/` |
| Parameter sweeps and result collection | `configs/`, `tools/run_experiment.py` |

## Search flow

1. The driver reads queries, ground truth and the IVF index. `SearchData` adapts
   packed codes, factors, centroids and IDs to the GPU/CPU layout before timing.
2. `coraanns::HybridIndex` prepares queries and selects probe lists. Pair
   scheduling prioritizes vanguard clusters to establish useful thresholds early.
3. GPU kernels filter candidates and reuse query-independent work across queries.
4. CPU workers apply secondary filtering, refine extended RaBitQ codes and update
   each query's top-k and shared threshold while GPU filtering continues.
5. The driver gathers IDs and measures recall outside the search timer.

## Interfaces and ownership

`HybridIndex` owns its GPU allocations, streams and workspace. It cannot be copied
or moved. `load_from_raw_pointers()` copies the short index to GPU memory and
borrows host extended codes, factors and IDs. Keep their owners alive until the
engine is destroyed. The CLI holds these arrays in `index::SearchData` and the
loaded RaBitQ IVF object.

`prepare_workspace()` sets query, top-k, probe and vanguard capacity. Searches
check these limits before submitting GPU work. The CLI splits queries into
chunks of at most 65,535; probe count is at most 65,536 and must fit the selected
index. Reprepare the workspace before increasing a capacity. One engine instance
serves one search at a time.

The vendored IVF interface exposes read-only metadata, clusters, IDs and rotation
serialization. Extended-code ownership transfers explicitly to `SearchData` for
bounded in-place compaction. No command driver changes C++ access specifiers.

## Baselines and experiments

The GPU baseline is `coraanns::baselines::GpuIndex`. Optional implementations use
separate `coraanns::experimental::{cpu_gpu,gpu_global,asymmetric,backlog_trace}`
namespaces. Their public headers can be included together. Build experimental
executables with `CORAANNS_BUILD_EXPERIMENTAL=ON`; run them individually.

The CPU baseline uses RaBitQ's parallel CPU search. Shared benchmark options are
in `include/coraanns/benchmark/`.
