# Implementation map

The project separates command-line orchestration, search computation, and
RaBitQ support code.

| Responsibility | Main location |
| --- | --- |
| Inputs, options, index adaptation, warmups, measurement | `apps/search.cpp` |
| Shared search state and CPU/GPU interface | `include/coraanns/search/hybrid.cuh` |
| GPU query preparation, probe selection, pair scheduling, filtering | `src/search/hybrid.cu` |
| CPU task consumption, secondary filtering, refinement, top-k updates | `src/search/cpu_refinement.cpp` |
| CPU index construction entry | `apps/build_index.cpp` |
| GPU index construction | `apps/build_index_gpu.cpp`, `src/index/build_gpu.cu` |
| IVF serialization, quantization, FastScan, rotations | `third_party/rabitq/include/rabitqlib/` |
| CLI and shape validation | `include/coraanns/benchmark/options.hpp` |
| Recall evaluation and mapped base access | `include/coraanns/benchmark/recall.hpp`, `distance.hpp` |
| Parameter sweeps and result collection | `configs/`, `tools/run_experiment.py` |

## Search flow

1. The driver reads queries, ground truth, and the IVF index. It adapts
   RaBitQ's packed codes, factors, and IDs to the GPU/CPU data layout.
2. The search engine allocates its workspace and prepares queries and probe
   lists. Pair scheduling processes vanguard clusters first.
3. GPU filtering publishes candidates to the shared task queue. Vanguard
   processing supplies early thresholds and performs recovery scanning.
4. CPU workers consume tasks, optionally apply the secondary threshold check,
   refine from extended RaBitQ codes, and update per-query top-k and thresholds.
5. The driver gathers IDs, applies the recall evaluator outside search
   timing, and prints per-round measurements and the final table.

The CPU worker implementation is shared by the search library and GPU
correctness tests. The driver handles packed-index adaptation and owns the
host arrays used by the engine.

## Other implementations

`apps/baselines/cpu.cpp` uses the parallel CPU RaBitQ search. The GPU baseline
is split between `apps/baselines/gpu.cpp`, its interface, and
`src/baselines/gpu.cu`. It supports host-excode, HBM, cache, and tiled-kernel
options.

`experimental/` contains CPU-filter/GPU-refinement, global GPU, asymmetric
two-phase, and backlog-tracing variants. Build them with
`CORAANNS_BUILD_EXPERIMENTAL=ON` and run their individual executables. Shared
mechanism options are in `include/coraanns/benchmark/mechanism_options.hpp`.
