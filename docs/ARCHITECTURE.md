# Implementation map

The project separates command-line orchestration from production computation
and inherited RaBitQ support code. The original implementation class names and
layout remain intact to keep this migration independent of algorithm changes.

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

1. The driver reads queries, ground truth, and the existing IVF index. It adapts
   RaBitQ's packed index layout to the original GPU/CPU data layout, preserving
   code compaction, factors, IDs, and ownership lifetimes.
2. The search engine allocates its workspace and prepares queries and probe
   lists. Pair scheduling retains the vanguard-first ordering.
3. GPU filtering publishes candidates to the original shared task mechanism.
   Vanguard processing supplies early thresholds and retains recovery behavior.
4. CPU workers consume tasks, optionally apply the secondary threshold check,
   refine from extended RaBitQ codes, and update per-query top-k and thresholds.
5. The driver gathers IDs, applies the retained recall evaluator outside search
   timing, and prints per-round measurements and the final table.

The candidate queue, memory fences, worker scheduling, kernel choices,
threshold updates, LUT arithmetic, and index binary layout have not been
redesigned. The production CPU worker definitions were moved out of the
command-line file so they can be linked by the library and GPU correctness
tests without importing `main()`.

The driver still uses the inherited packed-layout adapter, including its
`private` visibility macro around RaBitQ includes. It is a compatibility seam,
not a stable public serialization API. This migration keeps it to preserve the
existing loading and ownership behavior; future API work should be reviewed
separately from the paper artifact.

## Other implementations

`apps/baselines/cpu.cpp` uses the parallel CPU RaBitQ search. The GPU baseline
is split between `apps/baselines/gpu.cpp`, its interface, and
`src/baselines/gpu.cu`; host-excode, HBM, cache, and tiled-kernel options remain.

`experimental/` retains CPU-filter/GPU-refinement, global GPU, asymmetric
two-phase, and backlog-tracing variants. These do not replace the main system
and are opt-in through CMake. Their CLI and algorithm-specific output formats
remain as in the source. The common mechanism CLI is retained separately as
`include/coraanns/benchmark/mechanism_options.hpp`.
