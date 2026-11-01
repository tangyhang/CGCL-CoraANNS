# CPU–GPU candidate ring

GPU filtering publishes 16-byte candidates into a mapped host ring. CPU
workers refine candidates and return cleared storage to the producer.

| Position | Meaning |
| --- | --- |
| `Pw` | Next logical reservation, a 64-bit counter |
| CPU claim head | Next chunk assigned to a worker |
| `Pr` | End of the contiguous, consumed and cleared prefix |

For capacity `C`, the physical slot is `p & (C - 1)`. A reserved position `p`
can be published when `p < Pr + C`. Capacities are positive powers of two.

## Publication and release

1. A producer reserves a logical position and checks its cached writable limit.
2. At the limit, a warp checks the shared HBM snapshot. A device refresh lock
   lets one owner acquire the host `Pr` and update that snapshot. The HBM words
   use global-space acquire/release operations to keep generic-address atomic
   fallbacks out of the producer.
3. A writable candidate is transferred as one aligned 16-byte message. A full
   ring sends the candidate and its original logical position to HBM storage.
4. CPU workers claim chunks of `min(8192, C)` records. A generation check keeps
   a new logical chunk from reading the preceding generation's live records.
5. A worker snapshots each message, refines it, then clears its entire chunk.
   The completion table tracks logical end positions. Its mutex joins clears
   from different workers before a release store advances the contiguous `Pr`.
6. After producer kernels finish, the finish kernel scans pending HBM records
   and refills writable positions. It publishes `done` after the refill completes.
   HBM exhaustion reports an overflow and aborts the search.

Query reset clears counters and progress. Consumers clear reused payload slots
as they finish. The finish kernel and final synchronization are included in
search timing.

## Modules

| File | Responsibility |
| --- | --- |
| `include/coraanns/queue/candidate_queue.hpp` | Payload, shared progress, spill record and device descriptor |
| `include/coraanns/queue/release_tracker.hpp` | Contiguous CPU release tracking |
| `src/search/ring_queue.cuh` | GPU watermark cache, publication, reset and refill |
| `src/search/cpu_refinement.cpp` | Message snapshots, chunk consumption and clearing |
| `src/search/hybrid.cu` | Search kernels and queue storage lifecycle |

The tested fast path uses GPU `st.global.v4.u32` and Intel AVX `VMOVDQA`
message accesses. Progress uses acquire/release synchronization. The platform
references are [CUDA mapped-memory access guarantees](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html#mapped-memory)
and [GCC's Intel AVX 16-byte atomic implementation](https://gcc.gnu.org/pipermail/gcc-patches/2022-February/590960.html).

## Capacity options

| Option | Default | Allocation |
| --- | --- | --- |
| `--queue-capacity` | 67,108,864 | 1 GiB mapped host memory, 16 bytes per record |
| `--queue-spill-capacity` | 16,777,216 | 512 MiB HBM, 32 bytes per record |

`--queue-spill-capacity 0` disables spill storage. Set capacities before workspace
allocation. The experiment runner accepts both options and forwards them to
the hybrid search executables.

For a direct LAION100M top-100 run, set `DATA_ROOT` to the dataset directory:

```bash
build/bin/coraanns_search \
  "$DATA_ROOT/ivf_65536_8_gpu.index" \
  "$DATA_ROOT/laion-768d_query.fbin" "$DATA_ROOT/laion100m_gt.ivecs" \
  --topk 100 --nprobes 320 --rounds 5 --warmup 1 \
  --threads 72 --batch-q 4 --query-chunk 1000 \
  --vanguard-size 5 --cpu-secondary-filter true --cross-query-reuse true \
  --queue-capacity 16777216 --queue-spill-capacity 16777216 \
  --recall-once true --recall-base "$DATA_ROOT/laion100m-768d_base.fbin"
```

Each measured round emits `QueueStats` with total submitted candidates, spilled
candidates, and physical wrap count. For a single query chunk, the wrap count is
`(submitted - 1) / C` for a nonempty queue. Multi-chunk runs sum these counts.

## Correctness checks

```bash
cmake -S . -B build -DCORAANNS_BUILD_GPU_TESTS=ON
cmake --build build --target ring_release_tracker_test gpu_cpu_ring_queue_test -j2
ctest --test-dir build -R 'ring_release_tracker_test|gpu_cpu_ring_queue_test' --output-on-failure
```

The CPU test covers delayed and out-of-order completion, repeated reuse,
concurrent release, and positions beyond 32 bits. GPU integration tests link
the production engine and verify candidate counts, nearest IDs and distances,
partial chunks, empty searches, reset, refill, overflow recovery, vanguard and
secondary-filter settings, and CPU thread counts 1/4/16/72.

## QPS comparison

Freeze the baseline source, binary, CMake settings and SHA-256 before building
the ring version. The benchmark tool runs the binaries serially in A/B/B/A
order, with one warmup and five measured rounds per process.

```bash
python3 tools/benchmark_queue.py \
  --before /path/to/frozen/coraanns_search \
  --after build/bin/coraanns_search \
  --dataset laion100m --data-root /data/laion100m \
  --queue-capacity 16777216 --queue-spill-capacity 16777216 \
  --output-dir results/queue-laion100m
```

`--linear-after` compares two linear implementations, such as the pre-migration
and organized versions. Use `--recall-mode id` for datasets evaluated by ID.
Probe grids are in `configs/queue_benchmarks.json`; Deep100M checks require
actual wrapping in every measured round.

The comparison uses all rounds, a stratified bootstrap ratio interval, and a
`1e-4` recall-drop tolerance. It reports `PASS`, `WITHIN_NOISE`, or a failure
reason for each nprobe. Baseline process-mean spans above 3% are marked unstable.
Zero-spill comparisons also require a queue record for every measured round.
The result directory contains commands, binary fingerprints, input metadata,
raw logs and `comparison.json`.

See the [2026-11-01 validation report](RING_QUEUE_VALIDATION.zh-CN.md) for the
linear-project baseline, final ring measurements and compiler checks.
