# Data and experiment reproduction

## Input data and formats

Experiments use raw input vectors and squared-L2 distance. Keep base-vector
ordering, query ordering, and zero-based ground-truth IDs consistent.
The supplied configurations describe these artifact names relative
to `--data-root`:

| Dataset | Base | Queries | Ground truth | IVF clusters |
| --- | --- | --- | --- | --- |
| LAION100M | `laion100m-768d_base.fbin` | `laion-768d_query.fbin` | `laion100m_gt.ivecs` | 65,536 |
| Deep1B | `base.1B.fbin` | `query.public.10K.fbin` | `deep1B_groundtruth.ibin` | 262,144 |
| Wiki88M | `base.88M.fbin` | `queries.fbin` | `groundtruth.88M.neighbors.ibin` | 65,536 |

All three use 8 total quantization bits. Index names are
`ivf_<nlist>_8_gpu.index`. Edit a copy of the configuration if your artifacts
have different names. Configured paths can also be absolute. Place your dataset
files under `--data-root`, then train and build the index using the commands below.

Binary layouts, on the target little-endian Linux platform:

| Format | Layout |
| --- | --- |
| `.fbin` | Two uint32 values `(rows, dimensions)`, then contiguous float32 rows |
| `.ibin` | Two uint32 values `(rows, neighbors)`, then contiguous uint32 IDs |
| `.fvecs` | Each row: int32 dimension followed by float32 values |
| `.ivecs` | Each row: int32 width followed by int32 IDs |

The index builders accept `.fbin` or `.fvecs` base vectors, `.fvecs` centroids,
and `.ivecs` assignments with one cluster ID per base vector. Query drivers
autodetect the corresponding query and ground-truth formats. The evaluator
requires at least `topk` ground-truth neighbors per query.

## Train and construct an index

The training interface uses `faiss.index_factory`
with `IVF<nlist>,Flat`, `index.train(base)`, centroid reconstruction, then a
one-nearest-centroid assignment for each vector, with Faiss's default clustering
settings. The positional interface is available as
`tools/data/train_ivf_legacy.py`. Record the Faiss version with your experiment
environment.

Example for LAION100M:

```bash
python3 tools/data/train_ivf.py \
  --base /data/laion100m/laion100m-768d_base.fbin --nlist 65536 \
  --centroids /data/laion100m/centroids.fvecs \
  --assignments /data/laion100m/assignments.ivecs --metric l2
build/bin/coraanns_build_index_gpu \
  /data/laion100m/laion100m-768d_base.fbin \
  /data/laion100m/centroids.fvecs /data/laion100m/assignments.ivecs \
  8 /data/laion100m/ivf_65536_8_gpu.index l2 false
```

The CPU builder accepts the same arguments. Existing RaBitQ-Library IVF indexes
can be loaded directly. Set `RABITQ_GPU_BATCH_SIZE` to control the GPU builder's
batch size. Training and index construction load the base matrix into host
memory; size host RAM for the full matrix and construction buffers.

`tools/data/generate_gt_gpu.py --help` describes the exhaustive L2
ground-truth generator for `.fbin` inputs. It uses guarded FP32 candidate
filtering and FP64 distance recomputation and writes `.ivecs`. It requires
NumPy/PyTorch. The demo generator in the root README creates a small,
deterministic dataset.

## Run a sweep

```bash
python3 tools/run_experiment.py \
  --config configs/deep1b.json --data-root /data/deep1b \
  --build-dir build --experiment overall --numa-interleave
```

Use `--dry-run` to inspect the expanded commands. Execution validates input
files and executables, then creates a new results directory. Choose a fresh
path for `--output-dir`. Failed jobs keep their logs and exit status in the
results directory.

| Sweep | Systems or varied parameter |
| --- | --- |
| `overall` | CoraANNS, CPU RaBitQ, and GPU RaBitQ with host extended codes |
| `ablation` | The five systems below, using the same nprobe grid |
| `vanguard` | Vanguard sizes 1, 2, 4, 8 |
| `threads` | 18, 36, 72 workers, with vanguard size 4 |
| `topk` | LAION100M top-500 and top-1000 with separate probe grids |

For high-k, build `coraanns_search_highk` first. The runner uses the
512-capacity binary through top-512 and selects the 1024-capacity binary above
that. Supply ground truth with sufficient neighbors; the LAION top-k sweep
expects `laion100m_gt_top1000.ivecs`.

The ablation mapping is:

| System | Executable | Vanguard | CPU secondary filter |
| --- | --- | --- | --- |
| Baseline | `rabitq_gpu --index-residency host-excode` | N/A | N/A |
| Technique 1 | `coraanns_search` | 0 | false |
| Technique 1 + 2 | `coraanns_search` | 5 | false |
| Technique 1 + 3 | `coraanns_search` | 0 | true |
| All techniques | `coraanns_search` | 5 | true |

Technique 1 is GPU filtering and CPU refinement; technique 2 is vanguard
clusters; technique 3 is the CPU secondary filter. Shared settings are
top-k 100, query batch 4, 72 CPU threads, one warmup, five measured rounds,
and profiling disabled. `--threads N` overrides the fixed worker count for
other hardware; the `threads` sweep uses its own configured list.

`parameter_sources` in each JSON records the source scripts for its probe grids
and chunk sizes. The `overall` and `ablation` sweeps use the same grid. Include
hardware, NUMA binding, compiler flags, chunk size, and metric with reported results.

## Timing and recall

Search executables emit a final `nprobe / QPS / recall` table, which the runner
uses for its summary. QPS is the arithmetic mean of per-round search QPS.
Dataset loading and index setup precede search timing. GPU event timing covers
the filtering phase; candidate statistics describe filtering and refinement work.

The runner defaults to distance-aware recall. Every result table and metadata
file records the selected mode.

| Option | Effect |
| --- | --- |
| Runner: `--recall-mode distance-aware` | Passes base vectors to the evaluator through `--recall-base` |
| Runner: `--recall-mode id` | Evaluates returned IDs against ground-truth IDs |
| Search driver: `--recall-base FILE` | Counts GT ID hits and additional returned IDs satisfying the GT distance threshold |
| `coraanns_search`: `--exact-base FILE` | Uses FP32 base vectors for refinement |

The standard runner configures recall evaluation. For FP32 refinement
diagnostics, invoke `coraanns_search` directly with `--exact-base`.

The main and GPU drivers support `--recall-once true`: recall is evaluated
after timing using the first measured round's saved results. The default is
false, which evaluates recall for each measured round.

## Saved outputs

- `commands.sh`: exact executable arguments and relevant environment settings.
- `metadata.json`: resolved configuration, input file sizes/mtimes, environment,
  import-manifest hash, job return codes, and completion/failure state.
- One raw `.log` per system or parameter setting.
- `summary.csv`: dataset, experiment, job, top-k, nprobe, QPS, recall, recall
  mode, and source log; completed jobs are retained if a later job fails.

Archive the source revision, dependency environment, hardware details, and
dataset/index checksums with published results.
