# Data and experiment reproduction

## Input data and formats

Experiments use the original squared-L2 workflow without adding normalization.
Keep base-vector ordering, query ordering, and zero-based ground-truth IDs
consistent. The supplied configurations describe these artifact names relative
to `--data-root`:

| Dataset | Base | Queries | Ground truth | IVF clusters |
| --- | --- | --- | --- | --- |
| LAION100M | `laion100m-768d_base.fbin` | `laion-768d_query.fbin` | `laion100m_gt.ivecs` | 65,536 |
| Deep1B | `base.1B.fbin` | `query.public.10K.fbin` | `deep1B_groundtruth.ibin` | 262,144 |
| Wiki88M | `base.88M.fbin` | `queries.fbin` | `groundtruth.88M.neighbors.ibin` | 65,536 |

All three use 8 total quantization bits. Index names are
`ivf_<nlist>_8_gpu.index`. Edit a copy of the configuration if your artifacts
have different names. Configured paths can also be absolute. Source datasets
and prebuilt indexes are not included or automatically downloaded. The exact
100M/88M subsets and official public download locations were not supplied with
this migration; provide those when publishing rather than assuming a different
subset is equivalent.

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

The named training interface keeps the original recipe: `faiss.index_factory`
with `IVF<nlist>,Flat`, `index.train(base)`, centroid reconstruction, then a
one-nearest-centroid assignment for each vector. It preserves Faiss's default
clustering behavior and adds no sampling or training optimizations. The original
positional tool is retained as `tools/data/train_ivf_legacy.py` for comparison.
Pin the Faiss version in the environment used for published experiments; its
defaults may differ between releases.

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

The CPU builder accepts the same arguments. Existing indexes from the source
repository can be used directly; serialization has not changed. The GPU
builder's `RABITQ_GPU_BATCH_SIZE` environment variable is retained. Training
and index construction load the base matrix into host memory; the GPU batch
setting does not make those stages out of core. Large datasets need sufficient
host RAM, disk space, and GPU memory for the chosen settings.

`tools/data/generate_gt_gpu.py --help` exposes the retained exhaustive L2
ground-truth generator for `.fbin` inputs. It uses guarded FP32 candidate
filtering and FP64 distance recomputation and writes `.ivecs`. This tool is
separate from ANN indexing and requires NumPy/PyTorch. The demo generator in
the root README provides small deterministic data without external downloads.

## Run a sweep

```bash
python3 tools/run_experiment.py \
  --config configs/deep1b.json --data-root /data/deep1b \
  --build-dir build --experiment overall --numa-interleave
```

`--dry-run` prints fully expanded, shell-quoted commands without checking input
existence or creating results. Execution validates the required files and
executables before starting. Each new run gets an independent directory. An
existing `--output-dir` is rejected. Commands are executed as argument lists,
without a shell. A nonzero child exit or an incomplete output table fails the
run and leaves its logs and failure metadata available.

| Sweep | Systems or varied parameter |
| --- | --- |
| `overall` | CoraANNS, CPU RaBitQ, and GPU RaBitQ with host extended codes |
| `ablation` | The five systems below, using the same nprobe grid |
| `vanguard` | Vanguard sizes 1, 2, 4, 8 |
| `threads` | 18, 36, 72 workers, with vanguard size 4 |
| `topk` | LAION100M top-500 and top-1000 with their original probe grids |

For high-k, build `coraanns_search_highk` first. The runner uses the original
512-capacity binary through top-512 and selects the 1024-capacity binary above
that. Supply ground truth with sufficient neighbors; the LAION top-k sweep
expects `laion100m_gt_top1000.ivecs`.

The ablation mapping is fixed:

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

`parameter_sources` in each JSON names the original scripts used for its grids
and chunk sizes. The `overall` grid uses the ablation grid for a consistent new
comparison. This does not recreate every row in the original workbooks. In
particular, no 0.99 recall threshold is assumed for LAION's probe grid; check
the measured recall before drawing such a conclusion. Changes in hardware,
NUMA binding, compiler flags, chunk size, or metric must accompany reported results.

## Timing and recall

Search executables keep their original timer boundaries and emit the final
`nprobe / QPS / recall` table. The runner reads that table rather than deriving
QPS from wall time. QPS remains the arithmetic mean of per-round QPS. Dataset
loading and index setup are outside the search timing. GPU filtering time
and candidate diagnostics are separate measurements and must not be compared
to uninstrumented end-to-end QPS without qualification.

`--recall-base FILE` enables the retained distance-aware evaluator. It counts
returned GT IDs and additional IDs satisfying the GT distance threshold; ties,
duplicates, invalid IDs, and numeric boundaries follow the source evaluator's
existing rules. Without a base file, the executable reports ID recall only.
Every result table and metadata file records the chosen mode. A run in ID mode
cannot be presented as calibrated distance-aware recall.

The runner defaults to distance-aware mode and passes `--recall-base`; it
never silently substitutes historical values. Explicit `--recall-mode id`
permits timing-only work when the base vectors are unavailable. The base file
is used for evaluation, not automatically for search refinement.
`--exact-base FILE` enables the separate FP32 refinement diagnostic in the main
driver; it is not a standard paper configuration and the runner does not set it.

The main and GPU drivers also retain `--recall-once true`, which defers recall
evaluation until after timing and evaluates the first measured round's saved
results. The standard runner leaves the original default (false). If you use
this option directly, report its meaning with the results.

## Saved outputs

- `commands.sh`: exact executable arguments and relevant environment settings.
- `metadata.json`: resolved configuration, input file sizes/mtimes, environment,
  import-manifest hash, job return codes, and completion/failure state.
- One raw `.log` per system or parameter setting.
- `summary.csv`: dataset, experiment, job, top-k, nprobe, QPS, recall, recall
  mode, and source log; completed jobs are retained if a later job fails.

The runner does not hash multi-terabyte datasets. Archive your final source
revision, dependency environment, hardware details, and dataset/index checksums
along with published results. No historical tables or profiler outputs are
included as newly measured evidence.
