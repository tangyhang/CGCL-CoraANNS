# Source map

The source snapshot comes from the RaBitQ-Library working tree at HEAD
`14d844154606a901c0c6b58940fcffe2fb32a09f`. `import_manifest.json` records source
and destination paths, SHA-256 hashes, include-path mappings, and the character
ranges used to split the production C++ file.

| Original path in RaBitQ-Library | Current path |
| --- | --- |
| `sample/ivf_rabitq_querying_gpu_cpu.cpp` before CPU worker definitions | `apps/search.cpp` |
| Same file's CPU worker and threshold methods | `src/search/cpu_refinement.cpp` |
| `sample/ivf_rabitq_querying_gpu_cpu.cu` | `src/search/hybrid.cu` |
| `include/rabitqlib/index/ivf_rabitq_querying_gpu_cpu.cuh` | `include/coraanns/search/hybrid.cuh` |
| `sample/ivf_rabitq_indexing.cpp` | `apps/build_index.cpp` |
| `sample/ivf_rabitq_indexing_gpu.cpp`, `.cu` | `apps/build_index_gpu.cpp`, `src/index/build_gpu.cu` |
| `sample/ivf_rabitq_querying_parallel.cpp` | `apps/baselines/cpu.cpp` |
| `sample/ivf_rabitq_querying_gpu.cpp`, `.cu` | `apps/baselines/gpu.cpp`, `src/baselines/gpu.cu` |
| CPU/GPU, global, asymmetric, trace variants | `apps/experimental/`, `src/experimental/`, `include/coraanns/experimental/` |
| Shared benchmark, mechanism, recall, mapped-base headers | `include/coraanns/benchmark/` |
| Remaining `include/rabitqlib/` | `third_party/rabitq/include/rabitqlib/` |
| `python/ivf.py`, `python/utils/io.py` | `tools/data/train_ivf_legacy.py`, `vector_io.py` |
| `sample/generate_gt_gpu.py` | `tools/data/generate_gt_gpu.py` |

The CPU worker file includes `rabitqlib/utils/space.hpp` for extended-code
distance functions. Project tools provide target-based builds, IVF training,
demo data, parameter sweeps, and result collection.

## Verify the source snapshot

Check imported-file hashes:

```bash
python3 tools/verify_import.py
```

Compare against the RaBitQ-Library working tree:

```bash
python3 tools/verify_import.py --source-root /path/to/RaBitQ-Library
```

With `--source-root`, the checker also compares the active C++/CUDA token
sequences of 27 implementation, driver, and shared-header files after applying
the recorded include mappings and split-file headers.
