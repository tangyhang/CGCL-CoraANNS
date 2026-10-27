# Migration provenance

The input was the local RaBitQ-Library working tree, not just a Git revision.
Its HEAD was `14d844154606a901c0c6b58940fcffe2fb32a09f`. The source repository
was read only during this migration. No datasets, indexes, build products,
spreadsheets, profiler traces, historical result CSVs, or local agent settings
were imported.

The import manifest records source and destination SHA-256 hashes, paths, and
the character ranges for the production C++ split. Each source hash covers
the complete original file, including working-tree changes. The include-path
mapping is recorded too. The original dependency headers and license files
are preserved byte for byte; the six system-specific GPU headers now belong
to `include/coraanns/`.

Production files have consistent formatting, disabled-code comment blocks have
been removed, and unreadable CPU/interface comments have been replaced with
concise explanations. With `--source-root`, the audit also compares the active
C++/CUDA token sequences of all 27 imported system/driver/shared-header files.
It ignores comments and whitespace and permits the recorded include moves and
the explicit headers needed by the CPU file split. The computational code is
unchanged by this cleanup.

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

The split CPU file adds its required explicit headers, including
`rabitqlib/utils/space.hpp`; it previously received these transitively through
the command-line driver's IVF includes. Search code, data layouts, defaults,
and option names remain unchanged. The new executable target names replace
the old `ivf_rabitq_*` names; use the mapping above when comparing commands.

New project code consists of the target-based CMake build, named training
wrapper, small demo generator, experiment runner, migration checker, smoke and
runner integration tests, and documentation. Configurations extract probe grids
and selected parameters from named original scripts; they do not import the
scripts' local paths or historical recall values. External BANG/FlowANN/cuVS
comparison adapters and unrelated RaBitQ demos are outside this system artifact.

Check the local snapshot:

```bash
python3 tools/verify_import.py
```

Also check that the original files still match the recorded import sources:

```bash
python3 tools/verify_import.py --source-root /path/to/RaBitQ-Library
```

This check intentionally describes the imported snapshot. Subsequent intentional
code changes should have their own revision and review record; changing the
manifest to conceal an algorithm change would defeat its purpose.
