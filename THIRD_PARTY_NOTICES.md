# Third-party notices

This project is derived from the local RaBitQ-Library working tree, whose Git
origin is CPU-GPU-IVF-RabitQ and whose HEAD at import was
`14d844154606a901c0c6b58940fcffe2fb32a09f`. The actual imported working-tree
files are recorded by SHA-256 in `docs/import_manifest.json`; HEAD does not
identify uncommitted changes.

The inherited root Apache License 2.0 is retained in `LICENSE`. The RaBitQ
dependency and its existing modifications are under `third_party/rabitq/`;
its license and upstream README are retained there. The CoraANNS implementations
previously mixed into `sample/` and six GPU headers have been moved into the
project's own source and interface directories. This is a source reorganization.

Vendored dependencies retain their own notices:

| Component | Location | License file |
| --- | --- | --- |
| RaBitQ library | `third_party/rabitq/` | `third_party/rabitq/LICENSE` |
| Eigen | `third_party/rabitq/include/rabitqlib/third/Eigen/` | `LICENSE` in that directory (MPL 2.0); retain individual file notices |
| hnswlib | `third_party/rabitq/include/rabitqlib/third/hnswlib/` | `LICENSE` in that directory (Apache 2.0) |

CUDA, cuBLAS, RAFT/RMM/CCCL, NumPy, Faiss, and optional PyTorch are external
dependencies, not redistributed binaries. Datasets are not included.

The original RaBitQ README credits Yutong Gou, Jianyang Gao, Yuexuan Xu, Jifan
Shi, and Zhonghao Yang. See the retained upstream README for its acknowledgments
and research references. No CoraANNS paper title, DOI, authors, or BibTeX entry
has been invented for this repository.
