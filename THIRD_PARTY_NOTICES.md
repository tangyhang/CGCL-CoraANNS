# Third-party notices

CoraANNS builds on the RaBitQ library. Its source snapshot comes from
CPU-GPU-IVF-RabitQ at HEAD `14d844154606a901c0c6b58940fcffe2fb32a09f`, with
file-level SHA-256 records in `docs/import_manifest.json`.

The root license is Apache 2.0. The RaBitQ dependency, its license, and its
upstream README are under `third_party/rabitq/`.

Vendored dependencies retain their own notices:

| Component | Location | License file |
| --- | --- | --- |
| RaBitQ library | `third_party/rabitq/` | `third_party/rabitq/LICENSE` |
| Eigen | `third_party/rabitq/include/rabitqlib/third/Eigen/` | `LICENSE` in that directory (MPL 2.0); retain individual file notices |
| hnswlib | `third_party/rabitq/include/rabitqlib/third/hnswlib/` | `LICENSE` in that directory (Apache 2.0) |

External dependencies include CUDA, cuBLAS, RAFT/RMM/CCCL, NumPy, Faiss, and
optional PyTorch.

The original RaBitQ README credits Yutong Gou, Jianyang Gao, Yuexuan Xu, Jifan
Shi, and Zhonghao Yang. See the upstream README for its acknowledgments and
research references.
