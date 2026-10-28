# CoraANNS

CoraANNS 是基于 IVF-RaBitQ 的近似最近邻搜索系统，结合 GPU 过滤、CPU 精炼、
先锋簇调度、共享阈值、CPU 二次过滤和跨查询复用。

主要入口：

| 内容 | 位置 |
| --- | --- |
| 构建与小数据运行示例 | [README](README.md) |
| 依赖、CPU/CUDA 构建、可选目标 | [构建说明](docs/BUILD.md) |
| 数据格式、训练、实验配置、recall 口径 | [复现说明](docs/REPRODUCIBILITY.md) |
| 系统模块与代码定位 | [结构说明](docs/ARCHITECTURE.md) |
| 原文件到新文件的对应关系 | [迁移说明](docs/MIGRATION.md) |
| 补充材料提交与发布准备 | [开源提交要求](docs/ARTIFACT_REQUIREMENTS.zh-CN.md) |

目录按职责划分：`apps/` 放命令行入口，`src/` 放计算实现，
`include/coraanns/` 放接口与共享实验工具，`configs/` 放参数，
`tools/` 放数据和实验脚本，`third_party/rabitq/` 保留依赖及许可证。
调试、非对称和 backlog trace 版本位于独立的可选实验目录。

CPU-only 构建与测试：

```bash
cmake -S . -B build-cpu -DCORAANNS_ENABLE_CUDA=OFF
cmake --build build-cpu -j2
ctest --test-dir build-cpu --output-on-failure
```

实际运行 LAION100M 消融实验前，可查看完整命令：

```bash
python3 tools/run_experiment.py \
  --config configs/laion100m.json --data-root /data/laion100m \
  --experiment ablation --dry-run
```

实验结果目录保留命令、配置、输入文件元数据、原始日志和 CSV 汇总。
默认使用 distance-aware recall；指定 `--recall-mode id` 可切换为 ID recall。
结果中记录所用的评估模式。

项目采用 Apache 2.0 许可证，依赖归属见 [第三方声明](THIRD_PARTY_NOTICES.md)。
源码对应关系和哈希记录在 `docs/import_manifest.json`。
