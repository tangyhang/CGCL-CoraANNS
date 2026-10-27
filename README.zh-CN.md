# CoraANNS

本项目将 RaBitQ-Library 中主要位于 `sample/` 的 CoraANNS 整理成独立项目，
保留 GPU 过滤、CPU 精炼、先锋簇、共享阈值、CPU 二次过滤和跨查询复用的实现。
本次工作的范围是代码组织、构建入口和实验复现，不引入算法优化。

主要入口：

| 内容 | 位置 |
| --- | --- |
| 构建与小数据运行示例 | [README](README.md) |
| 依赖、CPU/CUDA 构建、可选目标 | [构建说明](docs/BUILD.md) |
| 数据格式、训练、实验配置、recall 口径 | [复现说明](docs/REPRODUCIBILITY.md) |
| 系统模块与代码定位 | [结构说明](docs/ARCHITECTURE.md) |
| 原文件到新文件的对应关系 | [迁移说明](docs/MIGRATION.md) |
| 截图中的提交要求与发布准备 | [开源提交要求](docs/ARTIFACT_REQUIREMENTS.zh-CN.md) |

目录按职责划分：`apps/` 放命令行入口，`src/` 放计算实现，
`include/coraanns/` 放接口与共享实验工具，`configs/` 放参数，
`tools/` 放数据和实验脚本，`third_party/rabitq/` 保留依赖及许可证。
调试、非对称和 backlog trace 版本位于独立的可选实验目录。

可先在没有 GPU 的机器上检查构建、索引序列化和 recall：

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
默认使用本次运行测得的 distance-aware recall；只有明确指定
`--recall-mode id` 时才使用 ID recall，且结果中会标注，避免混淆。
数据集、预构建索引和旧实验结果表未打包进源码。

根目录保留原项目 Apache 2.0 许可证和第三方声明，迁移清单记录原文件及整理后文件的哈希。
当前尚未发布远程仓库，也未填写未提供的论文作者、题目或引用信息。
