# 环形队列验收记录（2026-11-01）

## 环境与参数

- GPU：Tesla V100-PCIE-32GB；CPU：双路 Xeon Gold 5220，72 线程。
- CUDA 12.6.85；RAFT/RMM 26.06；CCCL 3.4.0；CMake 3.31.10。
- 整理版和环形版 C++：GCC 13.4，Release `-Ofast -march=native -mavx2 -mfma -mbmi2`。CUDA host：GCC 11.4；NVCC `-O3`，SM70。
- topk=100，batch-q=4，Vanguard=5，secondary filter=true，cross-query reuse=true。
- `OMP_PROC_BIND=close`、`OMP_PLACES=cores`、`OMP_DYNAMIC=FALSE`，`numactl --interleave=all`。
- 关闭 profile、candidate-stats、gpu-filter-timing。每进程预热 1 轮、测量 5 轮，每版两个独立进程，串行 A/B/B/A。
- reset、GPU 查询、CPU 精炼及最终回灌和同步均在计时内。recall 使用第一轮结果，在所有 QPS 测量结束后计算。
- Deep100M、LAION100M 使用 distance-aware recall；Wiki88M 使用 ID recall。

## 原项目与整理后的线性版本

原项目使用 `RaBitQ-Library/sample/results/ring_queue/before/` 中的线性版本快照，其三个生产源文件哈希与本项目初次导入记录一致。整理版冻结于提交 `18c289b70fd7313f3076349504e82d3f1e075187`。

| 数据集 | nprobe | 原项目 QPS | 整理版 QPS | 变化 |
| --- | ---: | ---: | ---: | ---: |
| deep100m | 34 | 80902.29 | 81979.95 | +1.33% |
| deep100m | 112 | 62024.84 | 62236.29 | +0.34% |
| deep100m | 320 | 41944.37 | 42177.84 | +0.56% |
| deep100m | 640 | 31029.35 | 31097.28 | +0.22% |
| wiki88m | 32 | 45201.18 | 45455.65 | +0.56% |
| wiki88m | 200 | 20999.01 | 21054.63 | +0.26% |
| wiki88m | 400 | 13039.76 | 13050.88 | +0.09% |
| wiki88m | 750 | 7940.74 | 7955.63 | +0.19% |
| wiki88m | 1024 | 6081.09 | 6085.58 | +0.07% |
| laion100m | 80 | 19950.44 | 20237.24 | +1.44% |
| laion100m | 320 | 9992.47 | 10076.64 | +0.84% |
| laion100m | 1000 | 5012.00 | 5034.64 | +0.45% |
| laion100m | 3200 | 1985.65 | 1989.74 | +0.21% |

13 个点均为 `PASS`。各数据集两个原版进程均值的最大跨度分别为 1.52%、0.66%、1.09%，低于 3% 验收阈值。

## 正确性与编译检查

6 项 CPU CTest 和 4 项 GPU CTest 全部通过。新增测试覆盖乱序和延迟归还、64 位位置、代际复用、1/4/16/72 线程、Vanguard/二次过滤开关、查询分块、部分块、零候选、重复 reset、强制 HBM 回灌、暂存区耗尽及重置恢复；核对候选数、top-k ID 和距离。

| 默认 Q=4 内核 | 寄存器 | Stack | 静态 Local | 静态 Shared |
| --- | ---: | ---: | ---: | ---: |
| 整理版线性队列 | 77 | 0 B | 0 B | 256 B |
| 环形队列 | 80 | 0 B | 0 B | 256 B |

两个版本在 SM70 上的寄存器预算都允许驻留三个 256-thread blocks。HBM 缓存与刷新锁使用显式 global-space acquire/release 操作，避免了通用地址空间原子操作的栈临时变量。消息发布使用 `STG.E.128`。默认 Q=4 的水位刷新冷函数有两处 `MEMBAR.ALL.GPU`，分别用于缓存发布和刷新锁归还；逐条消息发布路径没有屏障。CPU 消息读写使用 `VMOVDQA`，没有 `MFENCE`。

本次变更也包含 CPU 按 `done + produced` 停止领取、消费后清空槽位，以及释放每次精炼分配的临时 top-k 存储。筛选与精炼公式、数据精度和计时边界保持一致。

## 环形队列与整理版线性队列

正式容量为 `2^24` 条主机记录（256 MiB），HBM 暂存为 `2^24` 条（512 MiB）。三个数据集均使用新一组 A/B/B/A，每版 2 个进程、10 个测量轮次，全部纳入统计。

### Deep100M（distance-aware recall）

| nprobe | 线性 QPS | 环形 QPS | 变化 | 95% 比值区间 | 判定 |
| ---: | ---: | ---: | ---: | --- | --- |
| 34 | 81386.64 | 81448.61 | +0.08% | 0.9847–1.0160 | PASS |
| 112 | 61770.62 | 62199.38 | +0.69% | 0.9935–1.0221 | PASS |
| 320 | 42283.41 | 44250.06 | +4.65% | 1.0431–1.0491 | PASS |
| 640 | 31175.50 | 32359.63 | +3.80% | 1.0366–1.0394 | PASS |

最大 recall 均值绝对差为 `0.0000145`；线性基线进程均值最大跨度为 `0.88%`。

### Wiki88M（ID recall）

| nprobe | 线性 QPS | 环形 QPS | 变化 | 95% 比值区间 | 判定 |
| ---: | ---: | ---: | ---: | --- | --- |
| 32 | 45283.39 | 45546.42 | +0.58% | 0.9967–1.0143 | PASS |
| 200 | 21034.25 | 21281.31 | +1.17% | 1.0082–1.0153 | PASS |
| 400 | 13062.03 | 13268.27 | +1.58% | 1.0118–1.0213 | PASS |
| 750 | 7954.02 | 8068.80 | +1.44% | 1.0124–1.0169 | PASS |
| 1024 | 6096.41 | 6168.40 | +1.18% | 1.0072–1.0166 | PASS |

最大 recall 均值绝对差为 `0.0000095`；线性基线进程均值最大跨度为 `0.74%`。

### LAION100M（distance-aware recall）

| nprobe | 线性 QPS | 环形 QPS | 变化 | 95% 比值区间 | 判定 |
| ---: | ---: | ---: | ---: | --- | --- |
| 80 | 20052.22 | 21617.09 | +7.80% | 1.0615–1.0953 | PASS |
| 320 | 9986.12 | 10155.95 | +1.70% | 0.9825–1.0439 | PASS |
| 1000 | 5034.46 | 5112.22 | +1.54% | 1.0113–1.0206 | PASS |
| 3200 | 1996.24 | 2028.36 | +1.61% | 1.0151–1.0170 | PASS |

最大 recall 均值绝对差为 `0.0000150`；线性基线进程均值最大跨度为 `0.91%`。

共 13 个 `PASS`。Deep100M 的 nprobe=34 为 +0.08%，应理解为基本持平。所有正式轮次 `spilled=0`。Deep100M 每轮实际回绕 1–2 次，Wiki88M 和 LAION100M 无回绕。

首版使用 CCCL 通用地址空间原子操作，Deep100M 的 nprobe=34 首测为 `−1.40% / WITHIN_NOISE`；追加一次 A/B/B/A 后，合并结果为 `−1.72% / FAIL`，未作为最终通过版本。随后将已确定驻留 HBM 的缓存和刷新锁操作封装为等价的 global-space acquire/release 操作，移除了通用地址判断和 CAS 栈临时变量。最终版本重新测量全部三个数据集。首版失败样本和二进制指纹保存在 JSON 的 `rejected_v1` 中。

按 [验收规则](RING_QUEUE.md#qps-comparison)，均值上升记为 `PASS`；均值略低且分层 bootstrap 95% 比值区间包含 1，记为 `WITHIN_NOISE`。基线进程均值跨度须小于等于 3%，recall 均值下降须不超过 `1e-4`。bootstrap 使用 10,000 次重采样，固定种子 `20261031`。

## 复跑与证据

从提交 `18c289b70fd7313f3076349504e82d3f1e075187` 按 [构建说明](BUILD.md) 构建线性版本，将 `LINEAR` 指向其可执行文件。环形版本使用当前 `build/bin/coraanns_search`。

```bash
python3 tools/benchmark_queue.py --before "$LINEAR" --after build/bin/coraanns_search \
  --dataset deep100m --data-root "$DEEP_ROOT" --output-dir results/queue-recheck-deep100m
python3 tools/benchmark_queue.py --before "$LINEAR" --after build/bin/coraanns_search \
  --dataset wiki88m --data-root "$WIKI_ROOT" --recall-mode id --output-dir results/queue-recheck-wiki88m
python3 tools/benchmark_queue.py --before "$LINEAR" --after build/bin/coraanns_search \
  --dataset laion100m --data-root "$LAION_ROOT" --output-dir results/queue-recheck-laion100m
```

[完整逐轮数据与环境记录](benchmarks/ring_queue_20261101.json) 包含两阶段的样本、recall、逐轮队列统计、命令、输入元数据、二进制与核心源码 SHA-256、编译参数、正确性测试日志和默认内核资源。完整本地证据目录为 `results/ring_queue/20261101_migration/`，其中保存冻结的源码/二进制、原始日志、SASS、PTX 和 CPU 汇编。

二进制 SHA-256：

| 版本 | SHA-256 |
| --- | --- |
| organized_linear | `20a1c09e73f26739d1f2801580e3b187ad7c04a53871671fd2bba0aec8851794` |
| legacy_linear | `5252f3a65fb457d2fd731b42a206f9b62565a6558989c69493b9124589e0d68d` |
| ring | `dbc0b4ba8b264f82c732e6f4ed418ea2c09dc22ca4e3ae5de1d9074ac6271fb0` |
