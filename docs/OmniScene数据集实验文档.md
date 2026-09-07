# OmniScene Center150 实验说明（LoopSparseGS / comp_svfgs）

## 实验目标

本分支把 LoopSparseGS 作为逐场景优化基线运行在 OmniScene 上。默认协议为：

- 使用 nuScenes val 的 150 个场景，每个场景只取一个中央 bin；
- 每个样本从头初始化并进行一次连续的 10,000 次优化；
- 在同一条训练轨迹的 1,000、5,000、10,000 次迭代处评估；
- 每次同时报告完整 18 路目标视角和前 12 路新视角（`00`--`11`）的 PSNR、SSIM、LPIPS，并保存累计训练耗时；
- 150 个样本全部完成后，按场景等权平均并自动生成汇总结果。

这不是四轮独立训练，也不会根据中间评估结果调整或重启优化。

## Center150 数据所有权

Center150 清单位于：

```text
<data_root>/interp_12Hz_trainval/bins_center150_v1.json
```

清单只能由主项目 `~/Projects/SVF-GS` 生成。本项目不会生成或修改它，只会在加载前执行以下校验：

- OmniScene val 清单含 150 个场景分组；
- `adjacent_bins` 展平后与 val 的 `bins` 完全一致；
- 每个场景的 bin 连续、有序，场景 token 唯一；
- Center150 与 SVF-GS 的 lower-median 规则逐项一致；
- 150 个 bin 对应的 `bin_infos_3.2m/*.pkl` 均存在。

任何一项不满足都会直接报错，避免静默使用错误子集。

## 数据预处理

每个 bin 会被转换成 LoopSparseGS 可加载的独立场景：

```text
output/omniscene_preproc/112x200/
  001_<bin_token>/
    images/train/        # 6 个输入视角
    images/test/         # 12 个非输入目标视角
    depth/train/
    depth/test/
    depth/conf/
    cams/train.json
    cams/test.json       # 共 18 个评估视角
    points/init_points.ply
```

测试集由 12 个非输入视角和 6 个输入视角组成。`cams/test.json` 中的评估名称固定保持为 `00` 到 `17`，对应输出为 `00.png` 到 `17.png`。

Metric3D 深度、置信度阈值、深度截断/存储尺度以及初始点云生成逻辑保持本分支既有设计，本次 Center150 改造没有调整这些算法参数。

## 连续训练和阶段评估

默认训练命令由批处理脚本构造，核心参数为：

```text
--iterations 10000
--test_iterations 1000 5000 10000
--save_iterations 10000
--full_eval_metrics
-sps
```

三个里程碑属于同一次连续优化。中间评估只读取当前高斯状态，不加载新权重、不反馈指标、不改变训练参数；评估前后的 Torch RNG 状态会恢复，以免评估影响后续训练轨迹。18 路指标使用 `00`--`17`，其中 `novel_12` 严格按图像名选取 `00`--`11`，不依赖可能被 shuffle 的相机遍历顺序。

每个里程碑写出：

```text
<scene_output>/
  evaluation/iteration_<N>.json
  metrics_<N>.txt
  metrics_novel_12_<N>.txt
  training_time_<N>.txt
  test/ours_<N>/renders/00.png ... 17.png
  test/ours_<N>/gt/00.png ... 17.png
```

其中 `evaluation/iteration_<N>.json` 同时包含兼容旧格式的 18 路 `metrics`，以及 `view_metrics.all_18` 和 `view_metrics.novel_12`。`training_time_seconds` 是从优化循环开始到该里程碑的累计 wall time，排除了完整评估和高斯 PLY 保存所消耗的时间。场景初始化和预处理耗时也不计入训练耗时。

对于已经完成的旧实验，脚本不会重新训练，也不会覆盖已有 18 路指标和 `training_time_<N>.txt`；它只会从已保存的 `renders/gt` 中读取 `00.png`--`11.png`，在指定设备上补算 12 路新视角指标。新实验会在训练内即时报告两组指标，runner 随后仍以保存的 PNG 统一落盘 `novel_12`，使新旧实验采用相同的 12 路统计输入。

为了避免大量无用 I/O，只在 10,000 次迭代保存最终：

```text
point_cloud/iteration_10000/point_cloud.ply
```

不保存样本内 checkpoint。

## 断点续跑

启动脚本会逐场景检查：

- 1k、5k、10k 的 JSON/文本指标和训练耗时均存在且有效；
- 三个里程碑各有且仅有 18 对非空的 `00.png` 到 `17.png`；
- 累计训练耗时随里程碑单调不减；
- 最终 10k PLY 存在且非空。

满足全部条件的场景会快速打印 `[SKIP]` 并跳过。任一条件不满足时，场景会打印 `[RESTART]` 并从头训练；启动前会精确作废旧的里程碑记录与最终 PLY，防止不同训练轨迹的产物被混合判定为完成。无关文件不会被清理。

## 自动汇总

150 个场景全部完整后，脚本生成：

```text
<run_root>/center150_metrics_summary.json
<run_root>/center150_metrics_summary.txt
```

汇总包含每个样本的三个里程碑记录，以及 1k、5k、10k 时 18 路、12 路新视角 PSNR/SSIM/LPIPS 和累计纯训练耗时的 150 场景等权平均值。旧的 `averages` 字段继续表示 18 路结果，新增的 `view_subsets` 同时给出 `all_18` 与 `novel_12`。

实验根目录还会保存 `experiment_config.json`。若对同一 `--run_root` 使用不同数据路径、分辨率、迭代里程碑或 SPS 配置，脚本会拒绝混合结果，并提示换用新输出目录；对于没有该配置文件的既有非空目录也不会自动接管，避免覆盖旧版实验。

## 启动方式

默认参数已经是 Center150 正式协议：

```bash
python scripts/run_omniscene.py
```

默认输出目录为：

```text
output/omniscene_center150_112x200_iter10000_eval1000-5000-10000/
```

所有主要参数都可以覆盖。例如：

```bash
python scripts/run_omniscene.py \
  --reso 224x400 \
  --iterations 12000 \
  --eval_iterations 1000 5000 12000 \
  --run_root output/omniscene_center150_custom
```

最后一个评估里程碑必须等于总迭代数。若确需关闭 LoopSparseGS 默认启用的 sparse-friendly sampling，可额外传入 `--disable_sps`。

若只想为已经完整训练的场景补算指标并确保绝不启动训练，可使用：

```bash
CUDA_VISIBLE_DEVICES='' python scripts/run_omniscene.py \
  --metrics_only \
  --metric_device cpu
```
