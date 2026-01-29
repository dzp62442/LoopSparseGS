# OmniScene 数据集实验文档（LoopSparseGS / comp_svfgs 规划稿）

## 1. 背景与目标
本项目（LoopSparseGS）是**逐场景优化**的高斯重建方法，与 depthsplat 的**前馈式**重建流程不同：
- **depthsplat（前馈）**：一次前向推理得到输出，数据集在 `__getitem__` 中直接返回 `context/target`，训练由 batch/step 驱动。
- **LoopSparseGS（优化）**：每个场景需要**迭代优化**（多步梯度更新），训练过程内需要完整的相机列表（train/test）以及初始化点云（可选）。

目标：在 `comp_svfgs` 分支下，为 OmniScene 数据集建立**逐场景优化**的实验流程，实现“数据预处理→加载→训练→渲染→评估”单阶段完成，并与我的方法对比。

## 2. 参考资料与现有实现理解
### 2.1 depthsplat 的 OmniScene 方案（前馈）
- 入口文档：`depthsplat/docs/OmniScene数据集实验文档.md`。
- 数据集实现：`depthsplat/src/dataset/dataset_omniscene.py` + `depthsplat/src/dataset/utils_omniscene.py`。
- 关键要点：
  - `bins_*_3.2m.json` 列表控制 `train/val/test/demo`。
  - `__getitem__` 输出 `context`（6 视角）与 `target`（18 视角，包含输入视角）。
  - `load_conditions` 完成路径替换（`samples → samples_small`、`samples_param_small`）、图像 resize、内参缩放、掩码加载。

### 2.2 SVF-GS 的 OmniScene 方案（含深度/置信度）
- 参考实现：`SVF-GS/data/omniscene_dataset.py` + `SVF-GS/data/transforms/loading.py`。
- 关键要点：
  - `load_conditions` 额外读取 Metric3D 深度与置信度（`*_dptm_small/*_dpt.npy` + `*_conf.npy`）。
  - 置信度可用于筛选有效深度（>0.3）。
  - 深度与置信度可用于构造初始化点云或辅助监督。

## 3. 与前馈重建的差异（必须明确）
1) **数据组织方式不同**
- depthsplat：一次样本包含 `context/target`。
- LoopSparseGS：需要**拆分为 train/test 相机列表**，由训练脚本在每个场景内迭代优化。

2) **点云初始化需求**
- LoopSparseGS 期望读取 COLMAP 点云或自定义点云作为高斯初始化；需要决定 OmniScene 版本的初始化方式。

3) **评估流程不同**
- depthsplat：测试时模型直接输出。
- LoopSparseGS：训练后用渲染脚本对 test 相机生成图像，再计算指标。

## 4. OmniScene 数据在 LoopSparseGS 的预处理设计
### 4.1 场景划分与模式
- **val 模式默认启用**，仅 10 个 bin；每个 bin 视为一个**独立场景**（逐场景优化）。
- `train/test/demo` 仅为兼容性保留，不是主要模式。

### 4.2 预处理输出目录规划（写入本项目 `output/`）
```
output/
  omniscene_preproc/
    01_<bin_token>/
      images/
        train/      # 6 张输入视角
        test/       # 18 张输出视角
      cams/
        train.json  # 每张图的内外参
        test.json
      depth/
        train/      # 训练用 depth_8
        conf/       # 置信度（可选）
      points/
        init_points.ply  # 初始化点云（由 Metric3D depth 反投影）
```
命名采用 `01_<bin_token>` 前缀，确保排序一致。

### 4.3 预处理核心流程（计划）
1. 读取 `bin_infos_3.2m/{token}.pkl`，提取 6 个相机的 key-frame 作为 **train**。
2. 从同一 bin 中选取 index `[1,2]` 的帧作为 **test**（共 12 张），再把 train 视角并入 test，使 test 共 18 张（与 depthsplat 保持一致）。
3. 使用 depthsplat 的 `load_conditions`（路径转换逻辑**原样复用**）：
   - 图像从 `samples/sweeps` → `samples_small/sweeps_small`；
   - 内参随 resize 调整；
   - 本项目暂不加载动态掩码、相对深度。
4. 若需要初始化点云：
   - 参考 SVF-GS 的 `load_conditions` 加载 Metric3D 深度与置信度；
   - 置信度 > 0.3 的像素参与反投影；
   - 反投影到世界坐标后拼接为点云，保存 `init_points.ply`。
5. 生成训练深度图：
   - 将 Metric3D 深度（float）转换为 uint16 存为 `depth_8`，路径与训练图像一一对应。

> 注意：LoopSparseGS 的深度监督默认开启，需要确保训练图像都有 depth_8。

## 5. 数据加载设计（comp_svfgs）
### 5.1 新增数据加载文件
- 位置：`comp_svfgs/dataset_omniscene.py`
- 职责：
  - 读取预处理后的 `train/test.json`，生成 CameraInfo 列表；
  - 返回 train/test 两组相机列表供 Scene 初始化；
  - 默认 `val` 模式，支持 `train/test/demo` 作为兼容。

### 5.2 与 LoopSparseGS 主流程衔接
- 在 `scene/dataset_readers.py` 注册新数据集类型（例如 `OmniScene`）。
- `Scene` 识别 `omniscene.json` 或自定义标志文件后，调用 OmniScene reader。
- OmniScene reader 提供：
  - `train_cameras`: 6 张输入视角；
  - `test_cameras`: 18 张输出视角；
  - `point_cloud`: 读取 `init_points.ply`（若存在），否则 fallback 为随机点云；
  - `nerf_normalization`: 通过相机中心计算尺度（无需 poses_bounds.npy）。

## 6. 训练 / 渲染 / 评估流程设计
### 6.1 每场景流程
1. 若 `output/omniscene_preproc/01_<token>/` 不存在 → 预处理并保存。
2. 运行 4 轮训练（LoopSparseGS 结构保持）：
   - 第 0 轮：使用原始 train 数据；
   - 第 1~3 轮：是否启用 pseudo view 取决于后续实现。
3. 每轮训练结束后运行渲染并计算指标。

### 6.2 评估方式
- 使用 `render.py` 对 test 相机渲染输出。
- 使用 `metrics.py` 读取 `renders` 与 `gt` 计算 PSNR/SSIM/LPIPS。

## 7. 运行脚本设计
- 在 `scripts/` 下新增 `run_omniscene_comp_svfgs.py`：
  - 遍历 val 的 10 个 bin；
  - 每个场景先检查预处理是否完成；
  - 依次执行训练、渲染、评估；
  - 统一记录日志与结果目录。

脚本将替代 bash 方案，便于整合预处理步骤。

**运行示例**：
```
python scripts/run_omniscene.py \
  --stage val \
  --reso 112x200
```

## 8. 参数配置（已确认）
- 图像分辨率：默认 **112×200**，可选 **224×400**。
- `-r` 参数：统一设为 `1`，保持预处理分辨率不再二次缩放。
- 训练视角数：`--train_sub = 6`（全部输入视角）。
- 迭代次数：总迭代 **10k**，分 4 轮训练，**每轮 2.5k**。

## 9. Loop 机制与 OmniScene 的适配说明（已确认）
### 9.1 原 loop 为何要重新三角化
LoopSparseGS 的 loop 设计主要服务于 **PGI（逐步点云初始化）**：
- 先用稀疏视角训练得到初始高斯；
- 再渲染伪视角图像并喂给 COLMAP；
- 通过 COLMAP 重新三角化生成更密的点云与深度（`points3D.ply` + `depth_8`），为下一轮训练提供更好的几何初始化与监督。

### 9.2 OmniScene 的替代策略
OmniScene 自带**高质量稠密深度**（Metric3D），优于稀疏 COLMAP 结果，因此：
- **不使用 COLMAP 重新三角化**；
- **直接使用 Metric3D 深度**生成初始化点云与训练深度监督；
- loop 轮次仍保留（保持 4 轮），但每轮仅执行训练，不再执行 COLMAP 的 densify；
- `use_pseudo_view` 默认关闭，避免缺少伪视角深度带来的训练不稳定（后续若需要可再扩展伪视角深度生成）。

## 10. 待确认问题
目前**无待确认问题**。已确认：
- 总迭代 10k、4 轮、每轮 2.5k；
- 深度监督保持本项目原生默认配置；
- 初始化点云必须由 Metric3D 深度生成；
- loop 不再走 COLMAP 三角化，默认关闭 pseudo view。

---

## 11. 下一步
待你审阅本文件后，我将根据确认方案执行：
- 创建 `comp_svfgs/` 代码与数据加载实现；
- 实现 OmniScene 预处理与 point cloud 生成；
- 新增运行脚本与训练/评估流程整合。
