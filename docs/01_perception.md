# 01 · 感知

## 1. 概述

EvoBot 用「Agent + 技能库」完成机器人操作任务：Agent 决定做什么、按什么顺序做，技能负责具体的动作。抓取类技能在动手之前必须先回答三个问题——物体在哪里、从哪个方向夹、夹多宽。感知层就是回答这些问题的模块。

感知层向技能提供两类结果：

- **物体位姿**：物体在世界坐标系中的位置和大致尺寸。
- **抓取候选**：一组可尝试的抓取位姿，每个候选包含夹爪位置、姿态、开口宽度和置信度。

感知以技能的形式暴露（`darwin/skills/base.py` 里的 `Skill` / `SkillSpec`，`kind=PERCEPTION`），可以单独调用，也可以组合成「检测 → 分割 → 抓取」的链路。它被设计成可替换的一层：核心包不依赖任何具体的感知模型，调用方可以整体换成自己的实现。

## 2. 两条感知路径

仓库里存在两条并行的感知路径，适用的环境不同，不要混用。

### 2.1 仿真真值路径（LIBERO / MuJoCo）

当环境适配器直接暴露 MuJoCo 模型（`mj_model` / `mj_data`）时，物体的位置、几何、关节状态都是已知的真值。这条路不渲染图像、不跑检测网络，直接从模型里取物体几何并采样成点云，再交给抓取网络。

适用场景：LIBERO 基准跑分、需要精确位姿的实验、需要对比不同抓取策略的离线复现。LIBERO 技能库里的抓取链（`darwin/agents/libero_skills.py` 的 `grasp()`）走的就是这一条。

### 2.2 模型推理路径（真实 / 通用场景）

当只有相机图像、或者适配器只提供 `obs["vision"]` 图像快照（例如 RoboDojo 适配器）时，感知必须从图像出发：

```
RGB 图像 → YOLO 检测 → SAM 分割 → 深度 / 点云 → GraspNet 抓取
```

适用场景：真实机器人、只有 RGB-D 相机的仿真、以及没有真值位姿的第三方环境。

仓库里另有一套独立的 RGB-D 决策链（`darwin/perception/rgbd_fusion.py` → `scene_graph.py` → `reachability.py`），做法是把多路相机的深度图反投影成世界系点云、构建场景图、再做机械臂可达性预判。它面向多相机真实场景，与上面的技能路径是两套实现。

### 2.3 怎么选

选择由环境提供的能力决定，而不是由任务配置显式切换：

- 适配器能拿到 MuJoCo 模型 → 走真值路径。
- 只有图像或深度流 → 走模型推理路径。
- 技能参数也体现同样的区分：`GraspPoseSkill.execute()` 接受 `point_cloud`（真值路径）或 `depth` + `intrinsic`（相机路径）。

## 3. 感知技能一览

| 技能 | 文件 | 输入 | 输出 |
|------|------|------|------|
| `detect_objects` | `darwin/skills/perception/detect.py` 的 `DetectObjectsSkill` | RGB 图像（或 `env` 自动渲染） | 物体 bbox、类别、置信度、3D 中心 |
| `segment_object` | `darwin/skills/perception/segment.py` 的 `SegmentObjectSkill` | RGB 图像 + bbox 或提示点 | 二值 mask、可抓取区域中心 |
| `estimate_depth` | `darwin/skills/perception/detect.py` 的 `EstimateDepthSkill` | `env`（仿真）或深度相机 | 深度图（米） |
| `grasp_pose` | `darwin/skills/perception/grasp.py` 的 `GraspPoseSkill` | 点云，或深度图 + 内参 | 抓取位姿：位置、旋转、开口宽度、置信度 |

几点说明：

- **检测**用 YOLO26n（`yolo26n.pt`），默认置信度阈值 0.25。仿真场景里的物体大多不是 COCO 类别，YOLO 可能一个都测不到；这时 `_gt_detect()` 会退回到从模型里枚举物体，效果等价于真值检测。
- **分割**用 MobileSAM，通过 `ultralytics.SAM` 加载，接口与 `ultralytics` 保持一致。
- **深度**在仿真里直接用 MuJoCo 渲染的真值深度，真实环境接深度相机。之所以不让 YOLO 顺带做深度，是因为仿真里深度真值本来就有，而网络估计出来的深度误差会一路传到抓取位姿里。
- **抓取**先尝试 GraspNet 模型；模型不可用或对某个物体产出 0 个候选时，退回几何方法（对点云做 PCA，按主轴生成候选），并标记 `source="geometry_fallback"`。

## 4. 抓取候选是怎么枚举出来的

以 LIBERO 真值路径为例，候选的产出分三步。

### 4.1 第一步：点云采样

抓取网络需要物体的表面点云。`sample_object_point_cloud()` 从仿真几何采样单个物体，`sample_scene_point_cloud()` 采样**除机器人外的整个场景**。

LIBERO 的抓取链用的是全场景点云。只喂物体点云时，网络看不见桌面和柜架的支撑几何，容易综合出与支撑面互相干涉的抓取；把场景点一起喂进去，下游再做站点碰撞过滤，才能筛出真正抓在目标物上的候选。

采样有两个细节值得知道：

- **mesh 用真值顶点**：网格体的 `geom_size` 只是一层包围盒，不能代表真实形状，所以 mesh 类型直接从模型的顶点数组里取点，而不是在包围盒里撒点。
- **跳过纯视觉图元**：碰撞开关全关（`contype=0` 且 `conaffinity=0`）的图元，其尺寸是包围盒外壳，直接采样会把容器内腔填满，污染点云。这类图元跳过；但视觉 **mesh** 要保留，因为有些物体的真实形状只挂在视觉网格上。

### 4.2 第二步：候选生成

点云先做一次帧对齐，再送进抓取网络。网络是在相机坐标系下训练的（相机系里“向场景内”的 +z 就是物理接近方向），而仿真点云是世界系，所以送入前用 `F = diag(1, -1, -1)` 变换，预测后再变回世界系（见 `darwin/agents/libero_skills.py` 的 `_grasp_backend_candidates()`）。

`load_backend()`（`darwin/skills/perception/grasp_backends.py`）按环境变量 `DARWIN_GRASP_BACKEND` 选择后端，默认首选 `gsnet`，失败时按顺序试其它后端。某个后端加载失败会带着原因打印出来，最后用的不是首选时会明确记录，避免出现“以为在用 A、实际全程在用 B”的错觉。所有后端都不可用时返回 `None`，调用方得到空候选列表并转用下面的分析解。

网络输出的候选按置信度排序取 top-k。与此同时，感知还会根据物体几何直接算出一批**分析解**——`rim_pinch`（沿容器口沿深捏）和 `side_grip`（从侧面外夹）。两类候选**交替产出**，而不是先穷举一类再另一类：网络候选里常常夹着成串不可用的解，逐一穷举会耗光尝试预算，交替产出能保证任一侧的好解在前几次尝试里就能轮到。

### 4.3 第三步：筛选

候选生成之后要过一道物理和几何筛选，只有全部通过的候选才会被真正执行：

- **开口可达**：夹持半宽不能超过夹爪能张开的范围。
- **高度约束**：抓取点要落在物体的中上部，避免指尖探到支撑面把物体撬翻。
- **靠近目标**：候选位置必须离目标物体足够近；全场景候选常会落在旁边的物体上，这一步把它们滤掉。
- **站点可行**：对候选位姿求逆解，要求机械臂没有新的自接触，除目标物和其支撑面外不碰到别的物体（`filter_site_config()`）。
- **接近段无碰撞**：从预抓取位到抓取位之间做直线插值并逐点检查碰撞（`_approach_clear()`）。
- **闭爪能夹住**：用仿真预演一次闭爪，排除“能到达但夹空”的候选（`_closure_depth()`）。

候选按签名去重，失败过的候选在后续轮次里跳过。

## 5. 关键参数

以下数值取自代码默认值或 LIBERO 抓取链的实际调用，覆盖方式见括号。

| 参数 | 值 | 说明 |
|------|-----|------|
| `sample_object_point_cloud` 默认采样数 | 2048 | 单物体表面点 |
| LIBERO 抓取链物体点云 | 20000 | `graspnet_candidates()` 实际调用值 |
| 全场景点云 | 20000 | `sample_scene_point_cloud()`，每个 geom 固定采 64 点 |
| GraspPoseSkill 内部采样数 | 20000 | 超出无放回抽样，不足有放回抽样 |
| GSNet 输入重采样 | 15000 | `grasp_backends.py` 的 `_resample()` |
| GraspNet 输入重采样 | 20000 | 同上 |
| 全场景候选 top-k | 64 | 落目标的候选为零时翻页到 256 重取一次 |
| 单物体候选 top-k | 32 | `graspnet_candidates()` |
| `GraspPoseSkill` 默认 top-k | 1 | 只返回最优解 |
| YOLO 置信度阈值 | 0.25 | `DetectObjectsSkill.execute()` 的 `conf_threshold` |
| 分析解尝试上限 | 12 | `ANALYTIC_TRIES` |
| 几何回退开口宽度 | `clip(extents[0] * 1.2, 0.01, 0.12)` | 单位米 |
| 几何回退置信度 | 0.5 | 固定值 |
| 置信度分档 | <1.0 低 / <1.3 中 / ≥1.3 高 | `score_band_of()` |

## 6. 模型权重的准备

所有权重的加载都走同一个入口 `ensure_weight()`（`darwin/utils/download.py`），查找顺序是：

1. 环境变量指定的本地路径；
2. 项目公共目录下的同名文件；
3. 权重缓存目录，可用 `DARWIN_WEIGHTS_CACHE` 指定，默认是 `~/.cache/darwin/weights`；
4. 从镜像下载（Gitee / HuggingFace 镜像，失败再试原始地址）。

各模型对应的文件名、环境变量和默认位置：

| 模型 | 文件 | 覆盖变量 |
|------|------|----------|
| YOLO 检测 | `yolo26n.pt` | `DARWIN_YOLO_DET_WEIGHTS` |
| MobileSAM 分割 | `mobile_sam.pt` | `DARWIN_SAM_WEIGHTS` |
| GraspNet 权重 | `checkpoint-rs.tar` | `DARWIN_GRASPNET_CKPT` |
| GraspNet 代码仓库 | `graspnet-baseline` 目录 | `DARWIN_GRASPNET_ROOT` |
| GSNet 权重与代码 | `gsnet_graspness/` 目录 | `DARWIN_GRASP_MODELS_ROOT` |

YOLO 和 SAM 的权重可以直接从镜像下载。GraspNet 的 `checkpoint-rs.tar` 没有公开的下载链接，需要手动获取后放进权重缓存目录，或用 `DARWIN_GRASPNET_CKPT` 指向已有文件。GSNet 的权重在其模型根目录下的固定相对路径（`gsnet_graspness/checkpoints/checkp_realsense.tar`）。

推理设备统一用 `cuda:0`，无 GPU 时自动退到 CPU。

## 7. 结果稳定性

抓取候选里有两处随机性，代码分别做了固定：

- 从 mesh 顶点下采样时使用固定种子（`RandomState(42)`），保证同一场景采样出的点云一致。
- 送进抓取网络前的重采样使用固定种子（`_resample(pc, n, seed=0)`）。

在这之上，外层执行器会按尝试序号为整个流程设置一次随机种子，因此同一次尝试内可复现，不同尝试之间仍有采样差异，保留重试所需的多样性。

需要注意一点：即使输入点云完全固定，抓取网络在 GPU 上的前向计算仍会带来细微差别——现代深度学习框架的某些归约算子在 GPU 上默认是非确定性的。因此同一输入不同次运行可能得到略有差异的候选分数与排序。这是模型推理本身的特性，与感知层的采样逻辑无关。
