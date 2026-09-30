# 07 · 感知技能设计

> 对应代码：`darwin/skills/perception/detect.py`、`darwin/skills/perception/segment.py`、`darwin/skills/perception/grasp.py`

## 1. 设计目标

感知技能为 Agent 提供**物体检测、分割、抓取位姿估计**能力。设计目标：

1. **模块化**：检测 / 分割 / 抓取各自独立技能，可单独使用或组合。
2. **双模式回退**：优先用真实模型（YOLO / SAM / GraspNet），权重不可用时回退几何/仿真方法，保证链路不中断。
3. **仿真友好**：点云直接从仿真物体几何采样，不依赖深度相机输入。
4. **可替换**：感知由调用方注入，`ragbot` 核心包不依赖任何感知模型。

## 2. 目标检测：detect.py

### 2.1 DetectObjectsSkill

基于 **YOLO**（ultralytics）的目标检测：

- 输入：RGB 图像
- 输出：物体 bbox / 类别 / 置信度
- 模型权重：通过 `ensure_weight` 从 Gitee 镜像下载（GitHub 为 fallback）

```python
class DetectObjectsSkill(Skill):
    spec = SkillSpec(name="detect_objects", kind=SkillKind.PERCEPTION, ...)
    def execute(self, env, image=None, **kwargs):
        # image 为 None 时从 env 渲染获取
        results = model(image)
        return {"success": True, "objects": [...]}
```

### 2.2 EstimateDepthSkill

深度估计技能。**关键设计决策**：

> YOLO 仅用于目标检测，不用于深度估计。深度信息来自系统外部真值（仿真中直接读取物体位姿），而非相机输入。

这样做的原因：
- 仿真中深度真值可得，无需训练深度估计网络
- 避免深度估计误差累积到抓取位姿中
- 降低感知链路复杂度

## 3. 物体分割：segment.py

### SegmentObjectSkill

基于 **SAM**（Segment Anything）的物体分割：

- 输入：RGB 图像 + 可选提示点/bbox
- 输出：物体 mask
- 实现：使用 `ultralytics.SAM`（需 ultralytics 8.4.x+）

**关键技术决策**：

> `segment_anything` 包与 MobileSAM（TinyViT 架构）不兼容，改用 `ultralytics.SAM`。

`ultralytics.SAM` 提供统一的 SAM 接口，避免了原版 `segment_anything` 的架构依赖问题。

## 4. 抓取位姿估计：grasp.py

### GraspPoseSkill

基于 **GraspNet** 的抓取位姿估计，是感知技能中最核心的一个。

### 4.1 双模式架构

```python
@lru_cache(maxsize=1)
def _load_graspnet():
    if not _graspnet_available():
        return None
    try:
        # 加载真实模型 checkpoint-rs.tar
        net = GraspNet(...).to(device)
        net.load_state_dict(ckpt["model_state_dict"])
        net.eval()
        return net, pred_decode, device
    except Exception as e:
        print(f"[grasp] GraspNet 加载失败，使用几何 fallback: {e}")
        return None
```

| 模式 | 条件 | 实现 |
|------|------|------|
| **真实模型** | `checkpoint-rs.tar` 存在且 GraspNet repo 可导入 | GraspNet 模型预测抓取位姿 |
| **几何回退** | 权重不可用 | 从点云 PCA 估计抓取位姿 |

### 4.2 点云来源

**关键设计决策**：

> GraspNet 的点云输入直接从仿真物体几何采样，不依赖深度相机。

- 仿真中可直接获取物体 mesh / 几何信息
- 从物体表面采样点云 → 输入 GraspNet
- 避免了深度相机 → 点云的转换误差

### 4.3 抓取候选生成

`grasp_candidates_from_env(env, body, top_k)`：

1. 从仿真环境获取物体几何
2. 采样点云
3. GraspNet 预测 top-k 抓取位姿 `{position, rotation_matrix, width, score}`
4. 或几何 fallback：PCA 主轴方向生成抓取候选

### 4.4 辅助函数

```python
def object_features(env, body) -> dict:
    # 返回 {shape, size, center}

def rel_offset_of(grasp_pt, center, size) -> list:
    # (grasp_pt - center) / half_size

def abs_pos_of(rel_offset, center, size) -> list:
    # center + rel_offset * half_size

def score_band_of(score) -> str:
    # GraspNet 置信度分 band
```

这些函数是感知与 RAG 的**接口层**：感知输出绝对坐标，RAG 存储归一化偏移，`rel_offset_of` / `abs_pos_of` 负责转换。

## 5. 感知与 RAG 的接口

```
感知技能输出绝对坐标
        ↓ rel_offset_of
RAG 存储归一化偏移 rel_offset
        ↓ abs_pos_of
执行时重建绝对抓取点
```

`EpisodeRunner._build_candidates` 中：
```python
for c in candidates:
    off = rel_offset_of(c["position"], center, feat["size"])  # 绝对→归一化
    band = score_band_of(c["score"])
    if rag.is_failed(task, feat, off):    # 用归一化偏移查失败记忆
        continue
```

`_inject_memory` 中：
```python
rec = rag.best_success(task, feat)       # 取回归一化偏移
pos = abs_pos_of(rec["rel_offset"], center, feat["size"])  # 归一化→绝对
```

## 6. 模型权重管理

`darwin/utils/download.py` 的 `ensure_weight` 统一管理模型权重下载：

- **主源**：Gitee 镜像（国内访问快）
- **fallback**：GitHub
- 常用 ultralytics 模型的 Gitee 镜像配置在 `GITEE_MIRRORS` 字典中

这样确保在国内网络环境下也能稳定下载权重。

## 7. 感知技能的可替换性

感知技能完全独立于 `ragbot` 核心包：
- `ragbot` 只负责记忆，不感知、不控制
- 调用方可接入任意感知栈（YOLO / SAM / GraspNet / 自定义）
- 只需按 `{shape, size, center, rel_offset, score_band}` 格式提供特征即可使用 RAG

这种设计让 EvoBot 的记忆能力可被**任意机器人系统复用**，而不绑定特定的感知方案。

---

## 8. LIBERO / MuJoCo 侧的感知路径

上面 §2–§6 描述的是**技能路径**（`darwin/skills/perception/`）。仓库里还存在**第二条感知路径**，两者不要混淆：

| 路径 | 代码 | 点云来源 | 接线现状 |
|---|---|---|---|
| 技能路径 | `darwin/skills/perception/detect.py` / `segment.py` / `grasp.py` / `grasp_backends.py` | **仿真真值**：从物体 mesh/geom 表面采样 | LIBERO 侧实际在用（`libero_skills.py:1439-1440,1466-1468`）|
| RGB-D 决策链 | `darwin/perception/rgbd_fusion.py` → `scene_graph.py` → `reachability.py` | **深度图反投影**：3 路相机融合 | 未见调用点（除包内 `__init__.py` 导出）；RoboDojo 适配器只存了 `obs["vision"]` 快照 `envs/robodojo_adapter.py:71,107` |

### 8.1 真值点云采样

| 函数 | 契约 | 实际调用点数 |
|---|---|---|
| `sample_object_point_cloud(env, body_name, n_points=2048)` | 世界系表面点 `(n,3) float32`；按 geom 表面积/类型采样，mesh geom 用**真值顶点**而非 AABB | `20000`（`libero_skills.py:1468`）、`4096`（`:1949`）、`2048`/`1024`（`policies/rules.py:79,86`）|
| `sample_scene_point_cloud(env, n_points=20000)` | 全场景（排除 `robot*`），每 geom 固定 `n_g=64` | `20000` |
| `GraspPoseSkill.execute(...)` | 内部自采 `num_point=20000`（超则无放回、不足则有放回）| — |

`skills/perception/grasp.py:213,315,362-375`、`:162-166`

### 8.2 真值采样的三个陷阱（都有实测依据）

1. **纯视觉 geom 的尺寸是 AABB 壳**：`contype=0 且 conaffinity=0` 的 geom，`geom_size` 不代表真实形状；旧实现 `_sample_box` 会把容器空腔填满 → 污染 GraspNet → **`F=0` 夹空**的根因。`grasp.py:277-280`
2. **但不能整体跳过视觉 geom**：部分物体（`spatial:1/2` 实证）**只有视觉 mesh 携带真实形状**。`grasp.py:281-284`
3. **mesh 的 `geom_size` 同样不可信** → 必须用真值顶点 `m.mesh_vert`。`grasp.py:362-364`

### 8.3 为什么 mesh 顶点下采样用固定种子 `RandomState(42)`

不是随手写的：`spatial:9` 实证同一场景两次运行的投票方向分别是 `[-0.22,-0.98]` 与 `[-0.59,-0.80]`，**一个夹住、一个夹空**。故把这一层固定为种子 42，把"逐 attempt 的探索"交给显式的抖动配置（`cfg.jit`）。`grasp.py:376-381`

## 9. 抓取后端链与已知陷阱

### 9.1 注册、顺序与回退

```python
_BACKENDS = {"gsnet": GSNetBackend, "graspnet": GraspNetBackend}
pref = os.environ.get("DARWIN_GRASP_BACKEND", "gsnet")     # 默认首选 gsnet
order = [pref] + [其它后端]                                 # gsnet → graspnet
```

- 逐个构造，失败把 `f"{name}: {e}"` 收进 `errs` 继续下一个；`grasp_backends.py:195-216`
- **回退必须可见**：若最终用的不是首选后端，强制打印警告 + 失败原因。代码注释记录了事故原因——"首选后端静默失败是隐蔽陷阱（曾致『以为在用 gsnet，实际全程 graspnet』）"。`grasp_backends.py:206-209`
- 全部失败 → 返回 `None`，调用方返回空候选列表（不抛错）。`grasp_backends.py:215-216`、`libero_skills.py:1440-1442`

### 9.2 后端加载的环境 hack（为了让老代码在现环境跑起来）

`grasp_backends.py` 里有一组围绕"模型家族导入"的兼容层，排障时应知道它们存在：

| hack | 位置 |
|---|---|
| `_family_import` 临时清空 `sys.path`/`sys.modules`（按 `_FAMILY_TOPS` 名单）| `:35-65` |
| 伪造 `sys.argv = ["x","--dataset_root","/tmp","--camera","realsense"]` | `:44-48` |
| 给 `open3d` 塞空 stub | `:26-32` |
| monkeypatch MinkowskiEngine 的 `sparse_quantize`（整数坐标往返）| `:67-95` |
| `_ensure_knn_shim` 往 `/tmp/gsnet_shims/knn` **写文件**（生成用 `torch.cdist` 实现的 knn）| `:97-108` |

### 9.3 第二套回退（技能内部）与分析解交错

- `GraspPoseSkill` 先试 GraspNet；无候选时**不崩**，打印后回退几何 PCA，候选头写 `source="geometry_fallback"`、`score=0.5`、`width=clip(extents[0]*1.2, 0.01, 0.12)`；`grasp.py:35,88-93,141-146`
- LIBERO 抓取链里，**网络候选与分析解（`rim_pinch` ↔ `side_grip`）轮询交错**产出，学习源先出；`libero_skills.py:3636-3650`
- 点云送入网络前做一次帧对齐基变换 `F = diag(1,-1,-1)`（`GRASPNET_FRAME`）；`libero_skills.py:243,1443-1447`

## 10. 权重与设备

**全部硬编码路径（均可用环境变量覆盖，但默认值指向本机布局）：**

| 常量 | 默认值 | 覆盖变量 |
|---|---|---|
| `GRASPNET_ROOT` | `/home/lifd/Public/graspnet_repo/graspnet-baseline` | 同名 env |
| `DEFAULT_CHECKPOINT` | `/home/lifd/Public/checkpoint-rs.tar` | 同名 env |
| `MODELS_ROOT` | `/home/lifd/Public/grasp_models` | 同名 env |
| `GSNET_CHECKPOINT` | `<MODELS_ROOT>/gsnet_graspness/checkpoints/checkp_realsense.tar` | — |
| knn shim 目录 | `/tmp/gsnet_shims` | — |
| SAM 权重 | `mobile_sam.pt` | `DARWIN_SAM_WEIGHTS` |
| YOLO 权重 | `yolo26n.pt` | `DARWIN_YOLO_DET_WEIGHTS` |

`ensure_weight` 的查找顺序：env 指定的 local_path → `/home/lifd/Public/<name>` → 权重缓存（`DARWIN_WEIGHTS_CACHE`，默认 `~/.cache/darwin/weights`）→ Gitee/HuggingFace 镜像下载。`utils/download.py:22-24,113-137`

> ⚠️ **硬限制**：`checkpoint-rs.tar`（GraspNet）**没有公开直链**——注释写明"从百度网盘下载，已存于 `/home/lifd/Public/checkpoint-rs.tar`"，镜像映射值为 `None`；下载失败会抛 `FileNotFoundError`。也就是说**换台机器无法自动获得该权重**。`utils/download.py:34-35,67-68`

**推理设备**：三处同一写法 `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`。GSNet 先 `map_location="cpu"` 再 `.to(device)`；GraspNet 直接 `map_location=device`。`grasp.py:50`、`grasp_backends.py:141-142,145-147,178,180`

## 11. 随机源全景与确定性现状（2026-09-30 实测）

| 随机源 | 是否播种 | 位置 |
|---|---|---|
| `sample_object_point_cloud` / `sample_scene_point_cloud` 末尾裁剪 | **曾未播种**（全局 `np.random.choice`）| `grasp.py:310,357` |
| 图元采样 `_sample_box/_sphere/_cylinder`（全局 `np.random.uniform`）| 同上（依赖外层播种）| `grasp.py:387,408,418` |
| mesh 顶点下采样 | ✅ 固定 `RandomState(42)` | `grasp.py:380` |
| 后端输入重采样 `_resample(pc,n,seed=0)` | ✅ 独立 `RandomState(0)` | `grasp_backends.py:110-118`（GSNet→15000 `:151`；GraspNet→20000 `:187`）|
| 外层统一播种 | ✅ `np.random.seed(attempt)` | `libero_runner.py:243` |

**这是一个真实的 bug**：全局 RNG 从未播种 → 每次喂给 GraspNet 的点都不同 → 候选抓取时有时无（同一任务同一初始状态，一次 0 个候选、一次多个，整条路径分叉）。现由 runner 按 attempt 序号播种。

**但仍然不完全可复现**：播种后输入已确定，输出仍变 → 剩余方差在 **GraspNet 的 GPU 前向**（`device=cuda:0`，CUDA 的 scatter/atomicAdd 类算子默认非确定）。已排除挂钟超时、CPU 线程调度、RRT 种子；关 CUDA 复测未做（`CUDA_VISIBLE_DEVICES=""` 会让 robosuite 的 EGL 渲染解析空设备列表而崩）。详见 [12 调试纪要](12_libero_debug_2026-09-30.md)。
