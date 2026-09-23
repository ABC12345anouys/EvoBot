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
