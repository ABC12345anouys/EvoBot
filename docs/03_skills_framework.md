# 03 · 技能框架设计

> 对应代码：`darwin/skills/base.py`、`darwin/skills/__init__.py`、`darwin/skills/primitives/`、`darwin/skills/perception/`、`darwin/skills/forged/`

## 1. 设计目标

技能框架是 EvoBot 的**能力层**，将感知、控制、运动等原子能力封装为可被 Agent / LLM 直接调用的 Skill。设计目标：

1. **三层技能体系**：感知技能 → 控制/运动原语 → 策略技能（forged），分层解耦。
2. **元数据驱动**：每个技能有 `SkillSpec`（名称/描述/参数/置信度/证据），可导出为 OpenAI function calling 格式供 LLM 调用。
3. **可自动生成**：成功轨迹通过 `skill_forge` 自动沉淀为 forged 策略技能，无需手写。
4. **去分发**：Agent 直接持有 `Dict[str, Skill]`，无中间 registry。

## 2. Skill 基类与元数据

### 2.1 SkillSpec

```python
@dataclass
class SkillSpec:
    name: str
    description: str
    kind: SkillKind          # primitive / perception / strategy / failure / infra
    confidence: Confidence   # single-shot / probable / verified
    applies_when: str        # 触发条件（自然语言）
    entry_file: Optional[str]
    parameters: Dict[str, Any]   # JSON Schema 风格的参数定义
    required: List[str]
    evidence: Evidence
    supersedes: Optional[str]    # 替代的旧技能名
```

`to_tool_spec()` 将 `SkillSpec` 导出为 OpenAI function calling 格式，LLM 可直接据此选择技能并传参。

### 2.2 SkillKind

| kind | 含义 | 示例 |
|------|------|------|
| `primitive` | 原子操作技能 | move_above, descend, close_gripper, servo_align |
| `perception` | 感知技能 | detect_objects, segment_object, grasp_pose |
| `strategy` | 策略技能（多步组合） | forged pick, pickplace |
| `failure` | 避坑条目 | RAG 失败记忆 |
| `infra` | 基础设施 | grasp_stats 统计表 |

### 2.3 Confidence 与 Evidence

```python
@dataclass
class Evidence:
    cells: List[str]          # 验证单元唯一标识（task_seed）
    tasks: List[str]          # 已验证任务（跨任务才升 verified）
    attempts: int             # 总尝试次数
    solved_seeds: List[str]
    failed_seeds: List[str]
    contradicted_by: List[str]  # 矛盾记忆 ID
```

置信度升级规则（与 RAG `merge_evidence` 一致）：
- `cells >= 3` 且 `tasks >= 2` → `verified`
- `cells >= 2` → `probable`
- 否则 → `single-shot`

### 2.4 Skill 基类

```python
class Skill:
    spec: SkillSpec
    def execute(self, env, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError
    def get_skill_description(self) -> str: ...
```

`make_spec(name, description, fn, **kwargs)` 可从函数签名自动生成 `SkillSpec`（参数/必填项），减少样板代码。

## 3. 三层技能体系

### 3.1 感知技能（perception）

位于 `darwin/skills/perception/`：

| 技能 | 文件 | 功能 |
|------|------|------|
| `detect_objects` | `detect.py` | YOLO 目标检测，返回物体 bbox/类别 |
| `estimate_depth` | `detect.py` | 深度估计（注：深度来自系统外部真值，非相机输入） |
| `segment_object` | `segment.py` | SAM 物体分割（ultralytics.SAM，非 segment_anything） |
| `grasp_pose` | `grasp.py` | GraspNet 抓取位姿估计，双模式（真实模型 / 几何 PCA 回退） |

**GraspNet 双模式**：
- 优先用 `checkpoint-rs.tar` 真实模型预测
- 权重不可用时回退到几何方法（点云 PCA 估计抓取位姿）
- 点云直接从仿真物体几何采样，不依赖深度相机

### 3.2 控制原语（control.py）

位于 `darwin/skills/primitives/control.py`，提供**视觉伺服 + 力控**的参数化原语：

| 原语 | 功能 | 验证信号 |
|------|------|----------|
| `servo_align` | 视觉伺服对齐 | `reprojection_error < tol` |
| `guarded_move` |  guarded 移动（遇力停止） | `contact && travel < max` |
| `impedance_push` | 阻抗推入 | `depth_reached && force_peak < limit` |
| `spiral_search` | 螺旋搜索（找孔） | `depth_reached` |
| `pose_home` / `pose_move_above` / `pose_descend` / ... | 位姿空间 P 控制原语 | 到位误差 |

**设计哲学**：
- **参数化**：目标特征 + 容差 + 方向 + 阈值 + 终止条件，不绑定具体物体
- **三档执行模式**：`position`（默认，结构化无接触）/ `visual`（最后对齐，2-10cm）/ `force`（接触富集段）
- **失败映射**：`target_lost` / `excessive_force` / `stuck` / `slip` → 回给 Agent/RAG 做恢复

**力信号来源**：优先 `InsertEnv.get_ee_force`（`force_ee` sensor），无 sensor 时遍历 mujoco contact force 求和回退。

### 3.3 运动原语（motion.py）

位于 `darwin/skills/primitives/motion.py`，提供**路径规划 + 碰撞检测**：

| 原语 | 功能 | 失败映射 |
|------|------|----------|
| `path_plan` | mplib RRT-Connect 关节空间规划 | `no_path` / `joint_limit` / `collision` |
| `collision_check` | FCL 自碰撞 + 环境碰撞检测 | `self_collision` / `env_collision` |

**与 P 控制原语的互补关系**：

| 原语 | 适用场景 | 特点 |
|------|----------|------|
| `pose_move_to`（P 控制） | 近距离、无障碍迁移 | 不避障、不查碰撞、笛卡尔直线 |
| `path_plan`（RRT-Connect） | 远距离、复杂构型迁移 | 避自碰撞、关节空间规划 |
| `collision_check` | 构型安全性检查 | FCL 精确碰撞检测 |

典型用法：远距离用 `path_plan` 避障，最后几 cm 用 P 控制精确到位。

### 3.4 策略技能（forged）

位于 `darwin/skills/forged/`，由 `skill_forge` 从成功轨迹自动生成：

```python
class ForgedSkill(Skill):
    def bind(self, skills: Dict[str, Skill]):
        self._skills = skills   # 绑定原语表，forged 技能可调用原语
    def execute(self, env, **kwargs):
        for action in self._trajectory:
            result = self._skills[action["name"]].execute(env, **action["params"])
```

每个 forged 技能目录包含：
- `entry.py`：动作序列重放代码
- `SKILL.md`：元数据（name/description/kind/confidence/evidence/applies_when）

## 4. 技能加载

`load_skills()` 构建默认技能表，返回 `Dict[str, Skill]`：

```python
def load_skills() -> dict:
    skills = {}
    # 1. 感知技能
    for s in (DetectObjectsSkill(), EstimateDepthSkill(), SegmentObjectSkill(), GraspPoseSkill()):
        skills[s.spec.name] = s
    # 2. 原语技能（扫描 primitives 模块下所有 Skill 子类）
    from . import primitives
    for attr in dir(primitives):
        obj = getattr(primitives, attr)
        if isinstance(obj, type) and issubclass(obj, Skill) and obj is not Skill:
            inst = obj()
            skills[inst.spec.name] = inst
    # 3. forged 技能
    skills.update(load_forged_skills())
    return skills
```

forged 技能在加载后通过 `bind(self.skills)` 获得对原语的访问权。

## 5. 与 LLM 的对接

`ManipulationAgent._skill_specs_text()` 将所有技能的 `SkillSpec` 序列化为文本注入 LLM prompt：

```
- move_above: 移动到目标点上方 | 参数: {"point": "list", "hover": "float", ...}
- descend: 下降到抓取点 | 参数: {"point": "list", "body": "str", ...}
...
```

LLM 按 `<action>name</action><params>json</params>` 格式输出，Agent 解析后直接调用对应技能。
