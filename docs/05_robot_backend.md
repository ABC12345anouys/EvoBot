# 05 · 机器人后端抽象设计

> 对应代码：`darwin/robot/backend.py`、`darwin/robot/mujoco_robopal.py`、`darwin/assets/profiles/`

## 1. 设计目标

机器人后端抽象层解决**机型解耦**问题：让 `motion.py` 原语（path_plan / collision_check）不硬编码任何机型/SDK，换机型只需新增 Backend 子类 + YAML Profile，核心代码零改动。

设计目标：
1. **模板 + 动态**：Skill 层签名固定（模板），机型差异下沉到 Backend（动态多态）。
2. **配置驱动**：有 URDF 的机型只写 YAML Profile，无需写 Backend 子类。
3. **双臂向前兼容**：Profile actors dict（单臂一个 entry）、方法全 actor 参数化。
4. **优雅降级**：无 Profile 时 backend 为 None，motion 原语返回 `planner_unavailable`，不崩溃。

## 2. 架构分层

```
Skill 层 (motion.py)
    │  只调 env.backend 高层接口
    ▼
RobotBackend ABC (backend.py)
    │  定义接口契约
    ▼
具体 Backend 子类 (mujoco_robopal.py / ros_backend.py / franka_backend.py)
    │  实现 SDK 调用
    ▼
RobotProfile (YAML) ── 机型配置（URDF/link/joint/SRDF）
```

## 3. RobotBackend 抽象基类

```python
class RobotBackend(ABC):
    profile: RobotProfile
    _env: Any

    def bind(self, env): ...           # 绑定 env 引用

    # ---- 状态读取 ----
    @abstractmethod
    def get_arm_qpos(self, actor) -> np.ndarray
    @abstractmethod
    def get_eef_pose(self, actor) -> (pos, quat)
    @abstractmethod
    def get_eef_to_move_group_offset(self, actor) -> np.ndarray
    @abstractmethod
    def get_base_pose(self, actor) -> np.ndarray

    # ---- 资产准备 ----
    @abstractmethod
    def prepare_collision_assets(self)

    # ---- 高层运动操作 ----
    @abstractmethod
    def plan_path(self, goal_pose, actor, frame="world", current_qpos=None, **opts)
    @abstractmethod
    def execute_trajectory(self, traj, actor, skip=2, timeout=200)
    @abstractmethod
    def check_self_collision(self, actor=None, qpos_arm=None)
    @abstractmethod
    def check_env_collision(self, actor, qpos_arm=None, use_attach=True, with_pc=False)
```

**所有方法 actor 参数化**，单臂传 `"agent0"`，双臂传 `"agent0"` / `"agent1"`。

## 4. RobotProfile 配置

```python
@dataclass
class RobotProfile:
    name: str
    backend_type: str               # 选哪个 Backend 子类（注册表 key）
    manipulator: str                # mplib 缓存 key
    urdf_path: str
    mesh_root: str
    srdf_path: Optional[str]        # null=自动生成
    urdf_mesh_rewrite: Dict[str, str]  # package:// → meshes/
    adjacent_link_pairs: List[List[str]]    # 单臂内相邻对（SRDF 禁碰）
    extra_disabled_pairs: List[List[str]]   # 工作构型误报对
    actors: Dict[str, ArmConfig]    # 各臂配置
```

`ArmConfig`：
```python
@dataclass
class ArmConfig:
    joint_names: List[str]          # URDF 原生 j1..j7
    move_group_link: str            # mplib move_group，如 link7
    mujoco_joint_prefix: str        # robopal 双臂前缀 "0_" / "1_"
    mujoco_base_body: str
    mujoco_eef_body: str
```

### 4.1 YAML 示例（diana_med.yaml）

```yaml
name: diana_med
backend_type: mujoco_robopal
manipulator: diana_med
urdf_path: darwin/assets/robots/diana_med/diana_med.urdf
mesh_root: darwin/assets/robots/diana_med
urdf_mesh_rewrite:
  "package://diana_med/meshes": "meshes"
adjacent_link_pairs:
  - [base_link, link1]
  - [link1, link2]
  # ...
extra_disabled_pairs:
  - [link3, link5]   # 工作构型下的误报对
actors:
  agent0:
    joint_names: [j1, j2, j3, j4, j5, j6, j7]
    move_group_link: link7
    mujoco_joint_prefix: "0_"
    mujoco_base_body: "0_base_link"
    mujoco_eef_body: "0_eef"
```

## 5. Backend 注册表

```python
_BACKEND_REGISTRY: Dict[str, type] = {}

def register_backend(name, cls):
    _BACKEND_REGISTRY[name] = cls

def make_backend(profile_name, env=None) -> Optional[RobotBackend]:
    yaml_path = PROFILES_DIR / f"{profile_name}.yaml"
    if not yaml_path.exists():
        return None                          # 无 Profile → None（优雅降级）
    profile = RobotProfile.from_yaml(yaml_path)
    cls = _BACKEND_REGISTRY.get(profile.backend_type)
    if cls is None:
        raise ValueError(f"未知 backend_type: {profile.backend_type}")
    backend = cls(profile)
    if env is not None:
        backend.bind(env)
    return backend
```

子类模块 import 时调用 `register_backend("mujoco_robopal", MujocoRobopalBackend)` 完成注册。

## 6. MujocoRobopalBackend 实现要点

`mujoco_robopal.py` 是基于 robopal + mplib 的具体实现，关键技术点：

### 6.1 URDF mesh 路径修正

robopal 的 URDF 中 mesh 路径使用 `package://` 协议，mplib 不识别。解决：
- 复制 URDF 到 `darwin/assets/mplib_cache/`
- 用 `urdf_mesh_rewrite` 把 `package://diana_med/meshes` 改写为相对路径 `meshes/`
- 去掉 `robot/` 子目录前缀

### 6.2 SRDF 手动定义

mplib 无 SRDF 时会做百万级碰撞对采样，极度缓慢。解决：
- `adjacent_link_pairs`：单臂内相邻 link 对（如 base_link↔link1），SRDF 中禁碰
- `extra_disabled_pairs`：工作构型下的误报对（如 link3↔link5）
- 双臂跨臂对不禁用，保留避碰检测

### 6.3 eef → move_group 偏移

mplib 的规划目标是 `move_group_link`（如 link7），但 Agent 操作的是 eef（夹爪中心）。两者有固定偏移：

```python
def get_eef_to_move_group_offset(self, actor):
    # eef 在 world 的位置 - move_group_link 在 world 的位置
    return eef_pos - move_group_pos
```

规划目标位姿需减去此偏移，转换为 link7 目标。

### 6.4 轨迹执行补末帧

`execute_trajectory` 必须执行轨迹的**最后一个 waypoint**：
- 降采样（`skip` 参数）会跳过最后一帧（目标构型），导致到位误差
- 实现中确保 `trajectory[-1]` 一定被执行

### 6.5 准静态执行

用 `mj_forward` 而非控制器执行轨迹：
- 直接设置关节角 + `mj_forward` 步进
- 避免 robopal 控制器（尤其 CARTIMP）的不稳定性
- 适合路径规划后的准静态轨迹重放

## 7. 扩展新机型

### 情况 A：有 URDF 的机型（推荐）

1. 准备 URDF + mesh 文件
2. 写 `darwin/assets/profiles/<name>.yaml`（参考 diana_med.yaml）
3. 若使用 mujoco/robopal，`backend_type: mujoco_robopal` 即可，无需写子类
4. 在 `runner._get_env` 的 `_PROFILE_MAP` 中添加 `{robot_name: profile_name}` 映射

### 情况 B：ROS / Franka / Realman 等 SDK 机型

1. 继承 `RobotBackend` 实现子类，实现所有 abstractmethod
2. 在模块 import 时 `register_backend("sdk_name", MyBackend)`
3. 写 `darwin/assets/profiles/<name>.yaml`，`backend_type: sdk_name`
4. LLM 辅助生成：`scripts/gen_backend.py` 读 SDK 文档生成 Backend + Profile 骨架，人 review 后入库

### 情况 C：无 URDF 的纯关节机型

需要在 Backend 子类中自行实现运动学（不依赖 mplib），`plan_path` 可改用 IK + 插值，`check_self_collision` 可省略或简化。
