# LLM 生成器 Prompt:为新机型生成 RobotBackend 适配器

你是一个机器人软件工程师。任务:为**新机型**生成 `darwin/robot/` 下的 Backend 子类 + `darwin/assets/profiles/<name>.yaml`,使其接入 darwin 的 motion 原语(path_plan / collision_check),**不改 Skill 层、不改 Registry、不改 motion.py**。

## 背景:darwin 的 RobotBackend 抽象

darwin 的运动原语(`darwin/skills/primitives/motion.py` 的 `path_plan`/`collision_check`)不直接调任何 SDK,只调 `env.backend` 的高层接口。`RobotBackend` 是抽象基类(`darwin/robot/backend.py`),子类屏蔽 mujoco/ROS/Franka/Realman 等 SDK 差异。换机型 = 写一个 Backend 子类 + 一份 Profile YAML,运行时 `runner._get_env` 按 profile 注入 backend。

**ABC 接口(必须全部实现)**:
```python
class RobotBackend(ABC):
    profile: RobotProfile
    def bind(self, env) -> None
    def get_arm_qpos(self, actor) -> np.ndarray               # 7 维关节角
    def get_eef_pose(self, actor) -> (pos(3), quat(4) [w,x,y,z])
    def get_eef_to_move_group_offset(self, actor) -> np.ndarray  # eef→move_group 位置偏移
    def get_base_pose(self, actor) -> np.ndarray              # [pos(3), quat(4)]
    def prepare_collision_assets(self) -> None                # URDF/SRDF/mesh/convex 幂等准备
    def plan_path(self, goal_pose, actor, frame, current_qpos, **opts) -> Dict
    def execute_trajectory(self, traj, actor, skip, timeout) -> Dict
    def check_self_collision(self, actor=None, qpos_arm=None) -> List
    def check_env_collision(self, actor, qpos_arm, use_attach, with_pc) -> List
```
`plan_path` 返回 `{ok, status, trajectory, duration, reason}`;`execute_trajectory` 返回 `{executed, reason, final_qpos}`;collision 返回 mplib collision 对象列表(或空)。

## 参考实现

`darwin/robot/mujoco_robopal.py` 的 `MujocoRobopalBackend` 是默认实现(mujoco + robopal + mplib)。`darwin/assets/profiles/diana_med.yaml` 是单臂 profile 范例。**强烈建议**:若新机型有 URDF 且能用 mplib,直接继承 `MujocoRobopalBackend` 只改 `profile`(甚至不需要写子类,只写 YAML);只有 SDK 接口完全不同(如 ROS MoveIt、libfranka 实时流式)才从头实现 ABC。

## 你的任务

针对给定机型,产出两个文件:

### 文件 1:`darwin/assets/profiles/<name>.yaml`
按 `RobotProfile` schema 填(参照 diana_med.yaml)。必须从机型 SDK/URDF/文档抽出 8 个关键事实:

1. `manipulator`:mplib 缓存 key(机器人名,如 panda / rm75)
2. `urdf_path`:URDF 路径(相对 robopal assets 或绝对)
3. `mesh_root`:mesh 目录(用于 convex hull 生成)
4. `urdf_mesh_rewrite`:URDF 里 `package://...` 路径 → `meshes/`(相对 cache 目录)的重写规则
5. `adjacent_link_pairs`:单臂内相邻 link 对(base-link1...link6-link7),SRDF 必禁。**双臂时跨臂对不写**(要检测避碰)
6. `extra_disabled_pairs`:工作构型下 convex hull 误报对(如 joint 大角度弯曲时)
7. `actors`:每臂配置(单臂就 agent0):
   - `joint_names`:URDF 原生 joint 名(无前缀,如 j1..j7 或 panda_joint1..7)
   - `move_group_link`:mplib move_group(URDF 原生 link,如 link7 / panda_link8)
   - `mujoco_joint_prefix`:mujoco body 前缀(robopal 双臂 "0_"/"1_";纯 URDF 后端可空)
   - `mujoco_base_body` / `mujoco_eef_body`:mujoco body 名
8. `backend_type`:子类注册名(继承 MujocoRobopalBackend 时用 `mujoco_robopal`;新子类自定)

### 文件 2:`darwin/robot/<name>_backend.py`(仅当不能用 mujoco_robopal 时)
继承 `RobotBackend`,在模块末尾 `register_backend("<backend_type>", <Class>)`。实现 10 个抽象方法,调对应 SDK:
- mujoco env 读 qpos:`env.robot.get_arm_qpos(actor)` / body xpos
- ROS:rospy + MoveIt action client / JointTrajectory
- Franka:libfranka (`pyptrest`/`frankapy`) 实时控制 + FCL 碰撞
- 若该机型有 URDF 且接受 mplib,**优先继承 MujocoRobopalBackend 只改 profile**,不重写方法

## 输入

你会收到:
- 机型名 `<name>`
- SDK 路径或文档 URL(读源码/文档抽 8 个事实)
- (可选)URDF/mesh 文件路径

## 自检清单(生成后必须满足)

1. `python -c "from darwin.robot import make_backend; b=make_backend('<name>'); print(b is not None)"` → True
2. `python -c "from darwin.robot import make_backend; from <env构造>; b=make_backend('<name>', env); b.prepare_collision_assets()"` → 不抛异常
3. `collision_check` home → `collision_free=True`(无相邻对误报,说明 SRDF 正确)
4. `path_plan` z+2cm → `final_err < 1e-3`(说明 move_group/eef 偏移/base_pose 正确)
5. 注册表不变:`len(build_registry().list_names())` 不应因加 backend 变化(backend 不注册成 skill)

## 输出格式

先输出 `<name>.yaml` 完整内容(用 ```yaml 围栏),再输出 `<name>_backend.py` 完整内容(用 ```python 围栏,若继承 mujoco_robopal 则说明"仅需 YAML,无需新子类"并跳过)。最后给一句话总结:该机型走 mplib 路径还是自定义 SDK 路径。
