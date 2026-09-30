# 16 · 外围子系统设计（LLM / 机器人后端 / 基准 / 工具）

> 对应代码：`darwin/llm/`、`darwin/robot/`、`darwin/benchmarks/`、`darwin/utils/`
>
> 与 [05 机器人后端抽象](05_robot_backend.md)、[06 LLM 集成](06_llm_integration.md) 的关系：那两篇讲设计意图；本篇补**实现细节、接线现状与实测陷阱**。

## 1. `darwin/llm/`：LLM 规划

### 1.1 文件与协议

| 文件 | 职责 |
|---|---|
| `llm/client.py`（74 行）| OpenAI 兼容 `/chat/completions` 客户端；配置全走环境变量；含进程级 token 累计 |
| `llm/planner.py`（264 行）| `LLMPlanner`（指令 + WorldState → 结构化 JSON Plan，含 grounding 校验与一次修复）+ `OfflinePlanner`（无 LLM 的确定性堆叠计划）|

- **只用标准库 `urllib.request`**，无第三方 SDK（`client.py:13-15`）；仅实现 OpenAI 兼容 `/chat/completions`（`:37-39`）
- 后端分支：存在 `ARK_API_KEY` → 火山方舟 Ark（默认 base_url `https://ark.cn-beijing.volces.com/api/v3`）；否则 OpenAI（默认 `https://api.openai.com/v1`）`:19-27`
- 配置项：`ARK_API_KEY/ARK_BASE_URL/ARK_MODEL` 或 `OPENAI_API_KEY/OPENAI_BASE_URL/OPENAI_MODEL`；`available()` 判据是 `api_key and model` 均非空（无硬编码密钥）`:1-35`

### 1.2 结构化输出的约定（**不用 function calling**）

- 约定为"LLM 只输出 JSON"，Plan schema 写在模块 docstring：`goal_predicates` + `steps`（每步 `skill`/`args`/可选 `expect`）`planner.py:9-14`
- `_SYSTEM` 提示词强制"只输出 JSON，不要 markdown 代码块"，并注入可用谓词与技能清单 `:85-104`
- **解析容错**：先用正则剥 ```json 围栏；否则取首个 `{` 到末个 `}`；仍非法 → 抛 `PlanError` `:179-198`
- **grounding 校验** `validate()`：谓词名须在 `predicate_docs()`、goal/step 引用的实体 label 须存在于 WorldState、技能名须在 `MANIFEST`、`expect` 须可求值 `:56-77`
- 校验失败有**一次**"把错误喂回重出"的机会（`max_repairs=1`），仍失败抛 `PlanError` `:157-175`

### 1.3 超时/重试与采样参数

| 项 | 值 |
|---|---|
| HTTP 超时 | `120 s`（直接传给 `urlopen`）`client.py:20,53` |
| 网络层重试 | **无**——`chat()` 失败直接抛异常 `:37-64`；重试只体现在规划层的"一次 JSON 修复" |
| `temperature` | `0.2`（client 与 planner 各一处默认）|
| `max_tokens` | client 默认 `1024`；planner 实际调用 `1500` |
| token 累计 | `_USAGE={"total":0}`；响应无 usage 字段时按 `(len(messages)+len(content))//4` 粗估 `:60-70` |

> ⚠️ **接线落差**：`client.py:7` 与 `planner.py:4` 的注释声称"Planner 自动回退 `OfflinePlanner`"，但全仓库 grep **没有任何 `OfflinePlanner(` 实例化/选择点**；`agent.py:76` 的实际写法是 `LLMClient() if LLMClient().available() else None`。即**离线回退器已实现但未接线**。

## 2. `darwin/robot/`：机器人后端抽象

### 2.1 接口（`RobotBackend` ABC）

| 方法 | signature 要点 |
|---|---|
| `__init__(profile)` / `bind(env)` | 构造后注入 env；`env` property 在未 bind 时抛 `RuntimeError` |
| `get_arm_qpos(actor)` | 7 维臂关节角 |
| `get_eef_pose(actor)` | `(pos(3), quat(4)[w,x,y,z])` |
| `get_eef_to_move_group_offset(actor)` / `get_base_pose(actor)` | 偏移 / base 世界位姿 |
| `prepare_collision_assets()` | URDF/SRDF/mesh/convex **幂等**准备 |
| `plan_path(goal_pose, actor, frame="world", current_qpos=None, **opts)` | → `{ok,status,trajectory,duration,reason}` |
| `execute_trajectory(traj, actor, skip=2, timeout=200)` | → `{executed,reason,final_qpos}` |
| `check_self_collision(actor=None, qpos_arm=None)` / `check_env_collision(actor, qpos_arm=None, use_attach=True, with_pc=False)` | 碰撞列表 |

除 `bind`/`env` 外全部是 `@abstractmethod`（`backend.py:81-144`）。

### 2.2 Profile 与解耦

- `RobotProfile` 字段：`name / backend_type / manipulator / urdf_path / mesh_root / srdf_path(null=自动生成) / urdf_mesh_rewrite / adjacent_link_pairs / extra_disabled_pairs / actors: Dict[str, ArmConfig]`（`backend.py:40-51`）
- `ArmConfig`：`joint_names / move_group_link / mujoco_joint_prefix / mujoco_base_body / mujoco_eef_body`（`:30-38`）
- 解耦方式 = **Adapter/Strategy + 配置驱动**：技能层只调 backend 接口，换机型只需新增 Backend 子类 + YAML
- 运行时分派：`make_backend(profile_name, env)` 读 `assets/profiles/<name>.yaml` → 按 `backend_type` 查 `_BACKEND_REGISTRY` → 构造并可选 bind；未知类型抛 `ValueError`，profile 不存在返回 `None` `:146-166`
- 注入点：`agents/env_utils.py:43-48` 对 `DianaTripleStack` 经 `_PROFILE_MAP` 映射到 `diana_med` 后调用 `make_backend(..., env=env)`
- 双臂前向兼容：方法全 `actor` 参数化、planner 缓存 key 为 `(manipulator, actor)`、SRDF **不禁用跨臂对**以自动避碰；**协同双臂规划 mplib 不支持**，留给未来 `BimanualBackend`（`backend.py:13-19`）

### 2.3 mplib 接入点（唯一实现 `mujoco_robopal.py`）

- 全局缓存 `_PLANNER_CACHE: {(manipulator, actor): Planner}`；`_get_planner` 懒构造（先 `prepare_collision_assets()`，再以缓存 URDF/SRDF + `move_group=arm.move_group_link` + `user_joint_names=arm.joint_names` 建 `mplib.Planner`）`:22,224-242`
- 同步：`planner.set_base_pose(get_base_pose(actor))` → `planner.robot.set_qpos(qpos, True)` `:244-251`
- 规划调用 `planner.plan_qpos_to_pose(...)`，默认 `planner_name="RRTConnect"`、`use_point_cloud=False`、`use_attach=False` `:271-282`
- 碰撞检查 `planner.check_for_self_collision` / `check_for_env_collision` `:312,325-326`
- 资产准备：URDF mesh 重写、mesh symlink、trimesh 预生成 `<mesh>.convex.stl`、SRDF 由 `adjacent_link_pairs`/`extra_disabled_pairs` 自动拼 `<disable_collisions>` `:159-222`
- 执行路径 `_set_arm_qpos_quasi_static`：直写 `data.qpos[adr]` 后 `mj_forward`，并**手动 `env._timestep += 1`** `:119-131`

**已知边界（注释内固化）**：

| 边界 | 说明 |
|---|---|
| **mplib 只规划单链** | `move_group=link7` 单链；双臂 = 每臂一个 planner、顺序规划 |
| **`mj_forward` 跳过动力学** | 夹爪保持当前 qpos（夹持状态保持、物体几何跟随）、**无惯性/无力觉**；手动自增 `_timestep` 会消耗 episode 步数 |
| **降采样会漏末帧** | `skip` 步进可能跳过最后一帧 → 补执行末位 waypoint 保证到达目标构型 |
| **eef→move_group 姿态不转** | 仅位置偏移、姿态沿用；注释注明"竖直夹爪场景够用"——**倾斜夹爪是已知失效边界** |
| **convex hull 预生成失败不阻断** | `except: pass`，但 `_get_planner` 依赖 `.convex.stl` 已存在 |

## 3. `darwin/benchmarks/`：基准与任务分布

- **静态字典 `BENCHMARKS`（5 条）+ `get_benchmark(name)` 查表**，未命中抛 `KeyError`——**没有装饰器/注册函数**（`benchmarks/__init__.py:35-91`）

| key | env_id / robot | 备注 |
|---|---|---|
| `pickplace` | `BimanualPickAndPlace-v0` / `DualPandaPickAndPlace` | 双臂，`mode=full`，含 agent0/agent1 bounds |
| `stack_grasp_red` / `stack_grasp_blue` | `MultiCubeStack-v1` / `DianaTripleStack` | 单臂，`mode=grasp` |
| `peg_in_hole` | `InsertEnv` / `DianaTripleStack`，`controller=CARTIK` | ⚠️ **名不符实**：注释指当前模型里 `red_block` 是 4cm 立方体、`red_goal` 是空中目标点（**无孔**），实际是 cube 放置任务，故用 `mode=full`；换成真 peg/hole 模型后才切回 `mode=insert` 启用力控插装链 |
| `drawer_place` | `DrawerBox-v1` / `DianaDrawerCube` | `mode=drawer`，含 drawer/cube goal site |

- 每条目字段：`env_id/robot/body/actor/grip_site/bimanual/mode/task_desc/search_space` + `_FEATURES`（公共项 `**_FEATURES` 展开）
- **LIBERO 走动态工厂** `get_libero_benchmark(suite, task_idx)`：注释明确"任务集不进静态 BENCHMARKS 表"；生成的 entry 为 `env_id="libero:<suite>:<idx>"`、`robot="libero_panda"`、`mode="libero"`，并携带 `libero_suite/libero_task_idx/libero_language/libero_goal/libero_spec`；`body` 取 BDDL goal 谓词中被抓物，无 goal 时退回 BDDL 首对象
- 与 robopal 无关（本模块不 import robopal）；`env_id`/`robot` 名由 `agents/env_utils.get_env` 消费（`robopal.make(...)`）
- **MAP-Elites 的数据源**也在本文件：`_FEATURES`（`feature_dimensions/fitness_key`）+ `COMMON_SEARCH_SPACE` + `INSERT_SEARCH_SPACE`（详见 [13 进化系统](13_evolution_map_elites.md) §6）

## 4. `darwin/utils/`

| 文件 | 职责 |
|---|---|
| `download.py`（138 行）| 统一权重下载：缓存 → Gitee/镜像 → 原始 URL |
| `mjtree.py`（26 行）| MuJoCo body 子树枚举（`subtree_body_ids`）|
| `recorder.py`（66 行）| EGL 离屏录像，包装 `env.step` 逐帧写 MP4 |

### 4.1 `download.py`

- 镜像表 `GITEE_MIRRORS` `:26-37`：`yolo26n*.pt` → `hf-mirror.com`（**注释说明 Gitee `ultralytics/assets` 仓库为空**）；`mobile_sam.pt`/`sam_b.pt`/`FastSAM-s.pt` → `gitee.com/ultralytics/assets` release v8.4.0；**`checkpoint-rs.tar` 为 `None`**（无公开直链）
- 备用 `FALLBACK_URLS` `:39-44`：4 个 ultralytics 权重的 GitHub release 直链
- 缓存 `CACHE_DIR`：`DARWIN_WEIGHTS_CACHE` 或 `~/.cache/darwin/weights` `:22-23`
- 下载顺序 `gitee_download`：本地存在且 MD5 通过 → GITEE_MIRRORS → FALLBACK_URLS；逐个试错、失败即 `unlink` 残file `:53-91`
- `_download_url`：优先 `wget -q --tries=3 --timeout=30 -O`，失败回退 `urllib` 写 `.part` 再 `rename` `:94-110`
- `ensure_weight` 四段式：本地指定路径 → `/home/lifd/Public/<name>` → 缓存 → 下载 `:113-137`
- 被 `skills/perception/{segment,grasp,detect}.py` 广泛 import

> ⚠️ **硬限制**：`checkpoint-rs.tar`（GraspNet）**无公开直链**——注释写"从百度网盘下载，已存于 `/home/lifd/Public/checkpoint-rs.tar`"，镜像值为 `None`；下载失败抛 `FileNotFoundError`。**换机器无法自动获得该权重**。

### 4.2 `mjtree.py` —— 一个值得记住的陷阱

`subtree_body_ids` 沿 `body_parentid` 链回溯收集子树 body id。**为什么不用 `body_rootid` 分组**（注释内固化）：

> mujoco 只把**自由关节装配根**标为子树根；焊在 world 下的 LIBERO 柜架/炉具 body 的 `rootid == 0`，按 rootid 等价类收集 geom 会得**空集**，导致柜架碰撞体对 `object_bounds`/点云采样隐身、抓取偏置失去柜侧净空约束。
> 实证：`spatial:6` → `ik_unreachable`，`coll=gripper/cabinet_base`。

### 4.3 `recorder.py`

- `make_camera(bimanual)`：`mjCAMERA_FREE`；双臂 lookat `[0.5,0,0.45]`/distance 2.6/azimuth 90/elevation −25；单臂 `[0.45,0,0.40]`/2.0/135/−22
- `StepRecorder`：构造时替换 `env.step`，每帧 `update_scene`+`render` 写 `cv2.VideoWriter`；**`close()` 必须调用**以恢复原 step
- 模块顶部就设环境：`MUJOCO_GL=egl`、`CUDA_VISIBLE_DEVICES=DARWIN_GPU 或 0`、egl 时 `pop DISPLAY` `:5-10`
- 参数：`fps=20`、`width=640`、`height=480`

## 5. 调参常量速查

| 子系统 | 常量 | 值 |
|---|---|---|
| llm | HTTP 超时 / `temperature` / `max_repairs` | `120s` / `0.2` / `1` |
| llm | `_combos`（离线规划重规划轮转）| `[(auto,0),(auto,1),(left,2),(right,0),(right,1)]` |
| robot | `skip` / `timeout` / `time_step` / `rrt_range` / `planning_time` / `planner_name` | `2` / `200` / `0.1` / `0.1` / `1.0` / `"RRTConnect"` |
| robot | `PROFILES_DIR` | `darwin/assets/profiles` |
| benchmarks | `COMMON_SEARCH_SPACE` | `hover [0.10,0.12,0.15]`、`k_descend [1.5,2.0,2.5]`、`lift_height [0.50,0.52,0.55]`、`jit [0.0,0.005,0.01]` |
| benchmarks | `INSERT_SEARCH_SPACE` | 追加 `stiffness [60,100,150]`、`damping [20,40,60]`、`spiral_radius [0.003,0.004,0.006]` |
| utils | 缓存 / MD5 块 / wget | `~/.cache/darwin/weights` / `8192` / `--tries=3 --timeout=30` |
| utils | recorder | `fps=20`、`640×480` |

## 6. 已核对的落差与限制（如实记录）

1. **`OfflinePlanner` 未接线**：注释声称自动回退，实际无实例化点（`agent.py:76` 走 `None`）。
2. **`AgentLoop` 未被实例化**：`agents/agent_loop.py` 的通用闭环在仓库内**没有任何 `.py` 实例化它**（与 [14](14_dynamic_runner.md) §8 的差异分析对应）。
3. **YAML profile 只有一个**：`assets/profiles/` 仅有 `diana_med.yaml`，而 `_PROFILE_MAP` 只映射 `DianaTripleStack → diana_med`；因此 benchmark 里的其他 robot（`DualPandaPickAndPlace`、`DianaDrawerCube`、`libero_panda`）**当前不会被注入 backend**（`make_backend` 只在映射命中时调用）。
4. **Gitee ultralytics 镜像仓库为空** → YOLO26 权重实际走 hf-mirror；`checkpoint-rs.tar` 无公开直链（见 §4.1）。
5. **eef→move_group 姿态不转**：倾斜夹爪场景是已知失效边界。
6. **双臂协同规划缺失**：mplib 不支持，需未来 `BimanualBackend`。
