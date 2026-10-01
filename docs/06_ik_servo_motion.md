# 06 · 逆运动学、末端伺服与运动控制

> 对应代码：`darwin/agents/libero_skills.py`、`darwin/skills/primitives/ik_servo.py`、`darwin/skills/primitives/motion.py`、`darwin/skills/primitives/control.py`、`darwin/envs/libero_adapter.py`
>
> 相关：[08 规划与执行栈](08_libero_planning_stack.md)（谁在调用这些动作）、[09 技能库](09_libero_skills.md)（上层技能）、[10 物理机制](10_physics_mechanisms.md)（失败如何分类）、[11 环境适配层](11_envs_libero_adapter.md)（动作如何进入仿真）

## 1. 三件事的分工

在 Agent + 技能的项目定位里，技能最终都要落到"把机械臂末端动到某个位姿"这一步。这一步由三层协作完成：

| 层 | 做什么 | 主要代码 |
|---|---|---|
| 逆运动学（IK） | 给定末端目标位姿，解出机械臂各关节角度 | `libero_skills.py` 的 `_ik_solve` / `_ik_solve_best` |
| 末端伺服（servo） | 不一次算到底，而是每步看一次当前误差、往前推一小步，形成闭环微调 | `libero_skills.py` 的 `serve`、`primitives/__init__.py` 的 `servo_step`、`primitives/ik_servo.py` 的 `IkServoSkill` |
| 运动控制（运控） | 决定走哪条路、走多快，把长距离移动拆成路径点并逐步执行 | `libero_skills.py` 的 `plan_arm_path` / `execute_arm_path`、`primitives/motion.py` 的 `PathPlanSkill`、`primitives/ik_servo.py` 的 `_follow_waypoints` / `CarrySkill` |

一句话概括：**IK 负责"关节摆成什么样"，伺服负责"够得准不准"，运控负责"从哪儿绕过去、用什么速度过去"。** 三者既能单独调用，也会串起来用：运控先用 IK 把目标位姿换成关节构型来做碰撞检查，再让伺服贴着路径点闭环前进。

三个术语先用白话交代清楚，后文会反复出现：

- **逆运动学（IK）**：正运动学是"给关节角算末端在哪"，逆运动学反过来——"给末端位置算关节角"。
- **伺服（servo）**：不追求一步到位，而是反复读当前状态、每次只修正一点点，让误差逐步缩小。
- **雅可比（Jacobian）**：一张描述"每个关节转动一点，末端会跟着动多少"的矩阵，IK 和伺服都靠它把末端误差换算成关节的调整量。
- **比例控制（P 控制）**：本步的修正量正比于当前误差——离得远就多走一点，离得近就少走一点。

## 2. 逆运动学（IK）

### 2.1 求解方法

实现是**阻尼最小二乘（DLS）迭代**，见 `libero_skills.py` 的 `_ik_solve`：

- 用 MuJoCo 的 `mj_jacSite` 取末端抓取点（site `gripper0_grip_site`）的位置雅可比；需要姿态时再叠加旋转雅可比。
- 每轮解 `dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e`，其中 `e` 是当前位置或姿态误差，λ 是阻尼系数 `IK_DAMP`。加阻尼是为了让手臂接近伸直（接近奇异位形）时不会算出过大的关节角。
- 关节增量按模型里每个关节的上下限逐关节截断，保证解不超出关节行程。
- 最多迭代 `IK_ITERS` 轮；位置误差小于 `IK_TOL` 就提前退出，带姿态时还要姿态误差小于 `ROT_TOL`。
- 求解在**一份独立的模型数据副本**上做，不扰动正在运行的仿真状态。

> 这套 IK 只在**规划、候选筛选、可达性检查**时调用。真正的执行路径（`servo_step`）不自己解 IK，而是把末端位姿增量直接交给仿真环境里的操作空间控制器（robosuite 的 OSC），由它内部完成角度换算。

### 2.2 输入输出

`_ik_solve(adapter, target, joints, dof_adr, target_rot=None, q_init=None)`：

- 输入：目标位置（世界系三维点）、可选目标姿态（3×3 旋转矩阵）、参与求解的臂关节列表、可选初始关节角。
- 输出：解出的关节角向量、末端位置雅可比、位置残差（米）；带姿态时额外返回姿态残差（弧度）。

### 2.3 多起点求解

单次 DLS 只保证局部收敛，而 7 自由度手臂存在冗余（同一个末端位姿可以有多种肘部朝向），第一个解很可能落在不合适的分支上。为此有两个包装：

- `_ik_solve_best`：第一次用当前关节角作起点，之后用随机采样、以及"当前与随机的插值"作起点，共 `IK_RESTARTS` 次；取"位置残差 + 姿态残差"最小的解。
- `_ik_in_tol_solutions`：枚举**全部容差内的解**。规划终点需要的是"够得到而且不撞"，而首个收敛解常常会横扫邻物，所以遍历候选解挑第一个无碰撞的，重启次数放大到 `4 × IK_RESTARTS`。

随机序列使用固定种子，同一输入的结果可重复。

### 2.4 失败如何报告

IK 本身不抛异常，只把结果交给调用方判断：

- 残差落在容差内 → 视为"够得到"。
- 残差超出容差 → 目标位姿不可达，上层按 `ik_unreachable`（目标点不在工作空间，或被障碍楔住）处理，`_fail` 会带上目标点、当前末端、残差和接触力等字段。

也就是说，求解过程没有独立的"失败"分支，只有"够不到"这一种结果。

## 3. 末端伺服（servo）

### 3.1 一步动作怎么算

统一入口是 `primitives/__init__.py` 的 `servo_step(env, site, target, gripper, k, vcap, actor)`：

- 读当前末端位置，算与目标的差值，乘比例增益 `k`，再按 `vcap` 限幅，得到本步的位移增量。
- 夹爪使用统一语义：`+1` 闭合、`-1` 张开、`0` 保持，由 `servo_step` 内部翻译成各环境的物理值。
- 如果环境自己实现了 `servo_step`（如 `LiberoEnvAdapter`），直接委托给它；否则按环境类型分别构造动作。
- 物理步完成后可选触发 `env._darwin_post_step()` 钩子（用于记录物理状态快照）。

### 3.2 闭环怎么做

LIBERO 侧的闭环是 `libero_skills.py` 的 `serve`：

1. 每步读当前末端位姿，算位置误差 `d_pos` 和姿态误差 `d_rot`。
2. 把本步的误差、臂关节角、夹持接触力记进窗口回归 `_ServoStats`。
3. 调用 `adapter.servo_step` 推进一步：例行用 `K_SERVO`，最后一段精细收敛用 `K_FINE`，限幅用 `VCAP`。

### 3.3 如何判定到位

- 调用方给了几何复验函数 `verify` 时，按它判定（例如 `goto` 传入 `tol` 时，构造"末端与目标距离 ≤ tol"的复验）。
- 否则要求位置残差小于 `max(IK_TOL, 3σ)`，**并且**姿态残差不超过 `ROT_TOL`。其中 `σ` 是最近一个窗口里误差的实测波动，相当于"进入测量噪声范围就算到了"，而不是固定死值。

### 3.4 没有进展时怎么判

每 `WIN` 步做一次窗口回归，拟合误差随时间的变化斜率：

- 斜率不够显著（t 统计量 < `TSTAT`），或窗口内的闭合距离不足 `0.25 × IK_TOL`，就记一次"停滞"。
- 连续 `STATIONARY_MAX` 个停滞窗口后，按物理证据分类：接触力超过基线 3σ → `contact_blocked`（真被挡住）；臂关节几乎不动 → `ik_unreachable`（到头了，或被楔住）。

设计取向是**不设预测步数预算**：进展慢但确实在前进的动作必须走完，真实止挡交给上面的统计判断来终止；`budget` 参数只留给调用方做硬性资源限制。

### 3.5 姿态伺服

`servo_step` 支持 `target_rot` 与旋转增益 `kr`。姿态误差用世界系轴角表示（`_rot_err`：三个坐标轴的叉积之和的一半），与旋转雅可比处在同一坐标系，便于对齐控制器"左乘期望旋转"的约定。

## 4. 运动控制（运控）

### 4.1 关节空间路径规划

`libero_skills.py` 的 `plan_arm_path` 用 **RRT-Connect 双树采样**（Kuffner & LaValle 2000）在关节空间找一条无碰撞路径：

- 距离度量用关节行程归一化后的欧氏距离；扩展步长 `RRT_STEP`，节点上限 `RRT_MAX_NODES`。
- 终点由 IK 解出（位置、姿态残差都在容差内，且自身无碰撞）。
- 找到后做 `SHORTCUT_N` 次"短路平滑"：两个点之间若能直线直达，就把中间的点剪掉。
- 碰撞细分与执行细分使用同一精度 `EXEC_DQ`，避免"规划时没事、执行时擦碰"。
- 采样序列有种子，同参数可重复；换种子重试是解开"同一次搜索恰好失败"的手段。

另一条更通用的路径在 `primitives/motion.py` 的 `PathPlanSkill`：把规划下沉到环境后端（`env.backend`），由后端用现成库（mplib 的 RRT-Connect + FCL 碰撞检测）规划，输出关节角序列；后端未注入时返回 `planner_unavailable`，不会让程序崩溃。

### 4.2 路径怎么执行

`execute_arm_path` 把规划出的路径按 `EXEC_DQ`（归一化关节行程）继续细分，逐点做前向运动学，边执行边过"跟踪门"：末端到当前目标点足够近，才允许继续前进；同时每一步都在真实物理状态上复验碰撞。LIBERO 里是逐点设置关节角、以准静态方式推进——`PathPlanSkill` 的执行阶段同样跳过控制器直设关节角，以免控制器重解 IK 把已经避开障碍的构型解掉。

### 4.3 轨迹、速度与加速度

- **路径点运动**：`ik_servo.py` 的 `_follow_waypoints` 对一串经过点逐段做三维距离收敛，每段自带容差和超时；`CarrySkill` 的"抬升 → 平移 → 下降"三段走廊就是它的典型用法。`IkServoSkill` 还提供 `above` / `descend` / `waypoint` 三种模式。
- **速度上限**：只有**每步位移上限** `vcap`（按 m/s 的语义使用，实际就是单步位移的截断值）。默认 `VCAP = 1.0`（OSC 全量程）；LIBERO 每个任务覆盖的 `vcap` 可低到 0.53，搬运和放置另有 `carry_vcap` / `place_vcap` = 0.05 的限速，推动类原语用 0.3，避障走廊段用 `SAFE_VCAP`。
- **加速度上限**：**当前未实现**。代码里没有显式的加速度或加加速度约束，运动的平滑程度由路径点的细分密度和每步限幅间接决定。
- **力与速度上限**：LIBERO 执行路径里**没有**关节力矩上限，也没有末端接触力上限。伺服的停止靠 3.3 的残差收敛与 3.4 的停滞分类，而不是"力超过某个值就停"。配置里的 `force_limit` / `stiffness` / `damping` 来自另一套面向刚度阻尼控制的原语（`primitives/control.py` 的位姿与力控原语），LIBERO 技能不使用它们。

## 5. 与环境的接口

- 技能层统一调 `servo_step`，不直接调 `env.step`。
- `LiberoEnvAdapter.servo_step`（`darwin/envs/libero_adapter.py`）是 LIBERO 这一侧的最终入口：把目标位置换算成增量并限幅，带上姿态修正和夹爪指令，打包成 7 维动作 `[dx, dy, dz, drx, dry, drz, g]` 发给 robosuite 的 OSC 控制器；episode 步数用尽时抛 `episode_terminated`，让技能干净退出。
- 通用技能走 `primitives/__init__.py` 的 `servo_step`，它负责把"闭合为正"的统一夹爪语义翻译成各环境的物理值；LIBERO 与统一语义同号，直接透传。
- 除位姿解算外，适配层还提供 `set_nullspace_posture`：在不改变末端任务目标的前提下，把未被约束的关节自由度（零空间）拉到指定构型，让物理构型和规划构型保持一致。

## 6. 关键参数表

| 参数 | 代码位置 | 值 | 含义 |
|---|---|---|---|
| `IK_TOL` | `libero_skills.py` | 0.003 m | IK 与几何"零距离"的精度，兼作接触/落座的几何零判定 |
| `ROT_TOL` | `libero_skills.py` | ≈0.05 rad（`IK_TOL/0.06`） | 姿态残差的容差 |
| `IK_ITERS` | `libero_skills.py` | 80 | DLS 逆解的迭代上限 |
| `IK_DAMP` | `libero_skills.py` | 0.05 | 阻尼最小二乘的阻尼系数 λ |
| `IK_RESTARTS` | `libero_skills.py` | 6（规划终点用 4×） | IK 多起点求解的次数 |
| `K_SERVO` / `K_FINE` | `libero_skills.py` | 5.0 / 2.5 | 伺服比例增益（常规 / 精细段） |
| `VCAP` | `libero_skills.py` | 1.0 | LIBERO 伺服每步位移上限（m/s 语义） |
| `CONF_K` | `libero_skills.py` | 3.0 | 判定到位时的噪声包络倍数（3σ） |
| `TSTAT` | `libero_skills.py` | 2.0 | 判定"没有进展"的 t 统计门限 |
| `WIN` | `libero_skills.py` | 30 | 窗口回归的样本步数 |
| `STATIONARY_MAX` | `libero_skills.py` | 3 | 连续停滞窗口数的上限 |
| `EXEC_DQ` | `libero_skills.py` | 0.003 | 路径执行点间距（归一化关节行程） |
| `RRT_STEP` / `RRT_MAX_NODES` / `SHORTCUT_N` | `libero_skills.py` | 0.15 / 6000 / 80 | 关节路径规划：扩展步长 / 节点上限 / 平滑尝试次数 |
| `_DEFAULT_K` / `_DEFAULT_STALL_WINDOW` / `_DEFAULT_STALL_PROGRESS` | `primitives/ik_servo.py` | 5.0 / 30 / 0.005 | 通用 IK 技能的默认增益与停滞检测参数 |
| `reach_tol` / `hover` / `timeout` | `IkServoSkill.execute` | 0.006 m / 0.08 m / 120 步 | 下降到点容差 / 悬停高度 / 单段步数上限 |
| `contact_stop_band` / `reach_limit_band` / `f_free_n` | `IkServoSkill.execute` | 0.05 / 0.035 / 0.5 N | 接触软停带 / 可达极限接受带 / 自由空间接触力阈值 |
| `SAFE_Z` / `ABORT_DIST` / `SAFE_VCAP` | `primitives/collision.py` | 0.62 m / 0.004 m / 0.05 | 安全走廊高度 / 碰撞中止距离 / 走廊段每步位移上限 |
| `hover` / `k` / `k_descend` / `vcap` / `reach_tol` / `stop_above` | `skills/configs/ik_servo.*.yaml` | 每个任务各自覆盖 | LIBERO 每个任务的技能参数（如悬停高度、增益、限速、容差、下降偏移） |

## 7. 从末端目标位姿到关节动作

一条单独的点到点移动，走的是"规划阶段 + 执行阶段"两条路：

```
                  ┌─ 规划 / 检查 ─► _ik_solve（DLS 迭代，多起点）
                  │                    │
末端目标位姿 ──────┤                    ├─ 残差 > IK_TOL ─► ik_unreachable
 (x, y, z [, R])  │                    │
                  │                    └─ 残差 ≤ IK_TOL ─► 关节构型 q
                  │                                        （用于碰撞检查 / 路径终点）
                  │
                  └─ 执行 ───────► servo_step（末端 P 伺服一步）
                                       │  delta = clip(k·(target − end), ±vcap)
                                       └─► [dx,dy,dz, drx,dry,drz, g]
                                              ─► OSC 控制器 ─► 关节力矩 / 位置
```

对一条长距离、需要绕开障碍的移动，流程则是：

1. **定目标**：上层技能给出末端目标位姿（位置，必要时带姿态）。
2. **解 IK**：`_ik_solve` 多起点求解，取容差内、无碰撞的关节构型作为路径终点。
3. **规划路径**：`plan_arm_path` 在关节空间跑 RRT-Connect，再用短路平滑把路径压短。
4. **细分执行**：`execute_arm_path` 按 `EXEC_DQ` 细分，逐点过跟踪门并复验碰撞。
5. **收尾伺服**：最后一段用 `serve` 精细收敛到目标，`strict=False` 时把末端残余误差交给后续的事件驱动原语消除。
6. **落成动作**：每一步都由 `servo_step` 变成环境能接收的动作，送进 OSC 控制器产生关节运动。

对整个流程来说，**终点精度由 IK 容差和伺服收敛决定，路径安全由规划时的碰撞检查和执行时的跟踪门共同保证，移动快慢由 `vcap` 与路径点密度控制**。
