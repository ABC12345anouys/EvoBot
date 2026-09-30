# 09 · LIBERO 技能库设计

> 对应代码：`darwin/agents/libero_skills.py`（5338 行，本仓库最大的单文件）
>
> 与 [03 技能框架](03_skills_framework.md) 的关系：03 描述的是 `darwin/skills/` 那套**带 `SkillSpec` 元数据的技能框架**；本篇描述的是 LIBERO 侧的**具体技能实现**——函数式接口、面向"BDDL 谓词"而非"LLM function calling"。

## 1. 设计目标

文件头 docstring 自述两点约定（`libero_skills.py:2-16`）：

1. **本文件不含任务阈值**。所有数字分四类来源：① 几何/物理常量，② 统计判据约定（3σ、t 检验），③ 控制/规划参数，④ 由场景特征现算（见 [10 物理机制](10_physics_mechanisms.md) 的 `derives`）。
2. **技能与谓词种类一一对应**：`grasp` / `place_at` / `articulate` / `toggle` / `move_to`（外加 `push_to` 作为 place 的第二物理实现）。

设计取向：所有技能都返回 `{success, reason, mechanism, measures}`——**失败必须分类**，因为重规划要按机制决定下一个参数。

## 2. 对外技能清单

| 函数 | 行 | 职责 |
|---|---|---|
| `serve` | `:2575` | **通用伺服底座**：位置/位姿伺服到 target，到位与停滞由统计判据裁决 |
| `goto` | `:2666` | `serve` 的兼容包装：`tol` → 几何复验，`timeout` → 显式预算 |
| `execute_corridor` / `_reach_site` / `execute_arm_path` | `:2684` / `:2700` / `:2749` | 路径层：多腿走廊、位点到达（接触允许）、关节路径流式执行 |
| `grasp` | `:3439` | 任务无关抓取：全场景候选逐条走 TAMP 管线，任一环不过换下一候选 |
| `push_to` | `:4045` | 沿支撑面把物体推/拨进目标区域（On/In 的第二物理实现）|
| `place_at` | `:4369` | 放置：On = 物体顶面落座；In = 容器 region 释放（含扶正姿态并入 transit）|
| `articulate` | `:5169` | 开/关关节体（柜门、抽屉）|
| `toggle` | `:5252` | 旋钮 turnon/turnoff |
| `move_to` | `:5331` | 自由移动兜底（走廊现解）|

可复用的规划/感知辅助：`measure_hand` `:337`、`corridor_plan` `:658`、`pregrasp_point` `:1071`、`plan_arm_path` `:1236`、`graspnet_candidates` `:1463`、`graspnet_scene_candidates` `:1476`、`filter_site_config` `:1604`、`analytic_inclined_grasp` `:1836`、`analytic_rim_pinch` `:1916`、`analytic_side_grip` `:2113`、`solve_grasp_target` `:2339`。

## 3. 通用执行底座：`serve` / `goto`

`goto` **不是独立实现**：`tol=None` 时原样转 `serve`；给 `tol` 时构造 `_verify(end)`（`‖end−target‖ ≤ tol`）；`timeout` 直接当 `budget` 传下去。`libero_skills.py:2666-2681`

```python
serve(adapter, target, *, gripper=0.0, k=K_SERVO, verify=None,
      budget=None, target_rot=None, kr=K_SERVO)
```

### 3.1 三级终止判据

1. **到位**（二者之一）：给了 `verify` 就按它判；否则要求
   `d_pos < max(IK_TOL, 3σ)` 且 `d_rot ≤ ROT_TOL`——即**位置残差进入在线测量的噪声包络**，而不是固定死值。`:2612,2619-2620`
2. **停滞 → 按物理证据分类死因**：每 `WIN` 步做一次窗口回归（`slope / se / sigma / qmove / fbase`），若 t 统计量 `< TSTAT` 或窗口闭距 `< 0.25*IK_TOL` 记一次停滞，并分类：
   - 接触力超基线 3σ → `contact_blocked` `:2635-2639`
   - 关节几乎不移动（`qmove < IK_TOL`）→ `reach_limit`（机制记 `ik_unreachable`）`:2642-2645`
   - 连续 `STATIONARY_MAX` 个停滞窗口 → `ik_stalled` `:2649-2651`
3. **显式预算**（仅留给调用方做硬资源限制）：`t > budget` → `slow_budget` `:2657-2659`

### 3.2 为什么"不设预测步数预算"（原文）

> "**不设预测步数预算**：慢而持续的进展必须走完，预算误杀会把物理上 3mm 内的接触事件截断；真实物理止挡由上述统计分类保证终止。" `:2585-2588`

这是本文件最重要的设计决策之一：**用统计判据替代经验预算**，代价是需要 `_ServoStats` 这样的窗口回归设施 (`:2382-2412`)。

### 3.3 关键精度常量

| 常量 | 值 | 含义 |
|---|---|---|
| `IK_TOL` | `0.003 m` | IK/几何"零距离"精度，**兼作接触与落座的零判定** `:204` |
| `ROT_TOL` | `IK_TOL/0.06` | 姿态精度；由名义手部特征尺寸 0.06m 推导（注释说明实测指尖杠杆 ≈0.031m、整手包络 ≈0.125m，取 0.05rad 介于两者，经 30 任务回归验证）`:206-209` |
| `CONF_K` / `TSTAT` / `WIN` / `STATIONARY_MAX` | `3.0` / `2.0` / `30` / `3` | 统计判据四件套（3σ、t 门限、窗口、连续上限）`:220-224` |
| `K_SERVO` / `K_FINE` | `5.0` / `2.5` | 巡航增益 / 精细增益 `:227-228` |
| `EXEC_DQ` | `0.003` | 流式执行点间距，同时决定跟踪包络 `:234` |

## 4. `grasp`：候选枚举与全链闭环

### 4.1 候选来源与"缺页升级"

- 首选 `graspnet_scene_candidates(adapter, obj, top_k=64)`——**用全场景点云**而非单物体点云，让网络看到支撑几何。`:3495`
- **缺页升级门**：按"宽度可达 且 3D AABB 距 ≤ a_reach"复算落目标的候选数，**为 0 时翻页到 `top_k=256` 重取一次**。`:3497-3516`
  实证：`goal:1` 浅碗——64 个候选里**零落目标**，而 rim/side 分析解又被环境门饿死。`:3499`
- 候选是"物体几何签名的纯函数"，缓存于 `mem["__cands__"]`，避免每轮分钟级的全场景推理。`:3481-3493`

### 4.2 三路生成器轮询交错

不逐一穷举，而是让三类来源**轮询**产出（注释：逐一穷举会饿死分析解）：`:3636-3649`

| 生成器 | 内容 | 门 |
|---|---|---|
| `_scene_sols` `:3561` | GraspNet 候选 | 宽度两档：`a_reach = a_open+tip_r`（稳握）/ `a_sqz = a_open+2·tip_r`（大挤压，排后）；抓取点高度带 `z_min = z_bottom + min(h/3, 3·tip_r)`（防夹底推倒）；目标邻近门 `_on_target` |
| `_analytic_sols` `:3593` | `analytic_rim_pinch` × `ANALYTIC_TRIES` 与 `analytic_side_grip` × `ANALYTIC_TRIES` 交错 | 各带 `prior` 方位排斥 |
| `_chain` / `deferred` `:3652-3660` | 闭合深度临界候选 | 排到主生成器穷尽之后再试 |

挤压两档的数值来源有实证：罐/浅碗类 rim 物体候选宽 5.0cm，档① `4.75cm` 夹不到、档② `5.55cm` 可行（`goal:1`）。`:3537-3544`

### 4.3 单候选的 13 道门（`stage` 名清单）

`site_filter` `:3682` → `closure_gate` `:3702` → `closure_marginal` `:3715` → `pregrasp` `:3734` → `approach_clear` `:3747` → `plan_path` `:3761` → `exec_path` `:3787` → `approach_recover` `:3812` → `approach` `:3825/3836` → `reclose` `:3876` → `lift` `:3884/3896` → `grasp_all` `:3914`

其中两个门值得单独说明：

- **`filter_site_config`**（`:1604`）：6D IK 容差内 + 无机器人自接触新对 + robot-env 接触的 env 侧只能是目标或其支撑面；多解中取"非 pad 机器人 geom 对静态 geom **零接触间隙最大**"的解；另有埋入复验（`pad∩目标` 或 `robot∩支撑` 且 `dist < -0.5·tip_r` 判退化）。`:1637-1641,1670-1685`
- **闭爪仿真门**：把手指关节置**闭合限位**后 `mj_forward`，实测 `pad∩目标` 的最深贯穿深度。`.4.3 `:3699-3718`
  - `> -0.25·tip_r` → 判"捏空"拒绝（实证：`object:9` 窄颈瓶**28 连**——闭爪后 pad 零接触，approach+lift 全过但零持力）`:3691-3695`
  - 落在 `(-0.75, -0.25)·tip_r` → 裕度临界，塞进 `deferred` 排后（实证：`object:7/9` 临界候选 lift `f_hold=0`，深贯穿候选 40N 稳持）`:3655-3657`

### 4.4 `mem` 记忆字典（跨 attempt 复用）

| 键 | 语义 |
|---|---|
| `__failed__` | `{obj: [候选 key]}`，key = `center(4位)+R(3位)` 展平；重规划轮内不重复 `:3450,3663` |
| `__enum_sig__` | `{obj: 几何签名}`；**签名不变的重枚举直接快败**（省掉分钟级重枚举）`:3459-3471,3914` |
| `__cands__` | 场景候选缓存（按几何签名）`:3481` |
| `__exec_fail__` | `{obj: {sig, sites}}`，失败过的 rim/side 方位作为 `prior` **持久排斥**；签名变则作废 `:3600-3608` |
| `mem[obj]` | 成功后实测夹持几何 `{rel, half_h, off_xy, grasp_R}`；**`place_at` 依赖它**（缺记忆即 `wrong_state`）`:3902-3907,4388-4390` |

持久排斥方位有实证：`goal:4` 的 8 次 attempt **反复重试相同 2–4 个 rim 方位**烧光预算，其余可行方位从未被遍历。`:3596-3599`

## 5. `articulate` / `toggle`

### 5.1 关节声明与作用点

- `_joint_decl` `:4699-4717`：返回 `{jid,bid,joint,body,jtype,axis,point,range,qpos_adr}`（revolute/prismatic）
- `_handle_geom` `:4749-4782`：名字含 `handle/knob/bar` 者优先；否则 revolute 取**离轴力臂最大**的 geom，prismatic 取沿行进方向最外凸面（背板恒排除）

### 5.2 `_articulate_site` 的三个分支（`:5055`）

| 分支 | 触发 | site 取法 |
|---|---|---|
| **顶落式**（默认，柜门）| revolute | `[h0.x, h0.y, geom顶部 − (b_depth + tip_r)]`，姿态沿用当前腕姿 |
| **knob_tab 顶压角点**（竖直转轴薄耳片）| revolute 且 `|axis_z| > 0.7` 且 `min(obb_x,obb_y) < tip_r` | 取按钮体各 geom OBB **顶面真实角点**，按力臂降序逐个送 IK 预审，取第一个可行；全不可达则退回最大力臂角点。注释说明"用 AABB 角点不行——旋转盒的 AABB 角点在物体外，会悬空 5mm 永不接触" |
| **正面水平捏夹**（抽屉）| prismatic 且 `|D_z| < 0.3` | 竖直优先取最薄截面轴、接近轴沿行进反方向；site 嵌进前缘内 `h0 + D·max(e·D − 0.5·tip_r, 0)`，吸收 OSC 稳态偏置 |

实证依据：`goal:0` 木柜相邻横杆间距 7.4cm、掌心包络 ~9cm → 抽屉必须正面水平捏夹，否则先撞上层把手。`:5062-5065`

### 5.3 `_follow_manifold`：流形跟随的核心

- **流形点**：revolute 用 Rodrigues 绕轴旋转作用点，prismatic 用平移。`:4898-4909`
- **步距换算**（关键技巧）：先求 `s = ‖manifold(q0+1) − manifold(q0)‖`（每单位 q 的手空间位移），再令
  `step_big = max(0.01, 0.004/s)`（"手空间每 tick ≈4mm"）、`lead_max = 0.008/s`。`:4946-4952`
- **θ 驱动 vs q 驱动**（最核心的取舍）：
  - `track_rot=True`（旋钮）用 **θ 驱动**：指令角持续领先实测 q（上限 `lead_max`），手一直"绕行拖拽"；
  - 否则用 **q 驱动**：`θ = q + forward*step_q`（等 q 自己动）。
  原文理由：静摩擦下被驱动件**必须持续滑移**才能被拖动，q 驱动会"互相等死"；而柜门/抽屉这类自由运动件若用 θ 抢跑，会让 aim 甩开末端、低速件永远进不了容差。`:4927-4942`
  - **本轮新增**：非旋钮路径若关节**纹丝不动**（`< 5% travel`），会在同一 leg 内**原地升级为 θ 驱动**再跟一次，然后才判失败。`:5300-5307`
- **到位判据**：`arrive_tol = max(1% 行程, 3σ)`，且**必须双侧判定** `|residual| ≤ arrive_tol`——注释记录单侧判定会把"根本没动"判成到达（`goal:0`：手指没抓住把手，q 不动，每段"假成功"续程循环 300s 空转）。`:4969-4981`
- **姿态跟转**：`Rtrk = R_axis(θ−q0) @ R0`，随 `goto(..., target_rot=Rtrk)` 下发。实证：`goal:7` 27N 捏夹 + θ 驱动下 servo 停在目标前 9.5mm、q 纹丝不动，跟转后接触面才保持平行。`:5033-5036,4942`
- **`goto` 失败 ≠ 未到位**：先按同一到位包络复核，已到位则松爪上抬 0.08m 判成功。`:5037-5046`

### 5.4 编排

- `articulate` `:5169`：`q_goal` 由 `articulation_info` 现解（缺则取 range 极值）；首段跟随后**谓词驱动续程**最多 4 段，每段重解把手作用点、重抓再跟。`:5176-5245`
- `toggle` `:5252`：最多 6 段，每段 `track_rot=bool(knob)`；每段位移 `< 5% travel` 判真失败、`|q_goal−q_end| < 5% travel` 判收尾；全部段结束后按 `0.05·travel` 复核记"分段续转到位"。`:5280-5323`

## 6. 调参常量全表

**几何/精度**：`GRIP_SITE="gripper0_grip_site"` `:202`、`IK_TOL=0.003` `:204`、`ROT_TOL=IK_TOL/0.06` `:209`、`IK_ITERS=80` `:210`、`IK_DAMP=0.05` `:211`、`DE_ITERS=35` / `DE_POP=8` `:212-213`、`ANALYTIC_TRIES=12` `:214`、`IK_RESTARTS=6` `:236`、`PREGRASP_MAX=600` `:237`

**统计约定**：`CONF_K=3.0` `:220`、`TSTAT=2.0` `:221`、`WIN=30` `:222`、`SETTLE=12` `:223`、`STATIONARY_MAX=3` `:224`

**控制/规划**：`VCAP=1.0` `:226`、`K_SERVO=5.0` `:227`、`K_FINE=2.5` `:228`、`RRT_STEP=0.15` `:231`、`RRT_MAX_NODES=6000` `:232`、`SHORTCUT_N=80` `:233`、`EXEC_DQ=0.003` `:234`

**GraspNet 帧/摩擦**：`GRASPNET_FRAME=diag([1,-1,-1])` `:243`、`MU_PINCH=0.5` `:248`、`WALL_COS_MAX=MU_PINCH/√(1+MU²)≈0.447` `:249`、`WALL_GATE_SOFT=0.2` `:250`

**反复出现的约定性数值**（非模块常量）：`0.25·tip_r`（闭合深度下限 / 绝对进展地板）、`0.75·tip_r`（临界裕度上限）、`0.5·tip_r`（埋入界 / 顶压半开口修正）、`0.5 N`（`f_hold` 夹持力门）、`0.30 m`（抬升行程 IK 可达上界）、`4 轮`（`_lift_until_free` 柔度预算）。
`ANALYTIC_TRIES=12` **不可下调**的理由：窄长物体末端挤压夹开合余量小、打分垫底，8 次预算会在前 8 个方位耗尽、唯二可达方位得不到尝试（`object:1` 长条盒实证）。`:214-218`

## 7. 实证结论摘录（12 条）

| # | 结论 | 出处 |
|---|---|---|
| 1 | **RRT 固定种子的失败是伪影**：`object:7` milk 同场景固定种子每次恰在 **235 节点**告败，三个不同抓姿全同 → 概率完备下应换种子重试 | `:1255-1257` |
| 2 | **重枚举极贵**：`object:6` round 2 grasp 秒败后仍有 **300s 耗在重枚举**（rim/side 各 12 方位 × 24 重启 IK ≈ 分钟级）| `:3452-3458` |
| 3 | **不缓存全场景候选就走不完**：`object:0` 可行候选在第 8 个，枚举墙钟 ~510s ≫ 单 attempt 上限 | `:3474-3477` |
| 4 | **学习型 top 页会漏目标** → 升级 `top_k=256` | `:3497-3500` |
| 5 | **挤压抓两档数值来源**（候选宽 5.0cm：4.75 夹不到 / 5.55 可行）| `:3537-3544` |
| 6 | **须持久排斥已失败方位**（`goal:4` 8 次 attempt 反复重试同 2–4 个方位）| `:3596-3599` |
| 7 | **临界闭合深度候选不该占先**（临界候选 `f_hold=0`，深贯穿候选 40N）| `:3655-3657` |
| 8 | **闭爪仿真门不可省**（`object:9` 28 连捏空）| `:3691-3695` |
| 9 | **原位重闭能救回滑脱**：`object:7` milk 同一位形首次 lift `f=0`、重闭后 ok（腱传动手指负载下先伸展、吃掉闭合行程）| `:3866-3871` |
| 10 | **旋钮必须 θ 驱动且跟姿态**：手冻结挤压 300 tick，q 仅蠕动 **0.002 rad**；27N 捏夹 + θ 驱动下 servo 停在目标前 9.5mm | `:4924-4943` |
| 11 | **q 驱动会被"假成功"骗**：`goal:0` 手指未抓住把手时 q 不动，每段假成功续程循环 300s | `:4977-4980` |
| 12 | **闭环停点可能不过谓词**：`goal:0` 木柜开区间宽仅 0.02、目标裕量 0.016 被 3σ 停点吃掉 → 停在阈值上方、谓词永假、8 attempt 同路径死循环 | `:5208-5215` |

另有一组"接近段"的实测陷阱：接近段的**接触证据只认夹具 geom**（`goal:0`：0.2N 的盘子擦碰把接近段停在把手前 7mm，闭爪捏空）；`object:6`：link6 压桌 3.2cm 致 OSC 伺服渐近卡死。`:4743-4748,1684`

## 8. 已知落差与限制

1. **`_closure_holds` 定义了但未被调用**：`grasp` 直接调 `_closure_depth` 并自行比较 `-0.25 / -0.75·tip_r`，`_closure_holds` 只出现在定义处。`:3432-3436,3699-3712`
2. **与 `physics/articulation.py` 同构重复**：本文件自带 `_joint_decl` / `_manifold_point`（同款 Rodrigues）/ `_axis_angle`，**不 import physics**；`physics/articulation.py` 那份 `JointDecl` 反过来没有调用方。合并是待办项（见 [10 物理机制](10_physics_mechanisms.md) §10）。
3. **重枚举的墙钟代价未被预算保护**：`serve` 有意不设步数预算，但"重枚举"发生在 `grasp` 的候选枚举阶段，期间不走仿真步——**步数判据也管不到它**，只能靠挂钟兜底。
4. **RRT 固定种子**：`plan_arm_path(seed=0)` 是为了可复现，但会造成上面第 1 条的"伪影式失败"；两者是张力关系，目前选择保可复现。
