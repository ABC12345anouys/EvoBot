# 10 · 物理机制判别与退路设计

> 对应代码：`darwin/physics/`（9 个模块）
>
> 上游（谁产生证据/机制）：`darwin/skills/primitives/ik_servo.py`、`darwin/agents/libero_skills.py`
> 下游（谁消费机制）：`darwin/agents/runner_dynamic.py`、`darwin/policies/retry.py`、`darwin/agents/reflection.py`、`darwin/agents/libero_planner.py`、`darwin/ipc/agent_learner.py`

## 1. 设计目标

技能**不返回布尔值**，而返回**失败机制**（mechanism）。这一层的存在理由：

1. **没有机制就只能盲试**：代码注释记录了反面教材——"判别子不分类会把 5 次重试全烧错（`spatial:4` vs `spatial:2` 实证）"（`discriminators.py:37-39`）。
2. **失败与成功同等采集证据**：`observables.py` 在技能执行期逐样本观测，失败轨迹同样产出 `Evidence`，用于后面的割与推导。
3. **分层解耦**：把"**是什么坏了**（判别）"与"**该怎么改**（退路：割参数 or 给几何指令）"分开，前者是纯函数，后者可持久化。

三条表（注释里的编号）对应三个模块：**表1** 约束判据（`constraints.py`）、**表2** 机制→退路（`cuts.py`）、**表3** 遥测→机制（`discriminators.py`），另有**表4** 主动探针（`probes.py`）。

## 2. 目录与数据流

| 文件 | 职责 |
|---|---|
| `observables.py` | 证据采集：执行期逐样本观测 → `Evidence`；接触力按 O(nbody) 稀疏采样 |
| `discriminators.py` | **表3**：遥测签名 → 机制二分。纯函数 `Evidence + ctx → Mechanism` |
| `cuts.py` | **表2**：机制 + 证据 → Θ 半空间割（`theta_cut`）或几何指令（`directive`）|
| `derives.py` | 参数推导：参数 = f(场景特征)（timeout / band / straddle / push / 摩擦锥）|
| `probes.py` | **表4**：主动实验（接触/可达/摩擦/几何核对）|
| `posterior.py` | Θ 后验：半空间交 + 置信度 + per-env YAML 持久化 |
| `constraints.py` | **表1**：L1 约束判据（摩擦锥/支撑多边形/力封闭 lite/可达/夹持力/间隙/接近走廊）|
| `articulation.py` | 关节约束类：`JointDecl` + 流形/限位/力对齐三判据 |

```
技能执行 ──逐样本──► Evidence ──► discriminators ──► Mechanism
                                      │
                        ┌─────────────┴─────────────┐
                        ▼                           ▼
                 cuts (theta_cut)            cuts (directive)
                        │                           │
                        ▼                           ▼
                  Θ 后验 (posterior)         几何/路径指令执行器
                  per-env YAML 持久化         (runner_dynamic._directive_blacklist)
```

## 3. Mechanism 枚举与判别依据（`discriminators.py:18-32`，共 10 个）

关键常量：`F_FREE_N = 0.5 N`（自由空间力噪声裕度）、`RATE_EPS = 2e-4 m/步`（位移速率地板）、`EJECT_DIST_M = 0.05 m`（弹射判距）、`reach_limit_band = 0.035 m`。`observables.py:17-22`、`discriminators.py:71`

| 机制 | 判别依据（读什么量）|
|---|---|
| `REACH_LIMIT` | `f_at_end < F_FREE_N` 且 `z_rate_median ≤ RATE_EPS` 且 `above_goal ∈ [0, 0.035]`（自由空间、停在略高处）|
| `CONTACT_BLOCKED` | `f_at_end ≥ F_FREE_N`（真接触）且 `above_goal ≤ 0.035` |
| `BUDGET_SHORT` | `f_at_end < F_FREE_N` 且 `z_rate_median > RATE_EPS`（自由空间里匀速慢进展）|
| `IK_UNREACHABLE` | 三入口：① 无接触+速率为零+`above_goal ∉ [0,0.035]`；② 有力且 `above_goal > 0.035`；③ reason 含 `xy_drift` |
| `NO_GRIP_AIR` | ① 全程无接触（`max(f_trace) < F_FREE_N`）；② `obj_z_follows is False` |
| `FRICTION_SLIP` | 曾接触但力不足；滑移位移 `< EJECT_DIST_M`；`slip_direction is None` 时保守退化 |
| `GEOMETRY_SQUEEZE` | 滑移位移 `≥ 0.05 m`（被弹出）；或放置超时且 `clearance_min ≥ 0.005` |
| `CLEARANCE_VIOLATION` | reason 含 `collision`；或放置超时且 `clearance_min < 0.005` |
| `SUPPORT_OVERFLOW` | ⚠️ **只在枚举与割表里存在，没有任何判别分支会产出它**（声明态）|
| `UNKNOWN` | 所有关键词分支未命中时的兜底 |

**分发**：`discriminate(reason, ev, **ctx)` 按关键词路由——`xy_drift→IK_UNREACHABLE`、`stalled/stall→classify_stall`、`no_grip/grip_fail→classify_no_grip`、`slipped/slip→classify_slip`、`timeout→classify_place_timeout`、`collision→CLEARANCE_VIOLATION`。`discriminators.py:103-124`

**为什么要区分 `FRICTION_SLIP` 与 `GEOMETRY_SQUEEZE`**：注释写明是为避免 `place_vcap` 盲试——"`spatial:9`/`goal:2` place slipped，切这刀后**挤压飞走**走几何修正、**摩擦滑走**走物理修正"，两者量级判据是"瓶类沿爪下滑 <1cm vs 被沿口弹出 >5cm"。`discriminators.py:67-79`

## 4. 退路：两类输出（`cuts.py`）

`Cut(kind ∈ {theta_cut, directive})`，入口 `cut_for(mechanism, ev, *, env, body, f_grip, z_goal)`；单条割函数异常会被吞掉返回 `None`。`cuts.py:20-30,112-128`

### 4.1 `theta_cut`：写入 Θ 后验的不等式（`_CUT_TABLE`）

| 机制 | 割 | 置信度 |
|---|---|---|
| `BUDGET_SHORT` | `timeout_need ∈ [dist/rate×1.3+1, ∞)` | 0.9 |
| `FRICTION_SLIP` | `mu ≤ m·g/F_grip` | 0.85 |
| `REACH_LIMIT` | `reach_shortfall ∈ [0, short×1.5]` | 0.9 |
| 其余 7 个 | **空**（走 directive）| — |

`cuts.py:35-46,49-62,65-72,81-94`
预算割的裕度有实测依据："`spatial:4` 实证需 ~280 步 @0.3mm/步，90 步硬预算必 stall"，故取 1.3 倍。`cuts.py:36-46`

### 4.2 `directive`：几何/路径指令（`_DIRECTIVES`）

| 机制 | 指令 |
|---|---|
| `CONTACT_BLOCKED` | 抓取点移出障碍/沿口 AABB⊕margin，或换 straddle 方向 |
| `NO_GRIP_AIR` | 换候选点／调 straddle 偏置（几何推导，无 Θ）|
| `GEOMETRY_SQUEEZE` | 放置点修正：质心投影入支撑多边形／释放时机提前 |
| `IK_UNREACHABLE` | 目标 ∉ 工作空间投影 → 换路径点（approach 绕行）|
| `SUPPORT_OVERFLOW` | 放置点向支撑多边形内收 |
| `CLEARANCE_VIOLATION` | 提高 clearance margin／绕障路径 |

`cuts.py:96-109`

> 设计意图（注释）：**参数不是所有问题的解**。可达性、几何干涉这类问题改参数无效，必须改几何或路径——`constraints.py:87-89` 记录了 118 次 `ik_unreachable` 空转的教训。

## 5. 参数推导与主动探针

- `derives.py`：参数 = f(场景特征)，而不是手调常数。例：堆叠翻向 `<2cm` 为楔止风险（`spatial:4` 实证）；容器/实心物的软停带分别回落 `0.015 / 0.05`；`goal:5` 实测盘子厚 19.1mm、指尖-TCP 偏置 11.8mm ≈ 盘厚 2/3 → **力封闭不可建**，任务本体是"推"。`derives.py:19-23,64-65,153,176-190`
- `probes.py`：主动实验（接触/可达/摩擦/几何核对）。注意 `friction_probe` 目前是**接口占位**——返回 `None` 表示"μ 未测"，由割路径继续二分。`probes.py:88-99`

## 6. Θ 后验（`posterior.py`）

- 结构：多个半空间的交 + 置信度；按 env 落 YAML 持久化；`skills/skill_config.apply_theta_cuts` 消费 `theta_cut` 求交并写 history。`posterior.py:1-12`、`skills/skill_config.py:190-211`
- **软更新护栏**：只在置信度高于已有记录时才收紧；**矛盾时保宽域并把置信度减半**。注释把"误分类切错方向"列为本节头号风险。`posterior.py:38-44,62-72`

## 7. L1 约束判据（`constraints.py`）

摩擦锥 / 支撑多边形 / 力封闭 lite / 可达 / 夹持力 / 间隙 / 接近走廊，每条都带实证注释：

| 判据 | 实证依据 |
|---|---|
| 可达 | `goal:3/4/5` 共 118 次 `ik_unreachable` 空转 → "参数不是可达性的解" `constraints.py:87-89` |
| 夹持力 | `goal:5` 24 次 `lift_no_grip` → "stop_above/k 给不了更大法向力，参数方向存在但已到顶" `:100-103` |
| 装配间隙 | "间隙 < tol 时直插必失败（`goal:2` 瓶架 marginals）" `:111-113` |
| 接近走廊 | "`goal:3/4` 下降途中 TCP 被障碍楔偏，xy 漂 >15mm——**端点可达 ≠ 路径可达**"；`samples` 取奇数以保证中点被采到 `:125-131` |

## 8. `articulation.py`：关节约束的数学

### 8.1 关节声明与流形

`JointDecl` 用**一种声明**覆盖柜门/抽屉/翻盖/旋钮：关节类型（revolute/prismatic）、世界系轴 `axis`、轴上一点 `point`、限位 `q_range`、`qpos` 地址。

```python
def handle_point(self, q):
    # revolute：Rodrigues 绕轴旋转作用点
    v = h0 - point
    return point + v*cos(q) + cross(k,v)*sin(q) + k*(k·v)*(1-cos(q))
    # prismatic：h(q) = h0 + axis*q
```

`articulation.py:54-62`；世界轴由 `body_xmat @ jnt_axis` 得到，轴点由 `body_xpos + xmat @ jnt_pos`。`articulation.py:65-99`

### 8.2 三判据

| 判据 | signature | 含义 |
|---|---|---|
| `on_manifold_track` | `(p_now, q_now, decl) -> (bool, margin)` | 实测作用点是否还在流形上（阈值 `MANIFOLD_TOL_M=0.015`，门缝量级）|
| `within_limits` | `(q, q_range, margin=0.02) -> (bool, margin)` | 关节是否在限位内 |
| `aligned_wrench` | `(force, decl, apply_point=None, ratio_min=2.0) -> (bool, ratio)` | 力/力矩是否沿驱动方向 |

`articulation.py:111-177`

**一个值得记住的几何洞察**（注释）：力矩参考点取"作用点在轴上的投影"，于是 `τ·axis` 是**与轴上参考点选取无关的不变量**——"柜门把手在任何高度拉，开门力矩都只由水平力臂贡献"。退化情形 `w < 1e-9` 的处理语义也写清了："纯切向拉 = 完美对齐；径向对轴推 = 零驱动，不是对齐"。`articulation.py:159-175`

### 8.3 违例 → 机制

`mechanism_of_violation`：`off_manifold → geometry_squeeze`、`limit → geometry_squeeze`、`jammed → contact_blocked`、`slipped_off → friction_slip`。`articulation.py:181-193`

### 8.4 诚实标注的失效边界

> "已知轴 + 限位 + 单链刚体。**多链折叠门 / 柔性门封 / 变形物不在覆盖承诺内**。" `articulation.py:22-23`

## 9. 机制在系统里被谁消费

| 消费者 | 用途 | 位置 |
|---|---|---|
| `skills/primitives/ik_servo.py` | descend 停滞双窗口确认后**产出**机制；`reach_limit` 在带宽内可被判**接受** | `ik_servo.py:177,184-200` |
| `agents/runner_dynamic.py` | ① 技能未给机制时补判；② 失败步 → `cut_for` 产 `theta_cuts`；③ 步级换参 `retry_params`；④ 成功步 `force_closure_lite` 出 advisory；⑤ directive 执行器（门控 `DIRECTIVE_MECHS`）；⑥ attempt 级取链上**最早**带机制的失败步作因果机制 | `runner_dynamic.py:276-279,388-411,548-582,627-660,1004-1008,1457-1459` |
| `policies/retry.py` | 机制 → 执行参数（步级）：`budget_short`→timeout×2/3/4.5；`contact_blocked`→k/vcap 递减；`friction_slip`→place 时 k↑vcap↓；`ik_unreachable`→k↑+timeout↑；`reach_limit`→**None**（"已被接受准则处理"）| `retry.py:52-111` |
| `agents/reflection.py` | attempt 级 cfg 持久键；`geometry_squeeze` 明确"参数无效，保持 cfg" | `reflection.py:114-166,159-163` |
| `agents/libero_planner.py` | `replan_hint` 机制 → 下次候选参数 | `libero_planner.py:107-122` |
| `ipc/agent_learner.py` | directive 黑名单写入 + 同一套 cfg 适配 | `agent_learner.py:101-117,312` |
| `skills/skill_config.py` | 消费 `theta_cut` 与 Θ 后验求交 | `skill_config.py:186-211` |

**"取最早失败步"的实证依据**：`goal:2` 瓶"先滑脱 = `friction_slip`，后 `place_timeout` = `geometry_squeeze` 的表象"——只取最后一步会归因错。`runner_dynamic.py:1002-1007`
**黑名单双触活**：单次失败的机制标签有噪声（`r14` 实证误杀），故 `BLACKLIST_STRIKES=2`。`runner_dynamic.py:88-92`

## 10. 已知落差与限制（如实记录）

1. **`physics/articulation.py` 目前没有调用方**（除 `__init__.py` 导出）。`libero_skills.py` 里另有一份**同构但独立的**实现：`_joint_decl` / `_read_q` / `_manifold_point`（同款 Rodrigues）/ `_axis_angle` / `_follow_manifold`，**不 import physics**。两者尚未合并——这是真实存在的重复，不是设计意图。
2. **命名落差**：`Mechanism` 枚举的取值是 `no_grip_air`，而 `libero_planner.replan_hint` 判断的字符串是 `"grasp_miss"`；`libero_skills._fail` 产出的机制集合（`ik_unreachable` / `no_candidate` / `perception_fail` / `action_fail` / `timeout` / `slow_budget` / `contact_blocked`）与枚举只有**部分交集**。跨层对齐机制命名是待办项。
3. **`SUPPORT_OVERFLOW` 是声明态**：有枚举与割，但没有判别分支产出它。
4. **`friction_probe` 是接口占位**：返回 `None` 表示未测。
5. **覆盖边界**：多链折叠门、柔性门封、变形物不在承诺内。
