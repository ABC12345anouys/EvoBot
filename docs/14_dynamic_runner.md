# 14 · 动态执行路径设计（runner_dynamic）

> 对应代码：`darwin/agents/runner_dynamic.py`（1503 行）、`darwin/agents/methods.py`、`darwin/agents/chain_registry.py`、`darwin/policies/`
>
> 与 [08 LIBERO 规划与执行栈](08_libero_planning_stack.md) 的关系：08 是 LIBERO 基准的**确定性栈**（BDDL 直接解析成子目标）；本篇是 robopal 侧的**动态执行器**——条件驱动 + 方法库 + 机制闭环，允许自适应与经验回灌。两者并列、各自服务不同基准。
> 与 [02 智能体与执行器](02_agent_runner.md) 的关系：02 讲的是 `ManipulationAgent` 的五步闭环；本篇讲它下面的 `DynamicEpisodeRunner` 具体怎么把一次 episode 跑完。

## 1. 设计目标

1. **条件（objectives）驱动**，不是固定技能序列：`GoalCond` 子类（`JointAtLeast` / `BodyLifted` / `BodyNearSite` / `LiberoGoal` …）声明"什么算达成"，执行器按条件推进与核验。
2. **方法库解耦**：`Method.can_achieve(cond, entry)` 声明"我能达成这类条件"，`make_steps(ctx, cond)` 现场生成 skill 链——所以换任务不必改执行器。
3. **机制闭环**（本文件的核心价值）：技能失败返回**机制**而非布尔值，执行器用它做三件不同的事——步级换参、attempt 级调参、候选黑名单。判别与退路定义见 [10 物理机制](10_physics_mechanisms.md)。
4. **可自适应**：per-env/per-task 的 cfg 可被反思写回；方法库可按经验成功率重排；成功轨迹还能被 forged 成新方法。

## 2. 组件与调用关系

```
                    ┌─────────────────────── policies/ ───────────────────────┐
                    │ context.StepContext  registry(决策点→impl)  spaces(可行域) │
                    │ retry.py(机制→步级参数)  rules.py(抓取候选现状逻辑)        │
                    └───────────────────────────┬────────────────────────────┘
                                                │ propose / retry_params
┌───────────────────────────┐      ┌────────────▼──────────────┐      ┌──────────────┐
│ chain_registry.py          │◄────►│ runner_dynamic.py         │◄────►│ methods.py    │
│ 注册表：内置方法 / YAML /  │      │ DynamicEpisodeRunner      │      │ Method 基类 + │
│ forged 包装与锚点重投影    │      │ attempt 闭环 + 机制消费   │      │ 9 个内置方法  │
└───────────────────────────┘      └────────────┬──────────────┘      └──────────────┘
                                                │
                    ┌───────────────────────────▼────────────────────────────┐
                    │ physics/：discriminate（补判机制） cut_for（Θ割）        │
                    │          constraints（成功后的 advisory） derives（参数推导）│
                    └────────────────────────────────────────────────────────┘
```

依赖方向（已核对）：`runner_dynamic.py:52-55` import `methods.py`；`runner_dynamic.registry` 懒加载 `chain_registry.default_registry()`（进程级单例）`:195-199`；`chain_registry.py:37-44` 反向 import `methods.py` 的内置方法；`policies/rules.py:94-96` 函数内反向 import `runner_dynamic` 取常量（避免循环）。旧执行器 `agents/runner.py:33` 已 deprecated，明确指向 `DynamicEpisodeRunner`。

## 3. 一次 attempt 的骨架

**入口** `run(cfg) -> Dict` `:1316`：

1. cfg 底座合并（per-env skill YAML ← 调用方显式 cfg 优先）`:1318-1329`
2. 建 env、`objectives_from_entry`、障碍注入（LIBERO 走 `auto_obstacles`，`collision_mode` 默认 `"warn"` 灰度）`:1330-1352`
3. `EpisodeLogger` 挂载；`env.reset()` + bounds + RAG `tick/decay`；`_build_fresh_candidates`；`jit_scale = cfg["jit"] `:1353-1380`
4. **attempt 循环** `for attempt in range(self.max_attempts)` `:1386`：
   - `_run_single_attempt(...)`——**纯执行 + 核验，不做反思/自适应** `:1393-1395`
   - 成功 → `agent.reflect(success=True)`、记经验、`_maybe_mutate_success`、break `:1401-1411`
   - `cand is None`（感知无候选）→ 不写反思，直接结束 `:1413-1415`
   - 失败 → `_directive_blacklist(ar)` → 追加 `fail_phases` → `agent.reflect(success=False)` → 记经验 → `registry.reorder_by_experience` → `fail_cat_streak` 累加 → `_adapt_cfg(mechanism=...)` → method 连续失败 ≥3 触发 `_maybe_mutate_failure` `:1424-1458`
5. `finally` 收尾；**fitness** `score = float(success) - 0.02 * attempts` `:1487`

**`_run_single_attempt`（`:857`）内部**：reset → 每次重新观察世界取候选 → `cand = queue.pop(0)` → 抖动（`ang = 2.4*attempt+1.9`，仅非记忆候选）→ **observe 循环**（遍历 objectives，满足者锁定进 `done`，避免 Open↔Close 死循环）→ `_next_steps(...)` 拿方法链 → `_execute_chunk(...)` → 终态核验。

两个细节值得记住：
- **多条件任务**在 `chunk_count>0` 时重新观察、换候选执行下一 chunk `:926-929`；
- **`body0` 透传规则**：只有主物体的首个 chunk 才传 `body0`，其余传 `None`——防止偏心补偿被带偏 `:936-948`，实测依据见 `methods.py:406-413`（`r11 spatial:4` 双重补偿 8/8 失败）。

## 4. 方法体系与选择

| 概念 | 位置 | 说明 |
|---|---|---|
| `Method` 基类 | `methods.py:48-60` | 声明 `name`/`includes_home`/`flavor`/`next`；抽象 `can_achieve(cond, entry)` 与 `make_steps(ctx, cond)` |
| `PlanContext` | `methods.py:8-46` | 提供 `grasp_pt/hover/k_desc/lift_h/timeout_scale/cmn` |
| 内置方法（9 个）| `methods.py:104-589` | `OpenDrawerMethod`、`GraspLiftMethod`、`TransferCartMethod`、`TransferPoseMethod`、`TransferPoseCartMethod`、`LiberoPushMethod`、`IKLiberoTransferMethod`、`ReplayDemoMethod`、`LiberoArticulateMethod` |
| `ChainRegistry` | `chain_registry.py:213-441` | `_order`(list)+`_items`(map)；`select` 按顺序返回首个 `can_achieve`（`exclude` 跳过）；`register` 支持 before/after/**replace**（YAML 同名覆盖内置）；`reorder_by_experience` 按成功率重排 |
| `DeclarativeMethod` | `chain_registry.py:14-210` | 从 YAML 声明 `match`/`steps`；模板变量白名单 `$safe_z/$grasp_pt/$site:/$goal_site/$cond./$cfg./$entry.`（**无 eval**）|
| forged 方法 | `chain_registry.py:384-546` | 见 §5 |

选择顺序：`runner_dynamic._next_steps:427-448` 先看上一方法声明的 `next`（方法编排），否则 `registry.select`；无匹配则 `SkillCreator.create_for` 现场生成 `:418`。

> ⚠️ `methods.py:592-601` 的 `select_method` 是**旧版静态选择**，其 docstring 自注"生产环境应用 ChainRegistry.select"。
> ⚠️ `methods/` 目录当前**只有 `examples/`**（两个 YAML 示例），而 `load_dir()` 明确不加载 `examples/` → **YAML 覆盖层目前是空的**（`chain_registry.py:44`）。

## 5. forged 方法的防退化：锚点重投影

forged 链内嵌的是**锻造时那次 episode 的绝对坐标**，新 episode 里物体位置/抽屉开度已变——"直接重放必然抓空"（`chain_registry.py:384-405`）。故 `ForgedSkillMethod` 把"轨迹快照"升级为"参数化模板"：

| 机制 | 阈值 |
|---|---|
| 抓取段（`descend`/`move_above`）的 `point` 用**当前 body 位置**重投影 | `_GRASP_ACTS`:401-402 |
| 前序同坐标步骤沿链传播 | `< 0.02 m` `:483` |
| 其余步骤按最近锚点匹配 | `< 0.06 m` `:498,504` |
| 首个抓取点重投影后距最近锚点 `> 0.06 m` → **raise 拒绝执行**（判定"锻造时布局与当前环境不符"）| `:500-505` |
| 捕获到的步数 `< 2` → 视为环境不匹配，抛错换方法 | `:530-537` |
| 重投影自身异常 → 不阻断，保守保留原链 | `:543-546` |

## 6. 机制的三条消费路径（本篇最核心）

### 6.1 步级换参（`policies/retry.py`）

`_mechanism_params` `:88-107`，入口 `retry_params` `:51-70`（机制优先；未知机制走通用假设阶梯 `_GENERIC_SEQ` `:39-44`）。倍率按重试次数取三档：

| 机制 | 参数调整 |
|---|---|
| `budget_short` | `timeout ×(2.0 / 3.0 / 4.5)` |
| `contact_blocked` | `k ×(0.8/0.7/0.6)`、`vcap ×(0.6/0.5/0.4)`（被挡要更软更慢）|
| `friction_slip` | place 相：`k ×(1.25/1.56/1.95)` + `vcap ×(0.6/0.36/0.25)`；否则 `vcap ×(0.8/0.64/0.5)` |
| `ik_unreachable` | `k ×(1.25/1.5/1.75)` + `timeout ×(1.5/2.0/3.0)` |
| `reach_limit` | `None`——注释写明"已被接受准则处理，不该到这" |

边界：步级重试预算 `STEP_RETRY_BUDGET=5`（**不消耗 attempt**），可重试白名单 `RETRYABLE = {move_above, descend, ik_servo, lift, carry, place, pose_*}` `:20,26-30`。

### 6.2 attempt 级调参（`agents/reflection.py`）

`_mechanism_adapt` `:113-166` 与步级**同源同向**，但作用在持久 cfg 上：`budget_short`→`timeout_scale`；`contact_blocked`→`k/vcap` 降；`friction_slip@place`→`place_k↑/place_vcap↓`、`@grasp`→`stop_above↓/k↑`；`ik_unreachable`→`k↑/timeout_scale↑`；**`geometry_squeeze`→不调参只出 note**（`reflection.py:159-163`，"参数不是几何的解"）。调用点 `runner_dynamic.py:1264-1279`。

### 6.3 候选黑名单（directive）

门控常量：`DIRECTIVE_MECHS = {no_grip_air, geometry_squeeze, contact_blocked, ik_unreachable}`、`GRASP_PHASE_ACTIONS = {move_above, descend, ik_servo, pose_*}`、`BLACKLIST_XY_TOL = 0.02` `:276-280`。
`_directive_blacklist(ar) :379-411`：机制命中 ∧ 失败动作在抓取相 ∧ 候选 xy 可得 ∧ 有写权限 → 登记候选 xy，**双触活** `BLACKLIST_STRIKES=2` `:92`。

**跨进程单一来源**：模块级 `blacklist_read/register/is_hit` `:95/114/137`——sim 侧**只读**（`blacklist_write=False`，`ipc/sim_worker.py:96-101`），agent 侧**写**（`ipc/agent_learner.py:107-118`）。候选生成时过滤 `:239,846-850`，且**黑名单永不清空候选树**（全灭时回退到约束排序第一名）。

### 6.4 与 `physics/` 的调用点

| 调用 | 作用 |
|---|---|
| `_discriminate_result` → `discriminators.discriminate` + `observables.Evidence` `:691-720`（调用点 `:549-554`）| 技能没给机制时**现算** |
| `_theta_cuts` → `cuts.cut_for` `:723-753`（调用点 `:571-577`）| 失败步产出 `theta_cut` 写入结果 |
| `_constraint_check` → `constraints.force_closure_lite` + `posterior.body_mass` `:630-658` | lift/carry 成功后的 **advisory** 警告 |
| `_fail_ctx` `:661-689` | 现算判别上下文（`slip_direction` / `obj_z_follows`）|

## 7. `policies/` 目录职责

| 文件 | 职责 |
|---|---|
| `context.py` | policy 输入 `StepContext`（step/env/body/entry/cfg/attempt/retry/features/fail_history）|
| `registry.py` | 决策点 `(step)` → 多 `impl` 的可插拔注册表；`propose` 按 `cfg["<step>_policy"] → _DEFAULTS → "rule"` 选择，提案经 `ParamSpace.clip` 裁剪 |
| `spaces.py` | `ParamSpace`：参数可行域（`PARAM_SPEC` 范围 ∩ anchor 邻域）|
| `retry.py` | 步级换参（见 §6.1）|
| `rules.py` | 抓取候选的现状逻辑搬运：容器纯 Y straddle 偏置、实心物 GraspNet xy + z override、极薄包边 |

## 8. 与 `agent_loop.py` 的差异

`agent_loop.py` 是**任务无关的通用闭环**（生成器）：感知 → `planner.plan` → 逐步执行（`expect` 已满足则跳过）→ 谓词验证 → `planner.replan` 重规划。它**没有**候选生成/筛选、Method registry、机制判别、步级重试、黑名单、经验库、per-env cfg、日志落盘；技能来自另一套 `robodojo_skills.SKILLS`。`runner_dynamic` 则相反：非生成器、每 attempt `env.reset`、条件驱动 + 方法库 + 机制闭环 + 参数自适应 + 经验/变异，并支持 `prepare_episode/teardown_episode` 的 IPC 常驻模式。两者**互不调用**（`agent_loop.py:11-15` 只 import predicates/robodojo_skills/world_state）。

## 9. 实证结论摘录（注释内固化）

| # | 结论 | 出处 |
|---|---|---|
| 1 | **单次失败即拉黑是误杀器**：`goal:3` 一次 `geometry_squeeze@ik_servo` 曾杀光全部候选 → `no_candidate` 空转 7 attempt（`r14`）→ 改为双触活 | `runner_dynamic.py:88-92` |
| 2 | **失败分类的词表顺序是语义**：`place_failed` 须在 `timeout` 前；`stall` 须在 `timeout` 前（stall 含 "timeout" 子串但对策相反：加大绕障余量而非提速）；`descend_stalled` 归 `grip_failed`（对策提 `k_descend`，绝不能落入 timeout 反向降 k）| `:54-66` |
| 3 | **四场失败的死因在候选空间不在参数空间**（`goal:2/3/4/5`）；`goal:3/4` 的 118 次 `ik_unreachable` 是下降途中被邻接物楔偏——"端点可达 ≠ 路径可达" | `:266-274` |
| 4 | **拉黑的因果必须对准候选本身**：place 相失败（`goal:2/8` 滑脱）不拉黑抓取点，否则误杀好候选 | `:382-386` |
| 5 | **黑名单永不清空候选树**：全灭时回退约束排序第一名 | `:843-846` |
| 6 | **attempt 级因果机制取链上最早的失败步**：`goal:2` 瓶先滑脱（`friction_slip`），后面的 `place_timeout` 只是 `geometry_squeeze` 表象 | `:1001-1007` |
| 7 | 容器偏心比的**死亡螺旋与硬地板**：`spatial:9` 比值 0.70→0.52 螺旋；`ratio<0.6` 插指进不了容器侧壁、8/8 夹空 → 0.62 硬地板 | `reflection.py:220-230` |
| 8 | BDDL `On` 阈值附近**必须加滞环**：落点 xy 在 0.024~0.034 抖动时不加滞环会让 `place_k/release_offset` 交替反向、resume 循环无限往返（`goal:8/9`）→ 阈值 0.030、带宽 0.008 | `reflection.py:86-106` |
| 9 | **参数到顶必须换候选**：`goal:5` 共 24 次 `lift_no_grip`，"`stop_above/k` 已给不出更大法向力" | `physics/constraints.py:98-104` |
| 10 | 物体被推离工作区的 reset 阈值是 **robopal 桌面系（z≈0.46）**；LIBERO 坐标系不同（z≈0.8+）故该分支跳过 | `runner_dynamic.py:1459-1470` |
| 11 | `LiberoPushMethod` 的 `goal:5` 探针证据：plate 厚 19.1mm、指尖-TCP 偏置 11.8mm、demo 155 步夹爪全程张开 → **任务是推不是抓** | `methods.py:237-247` |
| 12 | 变异边界：只对 `DeclarativeMethod` 变异（内置 Python 方法不可变异），需至少成功 2 次、变异体总数 `< 5`（防方法库膨胀）| `runner_dynamic.py:1231-1239` |

## 10. 已知限制

1. **单文件过大**：`runner_dynamic.py` 1503 行，且 `_run_single_attempt` 同时承担"执行 + 核验 + 机制选择"三种职责，是后续拆分的候选。
2. **YAML 覆盖层为空**：`methods/` 下只有 `examples/`（不自动加载），所以"不动 Python 动态改链"的能力**尚未被使用**。
3. **旧入口仍在**：`methods.select_method` 是旧静态选择器，与新 `ChainRegistry.select` 并存（自注"生产应用后者"）。
4. **fitness 惩罚项是线性的**：`score = success − 0.02×attempts`——鼓励"少试即成功"，但也会让"尝试多才成功"的策略在档案中排名偏低（这与 [13 进化系统](13_evolution_map_elites.md) 的分箱维度共同决定搜索行为）。
