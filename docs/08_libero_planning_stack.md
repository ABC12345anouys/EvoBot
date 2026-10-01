# 08 · LIBERO 规划与执行栈设计

> 对应代码：`darwin/agents/task_spec.py`、`darwin/agents/libero_planner.py`、`darwin/agents/libero_runner.py`
>
> 与 [02 智能体与执行器](02_agent_runner.md) 的关系：02 描述的是 robopal 栈（`agent.py` / `runner.py`，靠 LLM 决策 + RAG 记忆）；本篇描述的是 **LIBERO 基准专用的另一套编排栈**，两者并列、互不依赖。
>
> LIBERO 栈的规划取舍（2026-10-01 起）：**BDDL 是形式化规格，LLM 只做一次「谓词级分解」并冻结进仓库**——把任务定义物化成自描述 YAML（谓词表带 id + 隐式前置候选 + 词表 + 夹具属性 + 可用技能约束 + 确定性参考序），LLM 只决定**执行顺序**与**是否插隐式前置**，不选技能、不造谓词、不改参数；产物冻结在 `darwin/skills/configs/task_specs/`，**运行期零 token、完全确定**。确定性拓扑排序退为 validator / fallback / 参考序。

## 1. 设计目标

1. **分解一次、冻结重放**：LLM 在**离线生成阶段**读形式化规格做谓词级分解（排序 + 隐式前置），产物落成 `skills/configs/task_specs/<suite>_<idx>.yaml` 并提交仓库；运行期 `load_or_parse()` 只读 YAML，**不调 LLM**。于是"同一任务每次执行的计划完全相同"仍然成立，同时顺序不再是启发式规则硬编码的。
2. **LLM 的自由度被压到最小、且可核查**：LLM 只能排列给定 id 的顺序、只能从 `implicit_candidates` 里挑要插入的前置；`validate_order()` 是纯函数（id 已知、不重复、不漏、add 只取候选），校验不过就回退确定性解析并把原因写进 YAML 的 `llm.error`。见 §3.3。
3. **与任务名零耦合**：只做「谓词种类 → 技能」的映射（`place→grasp+place_at`、`articulate→articulate`、`toggle→toggle`），不做 `if task == "..."` 之类的任务特判。
4. **失败机制驱动重规划**：技能返回的是**机制**（`no_candidate` / `contact_blocked` / `ik_unreachable` …）而不是布尔值，重规划按机制决定下一个参数（机制表见 [10 物理机制与判别](10_physics_mechanisms.md)）。
5. **可长跑**：单任务 8 次 attempt × 数千仿真步，需要台账（ledger）断点续跑与两种超时判据。
6. **实测驱动**：代码注释里保留了大量 `goal:N` / `object:N` 的实证结论，本文件末尾汇总。

## 2. 三段式管线

```
BDDL 文件
   │  task_spec.build_formal             物化"任务定义"（谓词表带 id + 隐式前置候选 + 词表 + 技能约束）
   │  ──► LLM（离线，脚本 scripts/gen_task_specs.py）只做谓词级分解
   │  ──► validate_order 校验 ──► 冻结 skills/configs/task_specs/<suite>_<idx>.yaml
   │  task_spec.load_or_parse            运行期只读冻结 YAML（零 token）；缺失时回退 parse_bddl_spec
   ▼
[子目标序列]
   │  libero_planner.plan_subgoal        每个子目标 → 技能序列
   ▼
[技能计划]  ──►  libero_skills.grasp / place_at / push_to / articulate / toggle
   │                    │
   │                    └─ 失败时返回 mechanism ──► libero_planner.replan_hint ──► 下一轮参数
   ▼
libero_runner.run_task                   attempt 闭环 + 台账 + 视频
```

## 3. `task_spec.py`：BDDL → 有序子目标

### 3.1 谓词 → 子目标大类

| 谓词 | kind | 说明 |
|------|------|------|
| `On` / `In` | `place` | 抓 `object` 放到 `target`（物体或 region）|
| `Open` / `Close` | `articulate` | 开合夹具（抽屉/柜门/微波炉）|
| `Turnon` / `Turnoff` | `toggle` | 旋钮/按钮 |
| `NextTo` | `unsupported` | 已识别但 v1 不支持 |

- 匹配用 `_PRED_RE`，大小写照 BDDL 实际写法（`Turnon` 是**小写 n**）；`task_spec.py:35-37`
- 仅在 `(:goal` 之后匹配，避免命中 `(:init ...)` 里的同名词
- 未知谓词由 `_ANY_PRED_RE` 兜底发现（剥掉 `And/Or/Not` 逻辑词），并入返回值的 `unsupported`，**不抛异常**；`task_spec.py:39,89-92,110-122`
- 该模块还有一处硬编码枚举：`_SUFFIXES`（`_contain_region` / `_cook_region` / `_top_region` / … / `_region` / `_site`）用于把 region 名归一到"夹具主键"；`task_spec.py:46-63`

### 3.2 隐式 Open 补全

`place` 进"可开合 region"（`_ARTICULATED_HINT = (drawer|cabinet|microwave)`）而 goal 里没有显式 `Open` 时，自动补一个 `implicit=True` 的 Open 节点，`anchor = place.anchor - 0.5` 以保证排在该 place 之前；`task_spec.py:42,65-67,136-151`
（注意 `stove/cook` 是另一组：`_COOK_HINT`，用于把 `Turnon` 排到对应 place 之前。）

### 3.3 两条分解路径

#### 路径 A（主）：LLM 谓词级分解，离线一次、冻结入库

`build_formal()` 把 BDDL goal 物化成一份**自描述的任务定义**（这就是"把任务定义写到 skill 里"的落点，产物在 `darwin/skills/configs/task_specs/`）；`decompose_spec()` 把它交给 LLM，只要它做谓词级分解。

| 环节 | 代码 | 说明 |
|---|---|---|
| 物化任务定义 | `build_formal`（`task_spec.py:331-381`）| 谓词表带稳定 id（`p0`/`p1`…）、隐式前置候选、对象/目标词表、夹具属性（`articulated` / `cook`）、可用技能约束、**确定性参考序** |
| 提示词 | `_SYSTEM_PROMPT`（`:307-329`）| 只让 LLM 输出 `{"order": [...], "add": [...], "rationale": "..."}` |
| 解析容错 | `_extract_json`（`:400-411`）| 先剥 ```json 围栏，否则取首个 `{` 到末个 `}` |
| **校验** | `validate_order`（`:413-437`）| 纯函数：`order` 必须是全部 id 的一个排列（不重不漏，允许 `skip`）；`add` 只能取 `implicit_candidates` 里已有的 id。**违反即判失败** |
| 展开 | `materialize`（`:439-462`）| 按 `order` 展开子目标，把 `add` 的隐式前置插到**同夹具**子目标之前；没匹配上的插到最前 |
| 编排 | `decompose_spec`（`:464-513`）| 失败/不可用 → **回退确定性解析**，原因写进 `spec["llm"]["error"]` |

**LLM 被刻意限制的自由度**：只能排列给定 id 的顺序、只能从候选里挑要插的前置。不得新增/删除/改写谓词与对象名，不得选技能、不得给参数。这样"LLM 分解"既可核查（`validate_order` 是纯函数），又不会把基准结论建立在不可复现的自由生成上。

离线生成：`scripts/gen_task_specs.py`（产物 `skills/configs/task_specs/<suite>_<idx>.yaml`，提交仓库）。运行期 `load_or_parse()` 只读 YAML——**零 token、完全确定**。

#### 路径 B（参考序 / fallback）：依赖边 + Kahn 稳定拓扑

`parse_bddl_spec()`（`:139-157`）是纯确定性实现，**不调 LLM**：

- 依赖边只对 `place` 建立：`Open(key) → place`、`place → Close(key)`；若 place 的 target 命中 `_COOK_HINT`，再建 `Turnon(key) → place`；`task_spec.py:159-231`
- `Kahn` 每轮从入度零节点中取 `anchor` 最小者（`anchor` = BDDL 原始序号，隐式节点为 `原序号-0.5`），从而在满足依赖的前提下尽量保持原文顺序
- 无依赖的子目标保持 BDDL 原序（注释明确：作者按执行顺序书写）
- 无环假设；保险起见若某轮无 ready 节点则强制取最小 anchor（避免死循环）

它现在是三重角色：**fallback**（LLM 不可用/校验失败时）、**参考序**（写进 YAML 的 `reference_order`，供 diff 与回归）、**validator 的依据**（隐式前置候选就是从它的 `_ARTICULATED_HINT` 规则来的）。

### 3.4 冻结产物与实测（2026-10-01）

30 个任务全部由 `deepseek-v4-pro` 分解成功（`source: llm`，0 回退）；**28/30 与确定性参考序完全一致，2 处不同，且这 2 处正好是名字启发式瞎猜的地方**：

| 任务 | BDDL goal | 确定性规则（`_ARTICULATED_HINT` 只匹配名字） | LLM |
|---|---|---|---|
| `goal:2` / `goal:4` | `On(x, wooden_cabinet_1_top_side)` | 名字含 `cabinet` → 补隐式 `Open(wooden_cabinet_1_top_side)`，**但那是柜子的顶面、不是抽屉/柜门** | 不插。理由："把碗放到柜子顶面，不涉及打开柜门或抽屉，无需插入隐式前置" | 
| `goal:3` | `In(bowl, wooden_cabinet_1_top_region)` | 补隐式 Open ✅ | 补隐式 Open ✅（理由："碗放入柜子顶部抽屉前必须先打开该抽屉"）|

也就是说 LLM 的增益是**语义判断**（该 region 到底可不可开合），而不是重排顺序；其余 28 个任务它只是复述了同样的顺序。这一点应当如实记录：**这不是"LLM 会规划"的证据，而是"名字规则会瞎猜、LLM 能看出这是顶面"的证据。**

需要说清楚代价与实情：这个多余的 `Open` **并不会让任务失败**——`articulate` 把 `wooden_cabinet_1_top_side` 解析到了柜体关节、那一步真的执行了，`goal:2`/`goal:4` 在台账里仍是通过的，只是**多花一步**。省下这一步能带来多少收益，**尚未做 A/B**（见 §6 第 5 条）。

## 4. `libero_planner.py`：子目标 → 技能

### 4.1 映射表

| 子目标 kind | 技能序列 |
|---|---|
| `place` | `grasp(obj)` → `place_at(obj, target, predicate)` |
| `articulate` | `articulate(target, direction)` |
| `toggle` | `toggle(target, direction)` |
| `unsupported` | `[]`（runner 记录并跳过）|

方向表 `_DIRECTION = {Open:open, Close:close, Turnon:turnon, Turnoff:turnoff}`；`libero_planner.py:22-23,26-45`

### 4.2 析取规划（第二支）

`plan_subgoal_alternative()` **只对 `place`** 提供替代物理实现：`push_to(obj, target, predicate)`。
触发条件写在调用方：同一子目标此前有过 `grasp` 返回 `no_candidate`（抓取候选空间实测耗尽，属几何事实而非执行失败）。`libero_planner.py:47-59`

### 4.3 重规划 hint 映射

`replan_hint(failures, mechanism)`：

| mechanism | 返回 | 意图 |
|---|---|---|
| `grasp_miss` / `ik_unreachable` | `{cand: failures, strategy: 0}` | 换几何抓取候选（"目标点邻域内找可行点"是几何问题的标准响应）|
| 其他 | `{cand: min(failures, 0), strategy: 0}` | 世界已被动作改变，用新感知重跑同一子目标 |

> ⚠️ **已知坑**：`min(failures, 0)` 恒为 0，即"其他机制"下每轮参数完全相同。而 **`push_to` 根本不接受 `cand/strategy`**（见下表），所以走替代分支时后续轮次是逐字重复——实测 `goal:5` 与 `object:1/6/9` 各连挂 4 轮。当前对策见 §5.1。

hint 的分发（`execute_step`）：`grasp` 接 `cand`；`articulate` / `toggle` 接 `strategy`；`place_at` / `move_to` / `push_to` **两者都不接**。`libero_planner.py:85-105`

### 4.4 子目标满足判定

`subgoal_satisfied()` 的判定顺序（与 BDDL `_check_success` 同口径）：

1. `adapter.eval_subgoal(...)`：place 传 `(pred, object, target)`，其余传 `(pred, target)`；
2. 抛 `KeyError`（典型：隐式 `Open` 不在 goal_state 中）→ 回退关节夹具 `adapter.fixture_open/close(target)`；
3. 其他异常 → `False`；函数末尾兜底 `return False`。

`libero_planner.py:61-83`

## 5. `libero_runner.py`：attempt 闭环与长跑支撑

### 5.1 attempt 循环

```
for attempt in 1..max_attempts(默认 8):
    np.random.seed(attempt)          # 确定性（见 §6 与 07 感知篇）
    adapter.reset()                  # 回到该任务 demo_0 的固定初始状态
    adapter.step_budget = 7000       # 确定性截断判据
    signal.alarm(300)                # 挂钟兜底
    try:    run_episode(...)
    except AttemptStepLimit:  -> fail=step_budget>Nsteps
    except _AttemptTimeout:   -> fail=attempt_timeout>Ns
    except Exception:         -> fail=exception: ...
    ok = result.success AND adapter.check_success()   # 双闸门
    写台账；ok 则返回
```

- 三类异常都会先 `snap_dump_last(...)` 留现场再落失败结果；`libero_runner.py:251-269`
- **双闸门**：子目标全满足但 BDDL 终态为假时，会追加 `note` 说明，并**判失败**；`libero_runner.py:270-273`
- 替代分支失败即收尾（不烧剩余重规划轮次）：`libero_runner.py:197-201`

### 5.2 双判据超时（主判据确定性、挂钟兜底）

| 判据 | 参数 | 默认 | 作用 |
|---|---|---|---|
| **仿真步数**（主）| `--attempt-steps` | **7000** | 确定性截断，与机器负载无关 |
| 挂钟（兜底）| `--attempt-timeout` | **300s** | 只在"不走步但卡住"（如感知/规划卡死）时才可能先触发 |

标定依据（注释内固化）：`spatial:4` 一次完整失败尝试 = **1605 步 / 88.2s**；前 ~30s 是环境构建 + GraspNet 感知，**不走步**；伺服段约 32 步/秒，故旧 300s 对应约 8500 步，取 7000 作为预算。`libero_runner.py:328-333`

> 为什么需要两个：步数判据管得住"一直在走步只是跑太久"的轨迹，但**管不住"不走步的卡死"**——挂钟必须留着兜底。代价是后者的截断点仍随负载漂移。

### 5.3 台账（ledger）结构

```jsonc
{
  "tasks": {
    "libero:libero_goal:7": {
      "success": true,
      "ts": "2026-09-30T17:12:04",
      "attempts": [
        { "attempt": 1, "success": true, "fail": null, "steps": 1893, "secs": 63.3,
          "fail_mech": {"contact_blocked": 2},          // 非 ok 步的机制计数
          "trace": [ {"skill": "toggle", "ok": false, "reason": "...", "mech": "contact_blocked"} ]  // 末尾 12 步
        }
      ]
    }
  }
}
```

- 路径 `logs/libero_agent_progress.json`，**原子写**（`.tmp` + `replace`）；`libero_runner.py:43,59-75`
- 设计意图：原来 `history` 跑完即弃，失败后无法定位是哪一步、哪种机制；现在压缩落台账。`libero_runner.py:274-293`

### 5.4 队列与断点续跑

- `--tasks libero:<suite>:<idx>,...`（不以 `libero:` 开头会自动补前缀）；`--suites` 与 `--tasks` 二选一，`--suites` 留空默认跑 spatial + object + goal；`libero_runner.py:343-350`
- 非 `--force` 时从队列里过滤掉台账中 `success` 为真的任务 → **断点续跑**；每跑完一个任务立即写台账；`libero_runner.py:352-355,365-368`
- 退出码 `0` 表示队列全部通过，否则 `2`（`scripts/run_libero_agent.sh` 就靠这个决定是否继续 while 循环）；`libero_runner.py:378`

## 6. 实测结论汇总（代码注释内固化）

| 结论 | 出处 |
|---|---|
| `object:6`：跨 attempt 的机制级失败记忆（`mem/__failed__`）必需——同序列每 attempt 从头重复，300s 只够试 2–3 个候选，永远到不了可行方位 | `libero_runner.py:119` |
| `goal:4`：最后一轮只验证不执行——否则末轮达成的子目标等不到复核，明明全满足却报"轮次耗尽" | `libero_runner.py:144-146` |
| `goal:5` / `object:1,6,9`：替代分支只有一条 `push_to` 且不接 `cand/strategy`，后续每轮参数逐字重复连挂 4 轮 | `libero_runner.py:197-201` |
| 同一任务同一初始状态**结果会变**：一次 GraspNet 给 0 个候选、另一次给多个；根因是感知里 `np.random.choice` 用了**未播种的全局 RNG**，对策是按 attempt 序号播种 | `libero_runner.py:236-241` |
| 标定：`spatial:4` 一次完整失败尝试 = 1605 步 / 88.2s | `libero_runner.py:328-333` |
| LIBERO `agentview` 图像上下颠倒，录制需翻转 | `libero_runner.py:79` |
| `reflection.py` 侧的调参护栏（滞环/硬地板/不调参清单）另见其注释 `reflection.py:87-89,112-113,197-198,223-226,257-258` | — |

## 7. 已知限制

1. **可复现性只做到一半**：步数判据 + 按 attempt 播种消除了两个明确的非确定源，但同一任务同一初始状态**仍会给出不同结果**——剩余方差来自 GraspNet 的 GPU 前向（已排除挂钟超时、CPU 线程调度、RRT 种子）。详见 [07 感知](07_perception.md) 与 [11 环境适配](11_envs_libero_adapter.md)。
2. **`replan_hint` 对非几何机制不产生新参数**：除"替代分支失败即收尾"外，没有更细的升级策略。
3. **`unsupported` 谓词（如 `NextTo`）不执行**：只记录不报错。
4. **冻结产物是"快照"而非"自动同步"**：`skills/configs/task_specs/*.yaml` 由 `scripts/gen_task_specs.py` 离线生成后提交。BDDL 或 `_ARTICULATED_HINT` 规则变了，**需要重跑脚本**才同步；目前没有 CI 校验"YAML 与当前 BDDL 是否一致"（`reference_order` 字段可用于做这种校验，但尚未接线）。
5. **LLM 分解尚未在基准上做 A/B**：`goal:2` / `goal:4` 的确定性路径会多插一个 `Open(wooden_cabinet_1_top_side)`；`articulate` 只在 `adapter.articulation_info(target)` 抛 `KeyError` 时失败（`libero_skills.py:5174-5177`），而这两项的台账 trace 里该步**没有失败机制**，说明适配器把它解析到了柜体关节、那一步**真的执行了**（不是 no-op）。台账显示这两项仍是通过的（`goal:4` attempt 1 / 194.7s，trace = `articulate → grasp → place_at`）。
   所以这是"**多花一步但仍通过**"，不是"失败被容忍"。去掉它能否省时间/提通过率**尚未验证**，需要重跑才能给数字。
6. **挂钟兜底仍是非确定源**：只在"不走步的卡死"场景才可能触发。
