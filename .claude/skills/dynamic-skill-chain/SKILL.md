---
name: dynamic-skill-chain
View and edit the darwin-bot call chain, objective-to-method mapping, via chainctl and YAML methods. Use for adding task methods, changing skill steps, wiring sites or joints, or debugging which skills ran. Not for primitives or robopal internals.
---

# Dynamic Skill Chain（调用链动态增删改查）

darwin-bot 的任务执行是 **observe → select method → render steps → execute** 的规划闭环，
不是写死的任务名 if-else。本 skill 教你如何查看和动态修改「目标条件 → 方法 → skill 步骤」
这条调用链。所有操作在 conda 环境 `darwin` 中执行，仓库根目录运行。

## 架构地图（先读这个）

| 文件 | 做什么 |
|------|--------|
| darwin/agents/runner_dynamic.py | 条件谓词 GoalCond（joint_ge / body_lifted / body_near_site）、Method 基类、内置 Python 方法、observe-plan-execute 闭环、EpisodeLogger |
| darwin/agents/chain_registry.py | list+map 注册表 ChainRegistry、YAML 声明式方法 DeclarativeMethod、$模板变量白名单解析 |
| darwin/agents/methods/ | YAML 动态方法目录，启动时自动扫描 *.yaml；同名覆盖内置方法；examples/ 不加载 |
| scripts/chainctl.py | 注册表的确定性 CLI：list/show/validate/add/remove/disable/enable |
| scripts/probe_dynamic_runner.py | 无渲染验证脚本（drawer_place，约 1 分钟），改完必跑 |
| logs/episodes/*.jsonl | EpisodeLogger 输出；sample 每 10 控制步一条，skill 边界强制快照 |

执行数据流：

```
benchmark entry
  └─ objectives（声明式目标条件，或旧 mode 自动适配）
       └─ 每轮 observe：cond.measure(env) / cond.check(env)
            └─ registry.select(cond)   # list 顺序匹配第一个 can_achieve
                 └─ method.make_steps(ctx, cond)   # YAML 则现场渲染 $变量
                      └─ skill.execute(env, **params)   # primitives 原语
```

注册表是 `_order: list[str]` + `_items: dict[name, Method]`：list 定优先级，
map 做 O(1) 定位；register/unregister/reorder 是运行时 CRUD，YAML 文件是持久化层。

## 决策树：你要做的改动属于哪一种

### A. 只是排查"哪里有问题 / 跑了哪些 skill"

```bash
python scripts/chainctl.py list                    # 当前条件→方法映射与来源
ls -t logs/episodes/ | head                        # 最新 episode 日志
```

JSONL 关键事件：`plan`（为哪个条件选了哪个方法、渲染出的链）、`skill_end`
（ok/reason/min_clearance/coll_pair + 当时 TCP/物体位置/关节角）、`condition`、
`verify`、`episode_end`。失败时顺着 plan → 第一个 ok=false 的 skill_end 定位。

### B. 改一个方法的 skill 步骤 / 接线（最常见，首选 YAML，不改 Python）

适用：调步骤顺序、换 skill 参数、把新 site/关节接进来、同名替换内置方法。

```bash
# 1. 先看现有定义（内置方法看 runner_dynamic.py 同名 Method 类；YAML 直接 show）
python scripts/chainctl.py show transfer_cart
# 2. 从 examples 拷贝一份改，或新建 my_method.yaml
cp darwin/agents/methods/examples/transfer_cart.yaml /tmp/m.yaml
# 3. 校验语法/模板/匹配
python scripts/chainctl.py validate /tmp/m.yaml
# 4. 安装（同名即覆盖内置执行函数；--force 覆盖已存在）
python scripts/chainctl.py add /tmp/m.yaml --force
# 5. 必跑无渲染验证
python scripts/probe_dynamic_runner.py
# 6. 看最新 JSONL 的 plan/verify，确认渲染参数与结果
```

YAML 方法 schema 速览（完整字段与 $变量表见
[references/method-yaml-spec.md](references/method-yaml-spec.md)）：

```yaml
name: transfer_cart            # 必须与文件名一致；与内置同名 = 覆盖
match:
  kind: body_near_site         # joint_ge | body_lifted | body_near_site
  flavor: cart                 # body_near_site 才有：cart | pose
include_home: false            # 链首是否自带 home（默认 false，引擎按 flavor 补）
steps:
  - skill: move_above
    params:
      point: "$grasp_pt"       # $变量在规划当下从 env/cond/cfg 渲染
      hover: "$cfg.hover"
```

常用 $变量：`$safe_z`、`$grasp_pt(.x/.y/.z)`、`$site:<名字>(.axis)`、
`$goal_site(.axis)`（条件 site 的现场坐标）、`$cond.<属性>`、`$cfg.<键>`。
`actor` / `grip_site` 自动注入，不要写进 YAML。**没有 eval**，只有白名单取值；
需要新变量先在 chain_registry.py 的 resolve_var 里加。

### C. 需要一种全新的目标判定（现有谓词表达不了）

这时才写 Python，且只加"条件"，不要在执行端加任务名分支：

1. 在 darwin/agents/runner_dynamic.py 仿 BodyNearSite 加一个 GoalCond 子类
   （实现 measure(env) 和 check(env)），kind 用唯一名；
2. 在 _COND_BUILDERS 注册 kind → 构造器，使其能从 entry 的
   `objectives: [{kind: ..., ...}]` 声明；
3. 对应方法优先写 YAML（match.kind 填新 kind）；确实要复杂逻辑才写 Method 子类
   并在 chain_registry.py 的 default_registry() 注册；
4. benchmark entry 用 objectives 列表声明任务（不必动 mode 适配层）；
5. 跑 probe 验证。

### D. 启用/禁用/删除动态方法

```bash
python scripts/chainctl.py disable transfer_cart   # 改名 .disabled，新进程回退内置
python scripts/chainctl.py enable  transfer_cart
python scripts/chainctl.py remove  transfer_cart   # 仅 YAML；内置请用 disable 或 YAML 覆盖
```

注册表在进程启动时扫描；所有 add/remove/disable 对**下一个进程**生效。

## 验证与排障标准动作

1. chainctl.py validate 先过；
2. probe_dynamic_runner.py 必须 VERIFY ok=True（drawer_place：抽屉 qpos>=0.08
   且方块距 cube_goal < 50mm，通常 <15mm）；
3. JSONL 中每个 plan 的步骤参数人工核对一遍（特别是 $site: / $cond. 渲染值）；
4. 最小间隙看 skill_end.min_clearance 与 sample 的 clear_mm，abort 阈值 4mm。

## 红线（不要做）

- 不要改 robopal 源码；新行为落在 darwin 包内。
- 不要在 runner 执行端加 `if task == ...` 分支——新行为通过 objectives + YAML
  method 组合表达；执行端只认条件和方法。
- 不要为了走捷径在 method 里直写 mj_data.qpos 搬运 CARTIK 控制的臂；用原语。
- 不要把大数组/点云塞进 YAML 方法或日志参数。
- YAML 方法里无法写任意 Python；这是有意的安全边界，需要计算逻辑走新 Method 子类。
- 改完不跑 probe 不算完成。

## 新任务自动进化（SkillCreator + ExperienceStore）

解决"只跑了一两个任务、技能库不全"的问题：当一个目标条件在 registry 里
没有任何 method 能匹配时，runner 不会直接失败，而是自动生成新方法、注册、
重试，并把每次执行结果沉淀为经验。

### 触发链路

```
observe → 找未满足条件 → registry.select(cond)
                              └─ 无匹配 → SkillCreator.create_for(cond)
                                              ├─ LLM 可用 → 生成 YAML method（schema 校验）
                                              └─ LLM 不可用 → 规则回退模板（已知 cond.kind）
                                              → DeclarativeMethod 校验 + skill 名存在性校验
                                              → registry.register(replace=True)
                                              → 写入 methods/<name>.yaml（下次进程自动加载）
                              → 重试 select → 执行
```

### 经验沉淀（参考 EvoAgentX AFlow experience.json）

每次 attempt 结束，runner 把执行过的 `(method, cond)` 写入
`ExperienceStore`（`logs/experience/<task>.jsonl`）：

```json
{"task": "...", "cond": "...", "cond_kind": "body_near_site",
 "method": "auto_body_near_site", "success": false,
 "before": 0.0, "after": 0.0, "fail_phase": "lift",
 "params": {"k": 2.0}, "log": "logs/episodes/...", "ts": ...}
```

关键能力：
- `best_score(task, method)`：该方法历史最优分；
- `is_repeated_failure(task, method, params)`：防止 SkillCreator 重复生成
  已知失败的方法/参数组合；
- `format_for_creator(task, cond)`：生成给 LLM 的提示，明确列出"该条件上
  哪些方法在哪一阶段失败、参数是什么"，要求"换思路而非微调"。

### 两级回退

1. **LLM 生成**（有 ARK_API_KEY / OPENAI_API_KEY 时）：基于现有方法列表、
   可用 skill 名录、历史失败经验，生成定制化 YAML method；
2. **规则回退**（LLM 不可用或生成失败）：对 `body_near_site` / `body_lifted`
   等已知 cond.kind 输出通用抓放链模板，保证链路不中断。

### 手动触发/验证

```bash
# 精简 registry（只有开抽屉方法）跑 drawer_place → place 条件无匹配 → 自动生成
python scripts/probe_skill_creator.py

# 查看经验库（每个 task 一个 jsonl）
ls logs/experience/
```

### 与现有记忆的分工

| 层 | 存什么 | 在哪 |
|---|---|---|
| RAGMemory | 抓取点 rel_offset 级微观经验（哪类物体哪个位置抓得起来） | `darwin/memory/rag.py` |
| ExperienceStore | method/skill 链级宏观经验（哪种方法对哪类条件有效/无效） | `darwin/agents/experience_store.py` |
| skill_forge | 成功轨迹 → forged skill（重放模板） | `darwin/evolution/skill_forge.py` |

### 红线补充

- 自动生成的 YAML 方法必须通过 `DeclarativeMethod` 校验（模板变量在白名单）；
- 引用的 skill 名必须存在于 `agent.skills`，否则丢弃不注册；
- 经验库里已失败的方法/参数组合，SkillCreator 不会重复生成（LLM 提示明确禁止）。

## 架构重构与自进化增强（2026-09-19）

### 模块拆分（消除循环依赖 + 重复代码）
- `objectives.py`：GoalCond + JointAtLeast/BodyLifted/BodyNearSite + objectives_from_entry
- `methods.py`：PlanContext + Method 基类 + 4 个内置方法 + select_method
- `env_utils.py`：get_env + StepRecorder + apply_bounds（runner.py 与 runner_dynamic.py 共用）
- runner_dynamic.py 从 798 行降到 505 行；runner.py 从 385 降到 340 行

### 自进化闭环增强
| 能力 | 实现位置 | 说明 |
|------|---------|------|
| 跨进程经验累积 | ExperienceStore._load_history | 启动时加载 logs/experience/*.jsonl |
| 经验→方法排序 | ChainRegistry.reorder_by_experience | 失败后按成功率重排，下次优先选成功方法 |
| 方法空间搜索 | SkillCreator.mutate_existing | 成功后对 YAML 方法做小变异（参数±10%/skill替换/步交换） |
| forged 进 registry | ForgedSkillMethod + load_forged | 成功轨迹沉淀的技能可被 select 选中 |
| 方法编排 | Method.next 属性 + YAML next 字段 | 方法 A 成功后优先选 next 方法 |
| fail_phase 分类 | _classify_failure | collision/grip_failed/timeout/force_exceed/unknown_skill/other |
| $cfg 默认值 | chain_registry._CFG_DEFAULTS | cfg 缺失键返回合理默认值，避免 None 崩溃 |
| rel_offset 兜底 | _build_candidates | 超 [-2,2] 的候选点丢弃，用物体中心 |
| 进化主循环接入 | run_evolution.py | 用 DynamicEpisodeRunner + ExperienceStore + default_registry |

### 进化流程（更新后）
```
MAP-Elites 采样 cfg → DynamicEpisodeRunner.run(cfg)
  → observe(条件) → select(registry, 经验排序) → execute
  → 成功: skill_forge + ExperienceStore.add + mutate_existing(方法变异)
  → 失败: RAG failure + ExperienceStore.add + reorder_by_experience
→ 新方法写入 methods/*.yaml → 下次进程自动加载
```

## 失败驱动的自适应进化（2026-09-19 增强）

### 核心：失败了就要变，不能重复同样的尝试

每次 attempt 失败后触发三重自适应：

1. **参数自适应** (`_adapt_cfg`)：按 fail_cat 调整 cfg
   - grip_failed → k_descend ×1.25, hover -0.02, stop_above -0.005
   - timeout → place_timeout +50
   - collision → hover +0.03, k_descend ×0.8
   - force_exceed → stiffness ×0.7, damping ×1.2

2. **方法排序** (`reorder_by_experience`)：成功率 > 无记录(0.5) > 全失败(0)

3. **方法变异** (`_maybe_mutate_failure`)：某 method 连续失败 ≥3 次
   - YAML 方法 → `mutate_existing`（参数±10%/skill替换/步交换）
   - 内置 Python 方法 → `create_for` 生成 YAML 替代
   - 新方法移到 registry 最前面，下次 attempt 优先尝试

### 经验提示增强
`ExperienceStore.format_for_creator` 现在输出：
- 失败模式分布（如 `{"grip_failed": 6, "timeout": 1}`）
- 针对主要失败模式的改进建议（如 grip_failed → 增大 k_descend）

### 抓取候选优化
`_build_candidates` 按 `|rel_offset|` 升序排序，中心抓取优先（边缘抓取不稳定）。
