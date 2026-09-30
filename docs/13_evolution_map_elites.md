# 13 · 进化系统设计（MAP-Elites + skill_forge）

> 对应代码：`darwin/evolution/`（4 个文件）
>
> 输入来自 [14 动态执行路径](14_dynamic_runner.md) 的 episode 结果（`score`/`attempts`/`steps` + 成功轨迹），产出回灌到 `chain_registry` 与 `skills/forged/`。

## 1. 设计目标

1. **质量多样性搜索**：不追求单一最优参数，而是在"尝试次数 × 步数"的特征空间里铺一张档案（archive），每个 cell 留一个精英——避免只收敛到一种解法。
2. **成功即沉淀**：一次成功轨迹自动抽象为**可复用的 forged 技能**，下次同类任务可直接被方法库选中。
3. **失败不在这里处理**：失败记忆由 `agent.reflect` 写 RAG（见 `loop.py:56` 注释），本目录只负责"成功 → 技能"与"参数 → 搜索"。

## 2. 数据流

```
DynamicEpisodeRunner.run(cfg)
        │ metrics = {score, attempts, steps, shape}
        ▼
EvolutionLoop._run_episode  ──► MapElites(archive)          ← 参数搜索
        │ 成功                        │
        │                             ▼
        └──► forge_skill_from_trajectory ──► skills/forged/<name>/{entry.py, SKILL.md}
                                                    │
                                                    ▼
                                    chain_registry.load_forged → 可被 select 选中
```

## 3. MAP-Elites（`map_elites.py`，120 行）

### 3.1 档案与分箱

- archive 类型 `Dict[Tuple[int, ...], ArchiveEntry]`，键 = cell 坐标（`:55`）
- 维度由 benchmark 注入：**`["attempts", "steps"]`**，范围 `attempts (0,8)`、`steps (0,1500)`，`fitness_key="score"`（`benchmarks/__init__.py:15-19`）
- 每维默认 10 箱（`:32`）→ 档案上限 **10×10 = 100 个 cell**
- `_bin`：先归一化并 clamp 到 [0,1]，`idx = int(t*bins)` 再 `min(idx, bins-1)`；`hi <= lo` 时返回 0（`:114-120`）

### 3.2 quality 与 diversity

- quality = `metrics[fitness_key]` = `metrics["score"]`
- **真实 score 公式在 runner**：`score = float(success) - 0.02 * attempts`（`runner_dynamic.py:1485`）
- diversity 机制 = **每个 cell 只留一个精英**，仅当新 fitness **严格大于** 时才替换（`:75-81`）
- `ArchiveEntry(cfg, fitness, metrics, cell)` 为 frozen dataclass（`:16-21`）

### 3.3 算子：**只有变异，没有交叉**

| 算子 | 行为 |
|---|---|
| `_random_cfg()` | 搜索空间每维独立随机取值 `:90-91` |
| `_mutate(parent)` | 随机挑 **1 个 key**，从该 key 的非当前值里再随机取一个；choices ≤1 时原样返回 `:93-104` |
| 选择 | `random.random() > exploration_ratio` → 从档案随机取 parent 变异（`source="mutate"`），否则随机采样（`source="random"`）`:60-66` |

**没有交叉（crossover）算子**——注释也把 `exploration_ratio` 解释为"0.2 → 80% 变异利用，20% 随机探索" `:6`。

### 3.4 迭代与接线

- 主循环 `MapElites.run()` `:53-86`，`for step in range(n_iterations)`
- 返回值 `(best.cfg if best else None, {"archive", "history", "best"})`；history 每条 `{step, source, cfg, fitness, cell, accepted}` `:83-84`
- 构造时校验 `feature_ranges` 必须覆盖所有 `feature_dimensions`，否则 `raise ValueError` `:49-51`
- **`program` 形参在本项目恒为 `None`**：`EvolutionLoop.run` 调 `self.map_elites.run()` 不传 `program`（`loop.py:44`），真正执行 episode 的是被覆盖的 `evaluator`——`loop.py:43` 把 evaluator 换成 `lambda cfg: self._run_episode(cfg, agent_runner)`，该注释明确"evaluator(cfg) 执行 cfg 并返回 metrics（program 返回值会被丢弃）"`loop.py:42`

## 4. skill_forge：成功轨迹 → forged 技能

### 4.1 两条触发路径（注意：存在重复 forge 的可能）

1. `EvolutionLoop._run_episode` 中 `if success:` 直接调用（`loop.py:55-58`）
2. `ManipulationAgent.reflect → _maybe_forge → forge_skill_from_trajectory`（`agent.py:239,271-281`），异常被 `except Exception: pass` 吞掉，不影响主流程

同一次成功可能被两条路径各 forge 一次，靠目标目录已存在来判断去重。

### 4.2 输入与产出

- 输入 `trajectory: List[Dict]`，每步需有 `"action"`（`"params"` 可选）`:23-29`；**`feat` 形参存在但函数体内完全未使用** `:101`（两个调用点都传了它）
- 落盘 `FORGED_DIR = darwin/skills/forged`（`skills/forged/__init__.py:18`），目录名 `safe_name = re.sub(r"[^\w\-]", "_", name).lower()`；**同名已存在时追加时间戳后缀另建目录** `:47-50`

| 产出文件 | 内容 |
|---|---|
| `entry.py` | `def run(env=None, registry=None):`，逐步 `registry.execute('<action>', env=env, **{params})`，末尾 `return result`。docstring 固化："坐标为该次成功观测值"——**即快照坐标，复用必须重投影**（见 [14](14_dynamic_runner.md) §5）`:21-41` |
| `SKILL.md` | frontmatter：`name` / `description`（截断 200）/ `entry_file: entry.py` / `kind: strategy` / `confidence: single-shot` / `applies_when` / `evidence: {cells, tasks, attempts, solved_seeds, failed_seeds}`；正文 `# <safe_name>` + description `:54-60` |

### 4.3 证据更新与置信度升级（幂等）

- 若 `entry.py` 已存在 → **不重建，只更新 evidence**，返回 `created=False` `:104-117`
- `_update_evidence` `:64-96`：`cells/tasks/solved_seeds` 取并集排序、`attempts += 1`，然后重算 confidence：

| confidence | 条件 |
|---|---|
| `verified` | `len(cells) >= 3` **且** 涉及任务数 `>= 2` |
| `probable` | `len(cells) >= 2` |
| `single-shot` | 其他 |

- 解析失败/缺 frontmatter → 静默返回 `:68-75`

### 4.4 实存产物（核对当时的 `skills/forged/`）

9 个技能目录：`any / drawer_place / libero_goal_00 / libero_goal_07 / peg_in_hole / pick / pickplace / robust_grasp`。
例：`pickplace` 为 8 步链（`home→move_above→descend→close_gripper→lift→move_to_xy_top→place→open_gripper`），SKILL.md 显示 `confidence: probable`、`attempts: 18`；`libero_goal_00` 是 `single-shot`、`attempts: 1`——**说明上面的置信度阈值确实在生效**。

### 4.5 读回（加载侧）

`load_forged_skills()` 遍历子目录（需有 `entry.py`）→ `importlib` 动态执行取模块 `run`（回退 `main`）→ 用 `_parse_skill_md` 解析 `SKILL.md` 成 `SkillSpec` → 返回 `{name: ForgedSkill}`（`skills/forged/__init__.py:52-108`）。`ForgedSkill.execute` 注入 `_RegistryProxy`，把 `registry.execute` 转发到 skills dict，从而兼容历史签名 `run(env, registry)` `:22-49`。

## 5. 回灌：forged 如何重新参与决策

1. `chain_registry.load_forged(reg)` → 包装成 `ForgedSkillMethod` 注册（`chain_registry.py:548-571`）；`default_registry()` 在注册内置方法 + YAML 之后调用它 `:339-360`
2. 能否被选中：`ForgedSkillMethod.can_achieve` = `entry["task_name"] in self._tasks`，`_tasks` 来自 evidence.tasks；无 tasks 时回退用技能名 `:416-417,562-565`
3. **执行侧防退化**：不直接重放快照，而是锚点重投影；几何过期/环境不匹配时 `raise RuntimeError` 让 runner 换方法 `:452-546`

另一条回灌是**宏观经验**（不属于本目录，但同一闭环）：`runner_dynamic._record_experience` 按 `(method, cond)` 写 `logs/experience/<task>.jsonl`，`after = 1.0/0.0`、`params_hint` 只挑 `hover/k_descend/lift_height/stiffness/damping` 五个 key（`runner_dynamic.py:1205-1229`、`experience_store.py:79-86`）；再由 `format_for_creator` 转成"禁止重复这些失败方法"的提示 `experience_store.py:106-146`。

**分工**（`experience_store.py:10-13`）：RAGMemory 记"抓取点 `rel_offset`"**微观**经验；ExperienceStore 记"method/skill 链"**宏观**经验。

## 6. 调参常量

`darwin/evolution/` 目录内**没有任何模块级大写常量**，全部是默认参数：

| 位置 | 名称 | 值 | 含义 |
|---|---|---|---|
| `map_elites.py:32` | `feature_bins` | `10` | 每维分箱数（2 维 → 最多 100 cell）|
| `:33` | `fitness_key` | `"score"` | 取 fitness 的 metrics 字段 |
| `:34` | `n_iterations` | `100` | MapElites 迭代数 |
| `:35` | `exploration_ratio` | `0.2` | 随机探索比例 |
| `:36` | `seed` | `None` | **传值才 `random.seed`，否则每次运行不可复现** `:46-47` |
| `loop.py:26` | `n_iterations` | `50` | EvolutionLoop 默认迭代数（透传给 MapElites）|
| `loop.py:27` | `memory_store` | `None` | 仅存到 `self.memory`，目录内无其它引用 |
| `skill_forge.py:56` | description 截断 | `200` | SKILL.md description 上限 |

**真正决定分箱的常量在 `darwin/benchmarks/__init__.py`**：`feature_dimensions`/`feature_ranges`/`fitness_key` `:16-18`，以及搜索空间
`COMMON_SEARCH_SPACE`：`hover [0.10,0.12,0.15]`、`k_descend [1.5,2.0,2.5]`、`lift_height [0.50,0.52,0.55]`、`jit [0.0,0.005,0.01]` `:22-27`；
`INSERT_SEARCH_SPACE` 追加 `stiffness [60,100,150]`、`damping [20,40,60]`、`spiral_radius [0.003,0.004,0.006]` `:30-36`。
惩罚系数 `0.02`（每次尝试扣 0.02）在 `runner_dynamic.py:1485`。

## 7. 已知落差（如实记录）

1. **fitness 描述与实现不一致**：`map_elites.py:5` 注释写 "fitness = 成功率 × (1 − 归一化步数)"，但代码里没有任何按步数归一化的计算——步数只参与**分箱**，不参与 fitness；真实公式是 `success − 0.02×attempts`（`runner_dynamic.py:1485`）。
2. **`loop.py` docstring 声称的两个能力没有实现**：`:6` "replay 验证后入库"、`:7` "失败 → failure_miner 写避坑记忆"——本目录内既无 replay 也无 failure_miner；失败记忆实际由 `agent.reflect` 写 RAG（`loop.py:56` 注释与 `skill_forge.py:7` 注释互相印证）。
3. **`cell` 同词不同义**：`skill_forge.py:115` 生成 `cell = f"{task_name}_{%H%M%S%f}"`（时间戳种子标签）写进 `evidence.cells`，与 MAP-Elites 的 cell 坐标（`map_elites.py:72`）**不是一回事**，命名易混淆。
4. **未接线字段**：`EvolutionLoop.memory` 只在 `__init__` 赋值（`loop.py:27,29`，`scripts/run_evolution.py:62` 实际传 `memory_store=None`）；`EvolutionLoop.results` 只被追加、从未被读取或返回（`loop.py:38,60`）。
5. **`forge_skill_from_trajectory` 的 `feat` 形参未被使用**（`skill_forge.py:101`），但两个调用点都传了它。

## 8. 已知限制

- **`seed=None` 使进化过程不可复现**（`map_elites.py:46-47`）——与 [12 调试纪要](12_libero_debug_2026-09-30.md) 讨论的基准噪声是同一类问题，若要复现实验需显式传 seed。
- **只有变异没有交叉**：搜索能力上限受限于"单点变异 + 随机重启"。
- **fitness 惩罚尝试次数**：`−0.02×attempts` 会系统性压低"试多次才成功"的策略，与分箱维度 `attempts` 存在取向上的张力。
