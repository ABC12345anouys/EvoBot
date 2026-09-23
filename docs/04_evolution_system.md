# 04 · 进化系统设计

> 对应代码：`darwin/evolution/loop.py`、`darwin/evolution/map_elites.py`、`darwin/evolution/skill_forge.py`

## 1. 设计目标

进化系统是 EvoBot 的**自进化引擎**，负责在反复执行任务的过程中探索多样化的技能配置，并将成功轨迹自动沉淀为可复用的代码技能。设计目标：

1. **质量多样性搜索**：不只找单一最优解，而是按特征维度分箱，每个 cell 保留最优配置，维护多样化技能库。
2. **成功即沉淀**：成功轨迹自动通过 `skill_forge` 生成 forged 策略技能，下次直接复用，减少对 LLM 的依赖。
3. **配置驱动**：进化搜索空间是超参数（hover / k_descend / lift_height / stiffness / ...），不修改技能代码本身。

## 2. MAP-Elites 质量多样性搜索

### 2.1 核心思想

传统进化算法只找全局最优解，MAP-Elites（Mouret & Clune, 2015）则维护一个**特征空间上的网格**，每个 cell 只保留该区域内的最优解。这样做的好处：

- 避免种群收敛到单一最优，保持技能多样性
- 不同特征区域的最优技能可应对不同任务场景
- 失败/退化解不会污染已有的优良解

### 2.2 MapElites 类

```python
class MapElites:
    search_space: Dict[str, List[Any]]      # 超参搜索空间
    feature_dimensions: List[str]           # 特征维度名
    feature_ranges: Dict[str, (float, float)]  # 各维度取值范围
    feature_bins: int = 10                  # 每维度分箱数
    fitness_key: str = "score"              # 适应度字段
    n_iterations: int = 100
    exploration_ratio: float = 0.2          # 20% 随机探索，80% 变异利用
```

### 2.3 主循环

```python
for step in range(n_iterations):
    # 80% 概率从 archive 随机选父代变异，20% 随机采样
    if archive and random() > exploration_ratio:
        parent = random.choice(archive.values())
        cfg = mutate(parent.cfg)
    else:
        cfg = random_cfg()

    metrics = evaluator(cfg)          # 执行 cfg，返回 metrics
    fitness = metrics[fitness_key]
    cell = bin(metrics)               # 按特征维度分箱

    # 该 cell 无存档或新解更优 → 接受
    if cell not in archive or fitness > archive[cell].fitness:
        archive[cell] = ArchiveEntry(cfg, fitness, metrics, cell)
```

### 2.4 变异策略

`_mutate(parent)` 只改一个超参数维度，从 `search_space` 中选不同于当前值的候选项：

```python
key = random.choice(search_space.keys())
alternatives = [v for v in choices if v != current]
cfg[key] = random.choice(alternatives)
```

单点变异保证搜索步长可控，避免跳变过大。

### 2.5 分箱函数

```python
def _bin(value, lo, hi, bins):
    t = clip((value - lo) / (hi - lo), 0, 1)
    return min(int(t * bins), bins - 1)
```

多维度特征组成 cell 坐标 `(bin_0, bin_1, ...)`。

## 3. EvolutionLoop

`EvolutionLoop` 是 MAP-Elites 与 Agent/RAG 的**粘合层**：

```python
class EvolutionLoop:
    benchmark: Dict
    map_elites: MapElites
    results: List[Dict]

    def run(self, agent_runner) -> dict:
        self.map_elites.evaluator = lambda cfg: self._run_episode(cfg, agent_runner)
        best_cfg, info = self.map_elites.run()
        return {"best_cfg": best_cfg, "archive_size": len(info["archive"]), "history": info["history"]}

    def _run_episode(self, cfg, agent_runner):
        result = agent_runner(cfg)            # EpisodeRunner.run(cfg)
        if result["success"]:
            forge_skill_from_trajectory(result["trajectory"], task_name, feat)
        self.results.append({"cfg": cfg, "success": result["success"], **result["metrics"]})
        return result["metrics"]              # 返回给 MAP-Elites 算 fitness
```

### 3.1 契约

- `agent_runner(cfg)` → `{success, trajectory, info, metrics}`
- `metrics` 必须包含 `fitness_key`（默认 `"score"`）和所有 `feature_dimensions` 字段
- `cfg` 只含 MAP-Elites 搜索的超参，任务名取自 `benchmark["task_name"]`

### 3.2 适应度

`EpisodeRunner` 返回的 `score = success - 0.02 * attempts`：
- 成功得 1.0 分
- 每多一次尝试扣 0.02 分，鼓励少尝试快成功
- 失败得 0 分（或负值）

## 4. skill_forge：成功轨迹 → 代码技能

### 4.1 核心思想

成功的执行轨迹本质上是一个**已验证的动作序列**，与其每次让 LLM 重新规划，不如直接把它沉淀为可复用的代码技能。这就是 `skill_forge`。

### 4.2 forge 流程

```
成功轨迹 trajectory (List[{action, params, result}])
    ↓ extract_skill_code
生成 entry.py: 按顺序调用 registry.execute(action, params)
    ↓ _save_forged_skill
写入 darwin/skills/forged/<name>/entry.py + SKILL.md
    ↓ _update_evidence
更新 SKILL.md 的 evidence.cells / tasks / confidence
```

### 4.3 entry.py 生成

```python
def extract_skill_code(trajectory, task_name):
    action_body = "\n".join(
        f"    result = registry.execute('{a['action']}', env=env, **{a.get('params') or {}})"
        for a in trajectory
    )
    code = f'''
def run(env=None, registry=None):
    """由成功轨迹抽象而来：按验证过的动作序列重放。"""
{action_body}
    return result
'''
```

生成的技能是**动作序列重放**模板，坐标为该次成功的观测值。后续可通过参数化改造提升泛化能力。

### 4.4 SKILL.md 元数据

```yaml
---
name: pick
description: 任务 pick 的成功轨迹抽象：home -> move_above -> descend -> close -> lift
entry_file: entry.py
kind: strategy
confidence: single-shot
applies_when: '类似 pick 的任务，涉及操作: {move_above, descend, close, lift, home}'
evidence:
  cells: [pick_143025123456]
  tasks: [pick]
  attempts: 1
  solved_seeds: [pick_143025123456]
  failed_seeds: []
---
```

### 4.5 evidence 累积

每次同一任务成功，`_update_evidence` 更新对应 forged 技能的 evidence：
- `cells` 加入新的验证单元
- `attempts += 1`
- `solved_seeds` 加入新 seed
- 置信度按规则自动升级（cells≥3 且 tasks≥2 → verified）

## 5. 与 RAG 的协同

进化系统与 RAG 记忆层形成**双层沉淀**：

| 层 | 沉淀内容 | 形式 | 用途 |
|----|----------|------|------|
| **RAG 记忆层** | 成功/失败的 `rel_offset` + 元数据 | `.md` frontmatter | 候选过滤、偏移复用、避坑 |
| **skill_forge 层** | 成功的完整动作序列 | `entry.py` + `SKILL.md` | 直接重放，跳过 LLM 规划 |

进化循环中：
1. Agent 执行时先用 RAG 检索最佳偏移（减少尝试）
2. 成功后 RAG 记录成功经验 + skill_forge 生成/更新策略技能
3. 下次遇到同类任务，优先调用 forged 技能（已验证），不再走 LLM

## 6. 防退化机制

- **MAP-Elites 分箱**：新解只在所属 cell 更优时才替换，不会全局覆盖
- **RAG 失败记忆**：失败偏移被记录，下次候选生成时过滤，避免重复踩坑
- **forged 技能 evidence**：多次验证才升级 confidence，单次成功不盲目信任
