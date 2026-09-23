# 01 · RAG 记忆层设计

> 对应代码：`ragbot/memory/store.py`、`ragbot/memory/rag.py`、`ragbot/__init__.py`

## 1. 设计目标

RAG 记忆层是 EvoBot 的**核心公开能力**，定位为"给 LLM 操控机械臂当记忆层"。设计目标有三：

1. **减少 token 消耗**：执行前检索历史成功/失败经验注入 prompt，避免 LLM 重复推理已知方案。
2. **提高准确率**：跳过近期失败的抓取偏移（避坑），复用已验证成功的偏移（复用）。
3. **越用越准**：执行后沉淀结果，经验库随 episode 增长而丰富，形成自进化闭环。

核心约束：**包 `ragbot` 仅依赖 `numpy` + `pyyaml`**，不引入任何 ML/仿真依赖，确保可被任意感知/控制栈接入。

## 2. 存储层：MemoryStore

### 2.1 数据格式

每条记忆是一个 `.md` 文件，采用 **YAML frontmatter + Markdown 正文** 的结构：

```markdown
---
scope: global
kind: strategy            # primitive / perception / strategy / failure / infra
confidence: single-shot   # single-shot / probable / verified
evidence:
  cells: [pick_box_s0]    # 验证单元唯一标识
  tasks: [pick]           # 已验证的任务（跨任务才升 verified）
  attempts: 3
  solved_seeds: [pick_box_s0]
  failed_seeds: []
applies_when: "box 物体 / pick 任务"
task: pick
shape: box
size: [0.05, 0.05, 0.05]
rel_offset: [0.05, 0.0, 0.0]
score_band: mid
---

成功经验描述正文...
```

**为什么用 .md + YAML frontmatter 而非 JSON？**
- 人可读、可 git 追踪、diff 友好
- frontmatter 存结构化元数据，正文存自然语言经验描述，兼顾机器检索与人工审核
- 与 `skill_forge` 生成的 `SKILL.md` 格式统一

### 2.2 evidence 合并与置信度升级

`merge_evidence(old, new)` 在写入同名记忆时合并证据，并按以下规则**自动升级置信度**：

| 条件 | 置信度 |
|------|--------|
| `cells < 2` | `single-shot`（单次验证） |
| `cells >= 2` | `probable`（多次验证） |
| `cells >= 3` 且 `tasks >= 2` | `verified`（跨任务验证，可信度最高） |

这一机制确保：**同一条经验在多个不同任务上验证通过后，才会被标记为 verified**，避免单次成功就过度信任。

### 2.3 SkillKind 与 Confidence 枚举

自包含于 `store.py`，不依赖 `darwin.skills.base`，确保 `ragbot` 包可独立发布：

- `SkillKind`: `primitive` / `perception` / `strategy` / `failure` / `infra`
- `Confidence`: `single-shot` / `probable` / `verified`

## 3. 经验层：RAGMemory

### 3.1 归一化抓取经验

跨物体泛化的关键是**归一化偏移**：

```
rel_offset = (grasp_point - object_center) / object_half_size
```

- `grasp_point`：抓取点世界坐标（调用方感知提供）
- `object_center`：物体中心世界坐标
- `object_half_size`：物体半尺寸（三维最大维用于 `size_band` 分级）

归一化后，不同尺寸物体的"顶部抓取"都对应 `rel_offset ≈ [0, 0, 1]`，经验可跨物体迁移。

### 3.2 多维条件匹配

检索时按 `(shape, task, size_band)` 三维过滤：

- `shape`：物体形状类别（box / cylinder / ...）
- `task`：任务名（pick / place / insert / ...），`task="any"` 可匹配任意任务
- `size_band`：物体半尺寸最大维分级，由 `size_band_of(size)` 计算：

  | 半尺寸最大维 (m) | size_band |
  |-------------------|-----------|
  | `< 0.02` | `tiny` |
  | `0.02 ~ 0.04` | `small` |
  | `0.04 ~ 0.09` | `mid` |
  | `>= 0.09` | `large` |

**尺寸差异大时 offset 语义不同**，因此 `is_failed` 严格要求 `size_band` 一致才匹配。

### 3.3 失败记忆 TTL 机制

失败记录不是永久拉黑，而是有 **TTL = 20 episodes** 的存活期：

```python
FAILURE_TTL = 20

def is_failed(self, task, feat, rel_offset):
    for rec in failure_records:
        if self.ts - rec.ts > self.FAILURE_TTL:
            continue  # 过期，不再视为失败
        if dist(off, rec.rel_offset) < OFFSET_TOL:
            return rec
    return None
```

- `tick()`：每个 episode 调用一次，推进全局时钟 `self.ts`
- `decay()`：删除 TTL 过期的失败记录文件

**为什么不用永久拉黑？** 仿真中物体位置/姿态会变化，某个偏移在第 3 集失败，不代表第 30 集还会失败。TTL 让失败记忆"自然遗忘"，避免经验库僵化。

### 3.4 语义检索

`_SemanticIndex` 提供两级后端：

| 后端 | 启用条件 | 实现 |
|------|----------|------|
| **TF-IDF 余弦**（默认） | 始终可用 | 纯 numpy 实现，无外部依赖 |
| **sentence-transformers** | `RAGBOT_EMBED=1` | `all-MiniLM-L6-v2`，失败自动回退 TF-IDF |

检索排序综合**语义相似度** + **中心度**（offset 越接近物体中心越优先）：

```python
score = 0.5 * semantic_similarity + 0.5 * center_closeness
```

### 3.5 UCB 探索式候选排序

`rank_candidates` 对候选抓取点按**统计成功率 + UCB 探索加成**重排：

```python
rate, n = success_rate(shape, task, band)   # 拉普拉斯平滑后的成功率
prio = rate + 0.15 / sqrt(1 + n)            # UCB bonus：试过越少越鼓励探索
```

- 拉普拉斯平滑：`(succ + 0.5) / (n + 1)`，无数据时中性先验 0.5
- UCB bonus：低样本量的 band 获得额外加成，鼓励探索未充分尝试的候选

### 3.6 control_mode 感知检索

`retrieve(task_desc, feat, k, control_mode)` 支持按控制方案加权：

- 同 `control_mode`（vla / ik / force / hybrid）经验 **+0.2** 加权
- 通用经验（无 `control_mode`）不加不减
- 其他 `control_mode` 经验不加

这确保在 VLA / IK / 力控混存的经验库中，优先取回与当前方案匹配的经验，同时保留通用经验防止冷启动。

### 3.7 边界异常过滤

`rel_offset` 写入前必须通过 sanity check：

```python
OFFSET_SANITY = 1.5
if np.linalg.norm(rel) > OFFSET_SANITY:
    rel = [0.0, 0.0, 0.0]  # 边界异常，不写入
```

防止仿真中物体在边界、抓取点飞出等噪声产生 `|rel_offset| > 1.5` 的异常值污染经验库。

## 4. 统计与成功率表

`grasp_stats.md` 维护 `(shape, task, score_band) → {succ, fail}` 的成功率表：

- 每次 `add_success` / `add_failure` 自动更新对应 key 的计数
- `success_rate(shape, task, band)` 返回 `(rate, n)`，带拉普拉斯平滑
- 同 key 无数据时回退到 `(shape, "any", band)`，再无则返回中性先验 `(0.5, 0)`

## 5. 公开 API

```python
from ragbot import RAGMemory, MemoryStore, merge_evidence, size_band_of

rag = RAGMemory()                         # 加载默认经验库
rag.best_success(task, feat)              # 最匹配成功经验
rag.is_failed(task, feat, rel_offset)     # 该偏移是否近期失败
rag.retrieve(desc, feat, k=5)             # 语义检索 top-k
rag.rank_candidates(task, feat, cands)    # UCB 重排候选
rag.add_success(task, feat, rel_offset, score_band, attempts, exp, **extra)
rag.add_failure(task, feat, rel_offset, score_band, fail_phase, exp, **extra)
rag.tick()                                # 推进 TTL 时钟
rag.decay()                               # 清理过期失败记忆
```

`**extra` 支持任意额外元数据自动序列化，已知字段包括：`control_mode` / `control_hints` / `perception` / `execution` / `environment` / `llm_meta` / `failure_detail`。

## 6. 写入节奏

经验库的写入遵循 **episode 节奏**：

```
执行前: retrieve / best_success / is_failed / rank_candidates
执行后: add_success / add_failure + tick()
周期性: decay() 清理过期失败
```

这一节奏由 `ManipulationAgent.reflect()` 和 `EpisodeRunner.run()` 保证，调用方只需按此模式使用即可。
