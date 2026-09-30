# 15 · 记忆层与跨进程学习设计

> 对应代码：`darwin/memory/`、`darwin/ipc/`（以及仓库根的 `ragbot/`）
>
> 相关：[01 RAG 记忆层](01_rag_memory.md)（`ragbot/memory/` 的设计叙述）、[14 动态执行路径](14_dynamic_runner.md)（谁在调用记忆层）

## 0. 先讲清两条边界（否则后面会误读）

**边界一：`darwin/ipc/` 不使用 `darwin/memory/`。** 证据：`ipc/agent_learner.py:36-39` 的 import 只有 `agents.reflection.adapt_cfg` / `skills.skill_config` / `skills.physics_profile` / `.protocol`，**没有任何 memory 相关 import**；`ipc/sim_worker.py:97` 显式构造 `DynamicEpisodeRunner(..., rag=None, ...)` 并在 `:84` 注明"`rag=None`：反思/记忆在 agent 进程；sim 只负责执行与上报"。

即：**记忆层是 agent 进程内的（单次 episode 循环）；跨进程学习的载体是磁盘 YAML/JSON + unix socket**，两者目前没有代码连接。

**边界二：`ragbot/` 与 `darwin/memory/` 是"复制 + 分叉"，不是依赖关系。** 证据：

- `pyproject.toml:6` 的包名是 `ragbot`，`:36-37` 注明"只打包 ragbot 包（不打包 darwin / robopal）"；`darwin` 没有任何打包入口。
- `ragbot/memory/store.py:10` 自述"本文件自包含，不依赖 ragbot 之外的任何包"，并在本地**重定义** `SkillKind`/`Confidence` 枚举；而 `darwin/memory/store.py:19` 改为 `from ..skills.base import Evidence, Confidence, SkillKind`——这是两边唯一的实质分歧。
- `rag.py` 两边行数 399/423、常量与公式完全相同，唯二差异是环境变量名与文档串（`RAGBOT_EMBED` vs `DARWIN_RAG_EMBED`）。
- **全仓库没有任何代码 `import ragbot`**：`darwin/**` 下零命中；`ragbot` 的消费者只有 README 与设计文档（`架构改进建议.md:80-91` 已记录"两份 `MemoryStore`/`_SemanticIndex`/`size_band_of` 几乎逐行复制"）。
- 但两者的 legacy 迁移来源指向**同一个文件**（`data/rag_memory_v2.json`），该文件已改名 `.imported`，故迁移分支现在不会触发。

## 1. 记忆层（`darwin/memory/`）

### 1.1 文件与职责

| 文件 | 职责 |
|---|---|
| `memory/__init__.py` | 仅 re-export：`MemoryStore, merge_evidence, RAGMemory, size_band_of` |
| `memory/store.py`（~120 行）| RPent 式 YAML frontmatter 记忆存储：evidence 合并 + 置信度自动升级 |
| `memory/rag.py`（399 行）| **RAGMemory v3**：抓取经验库 = MemoryStore 后端 + TTL + 语义检索 + UCB 重排 + v2 JSON 迁移 |
| `memory/experience/` | 实际落盘目录（默认根），现存 8 个 `.md` |
| `memory/failures/` | **空目录**——失败记录实际也写在 `experience/`（`rag.py:247` 的 `_write_record` 只写 `self.store.root`）|

### 1.2 一条记录：字段、键与命名

落盘 = Markdown + YAML frontmatter + 一行 body（`store.py:25-38`）。**存储键 = 文件名 stem**（`store.py:71-96`）。

RAG 条目命名：`f"exp_{self.ts:05d}_{rec['type']}_{len(self._cache):05d}"`（`rag.py:214-218`），如 `exp_00001_success_00000.md`、`exp_00000_failure_00006.md`。

字段分三类：
- **通用元数据协议**：`scope`(global/suite)、`kind`(primitive/perception/strategy/failure/infra)、`confidence`、`evidence{cells,attempts,solved_seeds?,failed_seeds?,contradicted_by?}`、`applies_when`（`store.py:1-10,20-22`）
- **RAG 自定义**：`task`、`shape`、`size[3]`、`rel_offset[3]`（四舍五入到 3 位）、`score_band`、`size_band`、`fail_phase`（仅 failure）、`attempts`、`ts`、`exp`（自然语言描述）`rag.py:247-266`
- **`**extra` 原样序列化**：调用方传什么写什么，例如 `control_mode` / `control_hints` / `fail_cat`（`rag.py:212`；写入点 `agent.py:265-271`）

写入门槛（`store.py:112-120`）：`scope`/`kind`/`confidence` 必须在枚举内，且 `evidence.cells` **必须是非空 list**。

特殊条目 `grasp_stats`（固定名、`kind=infra`）：字段 `table: {"{shape}|{task}|{band}": {succ, fail}}`（`rag.py:220-232`）。
> ⚠️ 它**必须绕过 `MemoryStore.add`** 直接写文件，否则 `add` 的 evidence 合并会用旧值覆盖新值（`rag.py:220-221`）——这是"复用 store 但语义不匹配"的破坏点。

### 1.3 生命周期：时钟是 episode 计数，不是挂钟

- `self.ts` 由 `tick()` 每 episode +1（`rag.py:140,234-236`）；调用点都是 `tick()` 紧接 `decay()`（`runner.py:232-234`、`agent.py:307-308`、`runner_dynamic.py:1172-1174,1365-1367`）
- 失效判定：`is_failed()` 中 `self.ts - ts > FAILURE_TTL(20)` → 直接跳过（`rag.py:293-306`）
- 物理清理：`decay()` 删除**过期的 failure**（只删 failure，success 永久保留），返回删除条数（`rag.py:367-379`）
- 计数语义：`n_failure` 只统计 TTL 内的 failure；`n_success` 统计 `kind == "strategy"`（`rag.py:387-393`）

**v2→v3 的三条动机**（`rag.py:2-17`）：v2 是内嵌 JSON 且有两套互不相通的记忆；v2 用"卡死时全删该 shape 失败记录"的粗暴做法 → v3 改为 TTL 自动失效 + `decay()`；匹配维度从 `shape` 扩到 `(shape, task, size_band)`。

### 1.4 持久化细节

- 默认目录 `darwin/memory/experience/`（`rag.py:32`），可用 `RAGMemory(root=...)` 覆盖（测试普遍用 `tempfile.mkdtemp()`）
- **写入非原子**（`store.py:78` 直接 `write_text`）——注意与 `SkillConfigStore.save()` 的原子写（tmp+rename）形成对比
- `save()` 是空操作（写入即时落盘）；`load()` = `_load_all()`
- 内存缓存 `self._cache` + `self._stats`；`fresh=True` 会**删光目录下所有 .md**（`rag.py:150-154`）
- 损坏文件在 `list_all()` 里被静默跳过（`store.py:88-96`）

### 1.5 检索与排序：默认不是向量，是 TF-IDF

- `_SemanticIndex`（`rag.py:45-101`）：**默认纯 TF-IDF 余弦**，词表来自 `re.findall(r"[a-z0-9_]+", lower)`；`idf = log((n+1)/(df+1)) + 1.0`；相似度线性映射到 0~1 `clip((cos+1)/2, 0, 1)`
- **embedding 是可选且默认关闭**：仅当 `DARWIN_RAG_EMBED == "1"` 才尝试 `sentence_transformers(all-MiniLM-L6-v2)`，任何异常静默回退 TF-IDF（`rag.py:52-63,66-73`）
- 检索语料是**字符串拼接的元数据**，不是原始观测：`f"{task} {shape} {score_band} {exp} {applies_when}"`（`rag.py:315-317`）

三个排序公式（全是实数常量，无学习）：

| 函数 | 公式 |
|---|---|
| `best_success` | 先按 `shape` 相等 + `task` 匹配（`task=="any"` 通配）硬过滤；`score = 0.5*sem + 0.5*center_closeness`，其中 `center_closeness = 1 - min(‖rel_offset‖, 2.0)/2.0`——**越靠物体中心越优先** |
| `retrieve(k=5)` | 先按 sem 降序，再 `boost()` 二次排序：`size_band` 命中 `+0.1`、`control_mode` 命中 `+0.2` |
| `rank_candidates` | **UCB 式**：`prio = rate + 0.15 / sqrt(1 + n)`，降序；`rate` 取 `success_rate(shape, task, band)`（键 `shape|task|band`，退化查 `shape|any|band`），无数据返 `(0.5, 0)`，有数据用拉普拉斯平滑 `(succ+0.5)/(n+1)` |

### 1.6 在 runner 里的实际候选管线

`runner_dynamic._build_candidates:215-256` 的顺序是：构造 meta → **`rel_offset` 合法性硬门槛**（`any(abs(v) > 2.0)` 丢弃）→ `rag.is_failed()` 过滤 → `rag.rank_candidates()` → `_constraint_rank()` 几何降权（`need = FINGER_R_M + 0.004`）→ 黑名单过滤 → 中心优先排序 + 中心候选插到第 0 位（除非被 RAG 标失败）。
`_inject_memory():249-262` 用 `best_success` 重建绝对抓取点并置顶，并叠加 `‖rel_offset‖ > 1.5` 的硬过滤。

## 2. 跨进程学习（`darwin/ipc/`）

### 2.1 文件与拓扑

| 文件 | 职责 |
|---|---|
| `ipc/protocol.py` | 消息 schema + `JsonLineConnection`（AF_UNIX + 换行 JSON）|
| `ipc/sim_worker.py` | **server 端**常驻仿真进程：env 只创建一次，按 `run` 反复执行 attempt |
| `ipc/agent_learner.py` | **client 端**反思学习进程：下发参数、分析结果、写回 YAML |
| `ipc/batch_learner.py` | 全任务批量学习：任务队列 + 三重预算 + 台账 + 哨兵回归/回滚（**本身不 import mujoco**，为每任务 spawn 子进程）|

```
agent_learner (决策方)  ──unix socket / 换行 JSON──►  sim_worker (唯一写 env 状态)
        │                                                      │
        └── 写 skills/configs/ik_servo.<task>.yaml（原子写）──► sim 只读
        └── 写 candidate_blacklist.json ─────────────────────► sim 每 attempt 重读
                                                            └── 写 physics.<env>.yaml 观测
```

### 2.2 通道机制

- 传输：`AF_UNIX, SOCK_STREAM`，每消息一行 JSON + `\n`，`ensure_ascii=False`（`protocol.py:108-138`）
- 路径默认 `/tmp/darwin_sim.sock`，协议版本 `PROTOCOL_VERSION = 1`，握手校验不匹配即报错（`protocol.py:39-40`；`sim_worker.py:388-395`）
- server：`listen()` 先 `os.unlink` 残留 socket → `listen(1)` → `chmod 0o660`；退出时 unlink。**只 accept 一条连接**；agent 断线后回到 accept 等重连，**env 不销毁**（`sim_worker.py:355-411`）——这正是"常驻仿真"省掉重复建环境的关键
- client：`connect(retries, retry_interval)` 固定间隔重试（应对 agent 先于 sim 启动），默认 `connect_retries=180`；**读超时 900s**（注释："attempt 可能跑数分钟，给 15 分钟读超时"）
- **强制约束**：字段只用 JSON 原生类型，**禁止 numpy 标量**（`protocol.py:4-6`）
- **文件旁路通道**（不是 socket）：`candidate_blacklist.json`，位置在 sock 目录下，**会话作用域**（不跨任务残留、无 RAG TTL 漂移）

### 2.3 消息 schema

| 方向 | 消息 | 字段 |
|---|---|---|
| agent→sim | `hello` | `{type, protocol, task}` |
| agent→sim | `run` | `{type, attempt, cfg(全量参数), resume?{rollback_steps}}`——带 `resume` 表示**回退物理状态后续跑，sim 不 reset** |
| agent→sim | `stop` | `{type}` |
| sim→agent | `ready` | `{type, protocol, task, param_spec, initial_cfg}`；`param_spec` 来自 `public_spec()`，结构 `{key:{type,default,min,max,desc}}` |
| sim→agent | `skill_event` | `{type, attempt, idx, action, success, reason, steps, tcp, body}` |
| sim→agent | `attempt_result` | `{type, attempt, success, fail_phase, failed_action, resume_action, resumable, n_resume, kind, rollback_steps, measures, telemetry, steps, methods_used, events, log_path}` |
| sim→agent | `error` / `bye` | `{type, message}` / `{type}` |

> ⚠️ **已核对的 schema 漂移**：`sim_worker._emit_result` 实际还会发 docstring 未列出的 `ok_all / mechanism / cand_xy / traj / physics`（`sim_worker.py:322-341`）。而 `agent_learner._apply_theta_cuts` 依赖 `traj`、`_directive_blacklist_write` 依赖 `mechanism / failed_action / cand_xy`——**文档 schema 落后于实现**。

### 2.4 `agent_learner` 学习循环（决策树）

构造：`AgentLearner(task, sock_path, max_attempts=15, skill_name="ik_servo", use_llm=False, max_resume=2, ...)` + `SkillConfigStore.load(...)` + `PhysicsProfile.load()` + 连 socket。

`run()` 主循环（`:216-355`）：

1. 握手 `hello` → 等 `ready`，校验协议版本
2. 循环条件：`attempt <= max_attempts and total_iters < max_attempts * 3`
3. 发 `run`：`mode=="resume"` 带 `rollback_steps`（默认 10），否则全新 attempt
4. 收流直到 `attempt_result`（`error`/`bye`/超时/断连 → 直接返回 False）
5. **成功** → 记历史 + `mark_learned(True)` + `save()`，更新 `_success_cfg` 快照，break
6. **失败** → `fail_streak[fail_phase] += 1`（其他 phase 清零）；
   - **振荡保护**：`streak >= 4` 且存在成功快照且当前参数有差异 → **回退到末次成功快照**（注释："规则调参已在发散（来回改同一参数）"）
   - 否则 `adapt_cfg(cfg_snapshot, fail_phase, streak, tel, failed_action, anchor=self._success_cfg, place_dir=..., mechanism=result["mechanism"])` → `store.update_params(deltas)`
7. 候选级失败：`_directive_blacklist_write(result)`（门控 `mechanism ∈ DIRECTIVE_MECHS` ∧ `failed_action ∈ GRASP_PHASE_ACTIONS` ∧ `cand_xy` 非空；**命中两次才生效**）
8. **resume vs reset**：`resumable and n_resume < max_resume` → `mode="resume"`（attempt 号不变）；否则 `mode="attempt"; attempt += 1`
9. 用完仍未成功 → `mark_learned(False)`（保证 per-task YAML 与台账状态一致，下轮 `build_queue` 仍会重跑）
10. 末尾输出机器行 `LLM_TOKENS=<n>` 供 batch 解析

每轮还会跑三条旁路写回：`_update_physics`（用 `result["physics"]` 更新观测桶并立即 `save()`——注释："agent 进程每任务重启，桶必须跨任务子进程积累"）、`_apply_theta_cuts`（收集 `traj` 里的 `theta_cuts` 与后验求交）、可选的 `_maybe_llm_reflect`（默认关闭）。

### 2.5 sim 侧的 resume 物理实现

- 每物理步把状态快照进环形缓冲：`env._darwin_post_step = self._snapshot`，`deque(maxlen=_CHECKPOINT_MAXLEN=300)`（注释：**LIBERO 20Hz 下 300 步 ≈ 15 秒**）
- **回退点必须在续跑 skill 自己的执行段内**——注释给出反例："`goal_not_reached` 时终态在 `open_gripper` 之后，相对终态回退会落到碗已释放的状态，重跑 place 没有意义"（`sim_worker.py:136-146`）
- 可续跑判定：`_RESUMABLE_PHASES = {place_failed, goal_not_reached, timeout}` ∧ `_RESUMABLE_ACTIONS = {place, carry, lift, open_gripper}` ∧ 只有一个 objective ∧ 快照非空
- 两种 resume 失败相位：`resume_unavailable`（缺缓存候选/method/body0）、`rollback_failed`（`restore_physics` 抛异常），都返回 `resumable=False`

## 3. 调参常量

**记忆层（`memory/rag.py`）**

| 常量 | 值 | 含义 |
|---|---|---|
| `OFFSET_TOL` | `0.5` | 归一化 offset 匹配半径（half-size 单位）|
| `FAILURE_TTL` | `20` | 失败记录存活 episode 数 |
| `OFFSET_SANITY` | `1.5` | `‖rel_offset‖` 超过即视为边界异常（**仅约定，调用方保证**）|
| size_band 边界 | `tiny<0.02≤small<0.04≤mid<0.09≤large`（m，取 half-size 最大维）| 尺寸分级键 |
| UCB 探索系数 | `0.15` | `prio = rate + 0.15/√(1+n)` |
| Laplace 平滑 | `(succ+0.5)/(n+1)`；无数据 `(0.5, 0)` | 成功率先验 |
| best_success 权重 | `0.5/0.5` + 中心度截断 `2.0` | 语义 vs 中心度 |
| retrieve 加权 | size_band `+0.1`；control_mode `+0.2`；`k=5` | 检索重排 |
| 置信度阈值 | `cells≥3 且 tasks≥2` → verified；`cells≥2` → probable；否则 single-shot | 自动升级 |

**IPC（`ipc/`）**

| 常量 | 值 | 含义 |
|---|---|---|
| `PROTOCOL_VERSION` | `1` | 协议版本（握手校验）|
| `DEFAULT_SOCK_PATH` | `/tmp/darwin_sim.sock` | 默认 socket 路径 |
| `DEFAULT_MAX_RESUME` | `2` | 同一失败点最多原位续跑次数 |
| `_ITER_CAP_MULTIPLIER` | `3` | 总迭代硬上限 = `max_attempts × 3` |
| agent 读超时 | `900 s` | 单次 attempt 等待上限 |
| `connect_retries` | `180`（间隔 1.0s）| 等 sim 启动 |
| 振荡回退阈值 | `streak ≥ 4` | 回退末次成功快照 |
| `_CHECKPOINT_MAXLEN` | `300` | 物理快照环形缓冲深度 |
| `_RESUMABLE_PHASES` | `{place_failed, goal_not_reached, timeout}` | 可续跑失败类别 |
| `_RESUMABLE_ACTIONS` | `{place, carry, lift, open_gripper}` | 可续跑动作 |
| `BLACKLIST_XY_TOL` | `0.02 m` | 候选 xy 身份匹配 |
| `BLACKLIST_STRIKES` | `2` | 同一候选两次失败才拉黑 |
| `DIRECTIVE_MECHS` | `{no_grip_air, geometry_squeeze, contact_blocked, ik_unreachable}` | 允许触发换候选的机制 |
| `GRASP_PHASE_ACTIONS` | `{move_above, descend, ik_servo, pose_move_above, pose_descend}` | 抓取相门控 |
| batch 三重预算 | `max_hours=1.0`、`per_task_attempts=8`、`sentinel_every=10` | 批量学习约束 |
| `SENTINEL_TASK` | `libero_spatial:0` | 哨兵回归任务 |

**反思（`agents/reflection.py`，IPC 循环实际调用的调参函数）**

| 常量 | 值 | 含义 |
|---|---|---|
| `_INFRA_PHASES` | `{resume_unsupported, resume_unavailable, rollback_failed, sim_exception, unknown_skill, no_candidate}` | 基建类失败**不改参** |
| `_DRIFT_FLOOR` | 每参数步长下限（`hover 0.02`、`k 1.0`、`k_descend 0.5`、`timeout_scale 0.2`、`place_k 0.3`、`place_timeout 25`、`release_offset 0.005`…）| 锚定窗口下限 |
| 锚定窗口 | `anchor ± max(0.5·|anchor|, _DRIFT_FLOOR[key])`，单次反思不超过 | 防发散 |
| `_XY_ON_THRESHOLD` / `_XY_HYSTERESIS` | `0.030 / 0.008 m` | BDDL `On` 阈值附近的滞环带 |
| accel | `1.0 + 0.3·(streak−1)` | 连续同种失败加速 |
| ratio 硬地板 | `0.62`（`grasp_container_offset_ratio`）| 偏心比下限 |
| `PARAM_SPEC` | 约 50 项 `key: (类型, 默认, min, max, 说明)` | 越界被 **clip** 而不是让 sim 崩 |

## 4. 实证结论与陷阱（摘录）

| # | 结论 |
|---|---|
| 1 | **黑名单双触活**：`agent_learner.py:104-108` 注明"（`r14` 实证单次即拉黑杀光候选树）"；常量必须与 sim 侧**同源**（延迟 import `runner_dynamic` 复用），"两处漂移会把黑名单写成误杀器" |
| 2 | **sim 启动必须清空黑名单**："（`r14` 实证 `/tmp` 残留会把上一轮的误杀带进下一轮）" |
| 3 | **回退点陷阱**：必须在续跑 skill 自己的执行段内（见 §2.5） |
| 4 | **参数振荡/"调好一个弄坏一个"**：连续 4 次同类失败回退快照；`_DRIFT_FLOOR` 的存在理由是"`k_descend` 被反思改乱"；"resume 被拒一次就被加深一次 `k_descend`"→ 由此产生 `_INFRA_PHASES` 不改参 |
| 5 | **滞环的必要性**：`goal:8/9` 实测不加滞环时 `place_k/release_offset` 会随 `fail_phase` 交替反向、在 resume 循环里无限往返 |
| 6 | **偏心比硬地板**：`ratio<0.6` 时插指进不了容器侧壁、8/8 夹空（`spatial:9` 死亡螺旋 0.70→0.52）|
| 7 | **几何类机制参数无效**：`geometry_squeeze` 只出 note 交 derive/换候选；"`k_descend` 对 libero IK 链无效"；无机制且无遥测时**保持 cfg**（旧 `other/unknown` 的"保守加深逼近"是盲方向、已删）|
| 8 | **参数默认值里的实证**：`place_vcap`（"`goal:2` 实证 place 起始 3~7 步内 slipped 8/8"）、`stop_above=-0.075` 档用于高处壶体、`corridor_z_max 0.865 = 沿口+5cm 实证安全` |
| 9 | **死因在候选空间而非参数空间**：四场失败（`goal:2/3/4/5`）的共同结构；"goal:3/4 的 118 次 `ik_unreachable` 是下降途中被邻接物楔偏" |
| 10 | **LIBERO 的 `rel_offset` 恒为 `[0,0,0]`**，候选身份只能用 `position` 的 xy |
| 11 | **台账/队列失效边界**：`libero_100` 的 task map 损坏 → 队列不含；基线哨兵不过则**不开批** |
| 12 | **sim 单次 attempt 异常不杀进程**（`_send_exception`），失败归 `fail_phase="sim_exception"`，而该 phase 属 `_INFRA_PHASES` → **不改参** |

## 5. 已知落差与限制

1. **`ipc/` 与 `memory/` 未连接**（§0 边界一）。即：走 IPC 批量学习路径时，`darwin/memory/` 的经验库**不会被写入**——实际核对也证实 `memory/experience/` 里现存 8 个文件都是早期（9 月中）残留，`grasp_stats.md` 的 table 只有两个 key。
2. **`ragbot/` 是复制分叉且无代码消费者**（§0 边界二）。两份 `MemoryStore`/`_SemanticIndex` 已在设计文档里被标为重复，但尚未合并。
3. **消息 schema 落后于实现**：`attempt_result` 实际多发 `ok_all/mechanism/cand_xy/traj/physics` 五个字段，而 agent 侧已经依赖它们。
4. **两份持久化的可靠性不一致**：`MemoryStore`（及 `RAGMemory`）**非原子写**；`SkillConfigStore.save()` 是**原子写**。
5. **`memory/failures/` 是空目录**：失败记录与成功记录同写 `experience/`，目录名有误导性。
6. **`OFFSET_SANITY` 只是约定**：不在这层强制，靠调用方过滤（`agent.py:254` 越界置零、`runner_dynamic.py:219-222` 直接丢弃候选）。
