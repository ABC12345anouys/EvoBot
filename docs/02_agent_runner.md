# 02 · 智能体与执行器设计

> 对应代码：`darwin/agents/agent.py`、`darwin/agents/runner.py`

## 1. 设计目标

智能体层是 EvoBot 的**编排核心**，负责把感知、RAG 记忆、LLM 决策、技能执行、经验沉淀串联成一个完整的自进化闭环。设计目标：

1. **闭环可运行**：perceive → retrieve → decide → execute → reflect → tick，单函数即可驱动一次 episode。
2. **去分发**：Agent 直接持有技能实例字典，执行时无中间分发层，forged 技能可复用原语。
3. **LLM 可选**：无 LLM 配置时自动回退规则规划，链路仍可端到端验证。
4. **记忆注入**：执行前检索经验过滤候选、注入最佳偏移；执行后沉淀成功/失败。

## 2. ManipulationAgent

### 2.1 核心数据

```python
class ManipulationAgent:
    skills: Dict[str, Skill]       # 感知 + 原语 + forged 技能
    rag: RAGMemory                 # 经验记忆
    llm: Optional[LLMClient]       # LLM 决策（可选）
    task_name: str
    seed: str
    max_turns: int = 30
```

初始化时：
- `load_skills()` 加载全部技能（感知 + primitives + forged）
- forged 技能调用 `bind(self.skills)` 获得对原语的访问权
- `llm` 自动检测：配置了 API Key 则启用，否则 `None`

### 2.2 五步闭环

#### 第一步：perceive（感知）

```python
def perceive(self, env, task, candidates=None) -> (feat, candidates):
```

- 优先使用调用方传入的 `candidates`（已感知）
- 否则尝试 `grasp_pose` 技能（GraspNet）
- 兜底返回中心候选 `[0,0,0]`
- `feat` 包含 `shape` / `size` / `center`

#### 第二步：retrieve（RAG 检索）

```python
def retrieve(self, task, feat, candidates) -> dict:
    valid = [c for c in candidates if not rag.is_failed(...)]   # 过滤近期失败
    ranked = rag.rank_candidates(task, feat, valid)             # UCB 重排
    best = rag.best_success(task, feat)                         # 最佳成功偏移
    ctx = rag.retrieve(f"{task} {shape}", feat, k=3)            # 语义上下文
```

返回 `{ranked_candidates, best_success, context, stats}`。

#### 第三步：decide（决策）

```python
def decide(self, task, feat, rag_ctx, plan=None) -> List[Dict]:
```

三级回退：
1. 传入 `plan`（list 或 callable）→ 直接用
2. `self.llm is not None` → LLM 多轮 function calling
3. 兜底 → `_default_plan`（规则：home → move_above → descend → close → lift → [place]）

**LLM 输出格式**：
```
<action>skill_name</action>
<params>{"param": value}</params>
```
完成时输出 `<action>finish</action><params>{"success": true}</params>`。

#### 第四步：execute（执行）

```python
def execute(self, plan, env) -> (trajectory, success):
```

**去分发架构**：直接 `self.skills[name].execute(env=env, **params)`，无 registry 中间层。

- 未知技能 → 立即返回失败
- 技能抛异常 → 捕获并标记失败
- 任一步失败 → 停止执行，返回已执行的 trajectory
- 全部成功 → 返回完整 trajectory + `success=True`

`trajectory` 每步包含 `{action, params, result, step_idx}`，`result.success` 标记该步成败。

#### 第五步：reflect（沉淀）

```python
def reflect(self, task, feat, trajectory, success, grasp_pt, control_mode, control_hints):
```

- 计算 `rel_offset = (grasp_pt - center) / half_size`，超 `OFFSET_SANITY` 归零
- 成功 → `rag.add_success(...)` + `_maybe_forge()` 尝试生成 forged skill
- 失败 → 定位首个失败步作为 `fail_phase` → `rag.add_failure(...)`
- `control_mode` / `control_hints` 透传到 RAG 元数据

### 2.3 run_episode 主入口

```python
def run_episode(self, env, task, feat=None, candidates=None, plan=None,
                grasp_pt=None, control_mode=None, control_hints=None) -> EpisodeResult:
    feat, candidates = perceive(...)          # 1
    rag_ctx = retrieve(task, feat, candidates) # 2
    actions = decide(task, feat, rag_ctx, plan) # 3
    trajectory, success = execute(actions, env) # 4
    reflect(task, feat, trajectory, success, grasp_pt, ...)  # 5
    rag.tick(); rag.decay()                    # 6
    return EpisodeResult(success, trajectory, summary, info)
```

## 3. EpisodeRunner

`EpisodeRunner` 面向 **benchmark 任务**，在 `ManipulationAgent` 之上增加候选生成、计划编排、多尝试重试、视频录制等能力。

### 3.1 候选生成（RAG 增强）

`_build_candidates(env, feat)`：

1. `grasp_candidates_from_env` 获取 GraspNet 候选点
2. 对每个候选计算 `rel_offset` 和 `score_band`
3. **RAG 失败过滤**：`rag.is_failed` 为真的候选跳过
4. **RAG UCB 排序**：`rag.rank_candidates` 重排
5. **中心候选兜底**：物体中心 `rel_offset=[0,0,0]` 若未失败则插队到首位
6. **全失败保护**：所有候选被过滤时强制保留中心候选，避免卡死

`_inject_memory(env, feat, candidates)`：

1. `rag.best_success` 获取最佳成功经验
2. 校验该偏移未失败且 `|rel_offset| <= 1.5`
3. 将绝对抓取点插入候选队首，标记 `from_memory=True`

### 3.2 计划编排

`_plan_for(cand, cfg, goal, env)` 按 `entry["mode"]` 分支：

| mode | 计划模板 |
|------|----------|
| `pick` | home → move_above → descend → close → lift |
| `full` (pickplace) | pick 流程 + move_to_xy_top + place + open |
| `insert` | pose_home → pose_move_above → ... → servo_align → guarded_move → impedance_push → spiral_search |
| `drawer` | 拉抽屉关节 → 抓方块 → 放入抽屉 |

`insert` 模式体现了**分层控制哲学**：远距离迁移用位姿原语，最后对齐用 `servo_align`（视觉伺服），接触段用 `guarded_move` + `impedance_push` + `spiral_search`（力控）。

### 3.3 多尝试与重试

```python
for attempt in range(max_attempts):
    cand = queue.pop(0)
    # 记忆候选或近中心候选不加抖动，其余加角度抖动避免重复失败
    jit = 0 if precise else jit_scale * [sin(ang), cos(ang), 0]
    plan = _plan_for(cand, cfg, goal, env)
    trajectory, ok = agent.execute(plan, env)
    if ok:
        reflect(success=True)
        break
    else:
        reflect(success=False, fail_phase=last_action)
        # 物体被推偏或出界 → reset 环境并重新感知
        if pushed or out_of_range:
            env.reset()
            queue = []
```

**抖动策略**：非记忆候选加确定性角度抖动（`ang = 2.4*attempt + 1.9`），避免每次尝试完全相同的抓取点导致重复失败。

### 3.4 物理验证

`drawer` 模式有额外物理验证：
- 抽屉关节 `qpos > 0.08`（真的拉开了）
- 方块位置与目标距离 `< 0.05m`（真的放进去了）

不满足则即使执行返回 `success=True` 也判定为失败。

## 4. 去分发架构的优势

传统技能系统通常有一个 `Registry` 中间层，Agent 通过 `registry.execute(name, **params)` 间接调用。本项目采用**去分发**：

```
传统: Agent → Registry → Skill.execute(...)
本项目: Agent.skills[name].execute(env, **params)
```

优势：
- **少一层间接**，调用栈更浅，调试更直接
- **forged 技能复用原语**更简单：`bind(skills)` 后直接 `self.skills[name].execute(...)`
- **技能表可注入**：测试时可传入 mock skills，无需改 registry

## 5. 环境缓存与 monkeypatch

`_get_env` 维护 `_ENV_CACHE`，同一 `(env_id, robot, controller)` 复用 env 实例，避免反复加载 MJCF。

同时 monkeypatch `MujocoEnv.close`：
```python
MujocoEnv.close = lambda self: self.renderer.close()
```
原因：robopal 原生 `close()` 调用 `os._exit(0)` 会终止整个进程，多 episode 运行时必须覆盖。
