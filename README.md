<br />

# EvoBot

机器人操作，只靠 Agent + Skills

**不用 RL · 不用 VLA · 不用 WAM**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.en.md) · [设计文档索引](docs/README.md) · [LIBERO 排障纪要](docs/12_libero_debug_2026-09-30.md)

***

## 这是什么

EvoBot 是一个面向**机器人操作**的**自进化框架**。它**不**使用强化学习、视觉-语言-动作模型或世界模型，只用 **Agent 组合可复用技能** + 记忆，从每次执行中学习——从而**需要的 LLM 调用更少、尝试次数更少**。

框架分两层：**上层 — 自进化循环：**

| 组件                | 做什么                            |
| ----------------- | ------------------------------ |
| **技能（Skills）**    | 参数化原语 + 自动生成的策略，Agent 组合它们完成任务 |
| **RAG 记忆**        | 每次决策前注入历史成功/失败；跳过已知坏偏移，复用已知好偏移 |
| **进化（Evolution）** | MAP-Elites 搜索配置；成功轨迹蒸馏为代码技能    |

**底层 — 操作能力：**

| 能力         | 做什么                                                                |
| ---------- | ------------------------------------------------------------------ |
| **IK 控制**  | CARTIK 关节空间控制，长距离运动稳定（替代不稳定的 CARTIMP）                              |
| **力控**     | `guarded_move`、`impedance_push`、`spiral_search`，由 `force_ee` 传感器终止 |
| **视觉伺服**   | `servo_align` — 基于重投影误差闭环对齐，实现最后厘米级精度                              |
| **MuJoCo** | 物理引擎，提供环境、接触动力学、真值感知                                               |

解决什么问题：

LLM 驱动的机器人操作有三个反复出现的成本：

1. **Token 浪费** — LLM 每个 episode 都从零重新规划同一个抓取放置任务。
2. **重复失败** — 没有"这个抓取偏移上次打滑了"的记忆，机器人反复重试同一个坏动作。
3. **无法泛化** — 在一个物体/机器人上学到的策略很难迁移到另一个。

EvoBot 将**经验视为一等的、可查询的资源**，从根本上解决这三个问题。

## 设计文档

`docs/` 下 16 篇模块设计文档。写作约定：每篇只写**能从代码读出来的事实**，关键论断带 `file:line` 证据；每篇末尾单列"**已核对的落差与限制**"，如实记录注释与实现不符之处。索引见 **[docs/README.md](docs/README.md)**。

**robopal 栈**（LLM 决策 + RAG 记忆，01–07）：

| 编号 | 文档 | 核心内容 |
|---|---|---|
| 01 | [RAG 记忆层](docs/01_rag_memory.md) | 归一化经验、失败 TTL、语义检索、UCB 排序、置信度升级 |
| 02 | [智能体与执行器](docs/02_agent_runner.md) | 五步自进化闭环、去分发架构、候选生成、多尝试重试 |
| 03 | [技能框架](docs/03_skills_framework.md) | 三层技能体系、SkillSpec 元数据、控制/运动/感知原语 |
| 04 | [进化系统](docs/04_evolution_system.md) | MAP-Elites 质量多样性、skill_forge 成功轨迹沉淀（设计叙述） |
| 05 | [机器人后端抽象](docs/05_robot_backend.md) | RobotBackend ABC、YAML Profile、机型解耦 |
| 06 | [LLM 集成](docs/06_llm_integration.md) | OpenAI 兼容协议、结构化输出、规则回退 |
| 07 | [感知技能](docs/07_perception.md) | YOLO/SAM/GraspNet 双模式、真值采样路径、随机源全景、GPU 前向与不可复现性 |

**LIBERO 基准栈**（把 BDDL 当形式化规格直接解析，08–12）：

| 编号 | 文档 | 核心内容 |
|---|---|---|
| 08 | [LIBERO 规划与执行栈](docs/08_libero_planning_stack.md) | BDDL→有序子目标、谓词→技能映射、机制驱动重规划、attempt 闭环与双判据超时 |
| 09 | [LIBERO 技能库](docs/09_libero_skills.md) | `serve`/`goto` 底座、grasp 候选枚举与 13 道门、`mem` 记忆字典、流形跟随 |
| 10 | [物理机制判别与退路](docs/10_physics_mechanisms.md) | 10 种失败机制及判别依据、Θ 割与几何指令两类退路 |
| 11 | [环境适配层](docs/11_envs_libero_adapter.md) | 固定初始状态、horizon 回卷、`servo_step` 统一动作语义、`AttemptStepLimit` |
| 12 | [调试纪要 2026-09-30](docs/12_libero_debug_2026-09-30.md) | 一次实际排障全记录：失败归因、`turnon` 端点 bug 取证链、不可复现性排查 |

**进化 / 执行 / 记忆 / 外围**（13–16，补 04–06 的实现细节）：

| 编号 | 文档 | 核心内容 |
|---|---|---|
| 13 | [进化系统（实现）](docs/13_evolution_map_elites.md) | MAP-Elites 分箱与算子、forged 链路的置信度升级、锚点重投影 |
| 14 | [动态执行路径](docs/14_dynamic_runner.md) | 条件驱动执行骨架、方法库与选择、**机制的三条消费路径** |
| 15 | [记忆层与跨进程学习](docs/15_memory_and_ipc.md) | RAGMemory v3（episode 时钟/TTL/TF-IDF/UCB）、ipc socket 协议与 resume |
| 16 | [外围子系统](docs/16_outer_subsystems.md) | LLM JSON 约定与 grounding 校验、RobotBackend 与 mplib、benchmark、下载镜像 |

## 如何工作：自进化闭环

![EvoBot 架构图](figs/Evobotarc.png)

**执行前**，RAG 过滤近期失败的抓取偏移，给出最佳已验证偏移；**执行后**，结果写回——成功变成可复用策略，失败变成临时陷阱（TTL 过期，非永久拉黑）。

技能分层：

```
策略技能（forged，由成功轨迹自动生成）
        │  调用
控制/运动原语（servo_align, guarded_move, path_plan, ...）
        │  调用
感知技能（detect, segment, grasp_pose）
```

## 两种运行形态

仓库里是**并列的两套栈**，共用底层技能与物理机制模块，但规划方式相反：

| | **robopal 栈** | **LIBERO 基准栈** |
|---|---|---|
| 入口 | `darwin/agents/agent.py` + `runner_dynamic.py` | `darwin/agents/libero_runner.py` |
| 任务来源 | 静态 benchmark 表 / YAML | BDDL 形式化规格（`task_spec.py` 解析） |
| 规划方式 | LLM 决策 + RAG 记忆 + 方法库，允许自适应 | **不猜**：BDDL→有序子目标（稳定拓扑排序） |
| 失败处理 | 机制驱动的步级/attempt 级调参、候选黑名单 | 机制驱动重规划 + `mem` 记忆字典 |
| 超时判据 | — | **仿真步数为主判据**（确定性），挂钟仅为兜底 |
| 文档 | 01–07、13–16 | 08–12 |

## LIBERO 基准

沿用 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 的环境与任务定义，但用上面那套确定性栈跑（不用 LLM 猜子目标）。

```bash
# 单任务（30 次尝试上限），走 IPC 常驻仿真 + 反思学习
PYTHONPATH=$PWD:/path/to/LIBERO MUJOCO_GL=egl python -u scripts/ipc_learn.py \
    libero_spatial:0 --max-attempts 30 --sock /tmp/libero.sock

# 直接跑 agent runner（按台账断点续跑、跳过已通过任务）
PYTHONPATH=$PWD:/path/to/LIBERO MUJOCO_GL=egl python -u -m darwin.agents.libero_runner \
    --suites libero_spatial,libero_object,libero_goal

# 30 任务全量回归（验收脚本）
bash scripts/verify_30.sh 8 verify
```

**当前状态（2026-09-30）**：30 个任务中 **23 个通过**（台账 `logs/libero_agent_progress.json`）。本轮修掉了 `goal:7`（`turn_on_the_stove`）的目标端点 bug——`articulation_info` 原先取关节量程的 max 侧，给出的目标角是 LIBERO 判定阈值的 3.6 倍；改取 min 侧后该任务首次尝试即通过。取证链与全部实测数据见 [docs/12](docs/12_libero_debug_2026-09-30.md)。

> ⚠️ **结果目前不完全可复现**：同一任务、同一固定初始状态下，重跑得到的结论会变。已排查并排除挂钟超时、CPU 线程数、RRT 种子；剩余噪声来自感知侧 GraspNet 的 GPU 前向（未播种的全局 `np.random` 已修，GPU 前向的残差仍在）。详见 [docs/07](docs/07_perception.md) 与 [docs/12](docs/12_libero_debug_2026-09-30.md) §4。
>
> **报告通过率时请注意：单次通过 ≠ 稳定通过。**

## 快速开始

### 安装

核心包（仅 RAG 记忆层，零 ML 依赖）：

```bash
pip install -e .
```

仿真与基准所需的额外依赖见 `requirements.txt` / `requirements-extra.txt`。

### 最小示例

```python
from ragbot import RAGMemory

rag = RAGMemory()
feat = {"shape": "box", "size": [0.05, 0.05, 0.05]}

# 执行前：从历史学习
best = rag.best_success("pick", feat)             # 已验证的抓取偏移
bad  = rag.is_failed("pick", feat, [0.2, 0, 0])  # 这个偏移近期是否失败过

# 执行后：写回经验
rag.add_success("pick", feat, rel_offset=[0.05, 0, 0], score_band="mid", attempts=2)
rag.add_failure("pick", feat, rel_offset=[0.2, 0, 0],  score_band="mid", fail_phase="descend")
rag.tick()    # 推进失败 TTL 时钟
rag.decay()   # 清理过期失败记忆
```

### 运行进化循环（仿真）

```bash
python scripts/run_evolution.py --task pickplace --episodes 20
```

## 架构

```
ragbot/                  公开包 — RAG 记忆层（仅 numpy + pyyaml）
  memory/
    store.py             YAML frontmatter 存储、evidence 合并、置信度升级
    rag.py               归一化偏移、TTL、语义检索、UCB 排序
    experience/          种子经验（.md，可 git 追踪）

darwin/                  仿真与进化验证框架
  agents/                ManipulationAgent（五步闭环）、runner_dynamic、
                         libero_runner / libero_planner / libero_skills（LIBERO 栈）
  skills/                base.py、primitives/、perception/、forged/、configs/
  evolution/             MAP-Elites、skill_forge、EvolutionLoop
  physics/               失败机制判别、Θ 割退路、参数推导、后验估计
  policies/              步级决策点（换参 / 规则 / 参数空间）
  memory/                darwin 侧记忆层（与 ragbot 同源分叉）
  ipc/                   常驻仿真进程 + 反思学习进程 + 批量学习
  envs/                  LIBERO 适配层
  robot/                 RobotBackend 抽象 + MujocoRobopalBackend + YAML profiles
  benchmarks/            静态 benchmark 表 + LIBERO 动态工厂 + MAP-Elites 搜索空间
  llm/                   OpenAI 兼容客户端（方舟 Ark / OpenAI）
  utils/                 权重下载、MuJoCo 树工具、EGL 录像

robopal/                 机器人操作库（本地开发副本）
docs/                    模块设计文档（01–16）+ planning/（内部规划）+ archive/ + source/（Sphinx）
scripts/                 可复现的入口脚本 + dev_probes/（一次性探针，保留供取证）+ llm_prompts/
reports/                 从产物中抢救保留的报告与台账
data/  experience/  figs/  assets/
```

### 目录约定

- `logs/`、`videos/`、`data/videos/`、`*.mp4`、`**/snapshots/` **不入库**（见 `.gitignore`）——运行产物。
- `reports/` 保留有长期价值的产物（标定报告、台账快照）。
- `docs/planning/` 是**内部规划/改进待办**，不是设计文档；设计文档只有 `docs/NN_*.md` 这一套。`docs/archive/` 存更早的设计全文。
- `scripts/dev_probes/` 是**一次性诊断探针**，归档不删——文档里引用的取证脚本（如 `probe_goal5_plate.py`）仍可复现。
- `docs/source/` 是 robopal 的 Sphinx 文档树，与上面的模块设计文档是两回事。

## 关键设计决策

- **归一化抓取偏移** `rel_offset = (抓取点 − 中心) / 半尺寸` — 经验可跨不同尺寸物体迁移。
- **失败 TTL（20 episodes）** 而非永久拉黑 — 场景变化后失败记忆自动过期。
- **失败返回机制而非布尔值** — 技能失败时给出机制（`friction_slip`/`contact_blocked`/…），执行器据此在**步级换参 / attempt 级调参 / 候选黑名单**三条路径上消费，而不是盲目重试。
- **超时用仿真步数而非挂钟** — 共享机器负载不影响判据，结果可复现（挂钟只作异常卡死的兜底）。
- **CARTIK 替代 CARTIMP** — robopal 的 CARTIMP 不稳定；改用 IK + 关节阻抗 + 力传感器做终止保护。
- **去分发执行** — Agent 直接持有 `Dict[str, Skill]`；forged 技能通过 `bind()` 复用原语。
- **机型无关的运动原语** — `motion.py` 只调 `env.backend`；加新机型 = 一个 YAML profile（有 URDF）或一个 `RobotBackend` 子类。

## 扩展

| 目标            | 做法                                                            |
| ------------- | ------------------------------------------------------------- |
| 新增有 URDF 的机型  | 写 `darwin/assets/profiles/<name>.yaml`                        |
| 新增 ROS/SDK 机型 | 继承 `RobotBackend`，注册，加 YAML profile                           |
| 新增控制原语        | 在 `darwin/skills/primitives/` 继承 `Skill`                      |
| 新增 LIBERO 技能   | 在 `darwin/agents/libero_skills.py` 加函数 + 在谓词映射里接线（见 [docs/08](docs/08_libero_planning_stack.md)） |
| 接入自定义感知栈      | 提供 `{shape, size, center, rel_offset}` —— `ragbot` 无需任何 ML 依赖 |

## License

MIT
