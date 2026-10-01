<div align="left">

# EvoBot

**用 BDDL 形式化规格 + 可复用技能跑 LIBERO 机器人操作基准**

不用 RL · 不用 VLA · 不用世界模型 · 运行期不调 LLM

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Benchmark](https://img.shields.io/badge/benchmark-LIBERO-orange)](https://github.com/Lifelong-Robot-Learning/LIBERO)

[English](README.en.md) · [设计文档](docs/README.md) · [排障纪要](docs/12_libero_debug_2026-09-30.md)

</div>

---

## 目录

- [这是什么](#这是什么)
- [核心设计](#核心设计)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [任务定义：形式化规格与 LLM 分解](#任务定义形式化规格与-llm-分解)
- [项目结构](#项目结构)
- [设计文档](#设计文档)
- [当前状态与可复现性](#当前状态与可复现性)
- [已知限制](#已知限制)
- [引用与许可](#引用与许可)

## 这是什么

一个跑 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 机器人操作基准的执行栈。它把任务的 BDDL 定义当**形式化规格**，而不是让模型自由发挥：

1. **BDDL → 有序子目标**。`On / In / Open / Close / Turnon / Turnoff` 被解析成结构化子目标；LLM 只在**离线生成阶段**读这份规格做一次「谓词级分解」（决定执行顺序、决定要不要插"先开抽屉"这类前置），产物冻结成 YAML 提交仓库，**运行期零 token、完全确定**。
2. **子目标 → 技能**。谓词类别映射到参数化技能（`place → grasp + place_at`、`articulate → articulate`、`toggle → toggle`），与任务名零耦合。
3. **技能失败 → 返回机制**。技能失败时返回的是**失败机制**（`friction_slip`、`contact_blocked`、`ik_unreachable`、`reach_limit` …）而不是布尔值；执行器据此换参数、换抓取候选或走几何退路。
4. **确定性截断**。单条轨迹的判据是**仿真步数**（`--attempt-steps`，默认 7000），挂钟超时只作异常卡死的兜底——避免机器负载影响结论。

```
BDDL ──► task_spec ──► libero_planner ──► libero_skills ──► envs/libero_adapter ──► LIBERO/robosuite
             │              │                  │                    │
             │              │                  └── 失败机制 ──► physics 判别与退路
             └──────────────┴──► libero_runner：attempt 闭环 / 双判据超时 / 台账 / 断点续跑
```

## 核心设计

| 决策 | 为什么 |
|---|---|
| **LLM 分解一次、冻结入库** | 顺序不再是硬编码启发式（不再靠"名字里含 cabinet 就以为要开柜门"），同时运行期零 token、计划完全可复现。`validate_order()` 是纯函数校验（id 不重不漏、`add` 只能取候选），校验不过就回退确定性拓扑排序。 |
| **确定性拓扑排序保留为参考序** | 它同时是 fallback、LLM 输出的校验依据，以及写进 YAML 的 `reference_order`（用于 diff 与回归）。 |
| **失败返回机制而非布尔值** | 不同失败需要不同对策（被挡 → 更软更慢；够不到 → 换候选/加时间），布尔值丢掉了这个信息。 |
| **步数判据优先于挂钟** | 结果不应随共享机器的负载变化。 |
| **规格即产物** | 每个任务的定义落成 `darwin/skills/configs/task_specs/<suite>_<idx>.yaml`，可读、可 diff、可重放。 |

## 环境要求

- **Python 3.10**（conda 环境，实测用 `python=3.10`；3.12 跑不了 robosuite/MinkowskiEngine 那一套）
- **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 源码**（需在 `PYTHONPATH` 上）
- MuJoCo + robosuite（LIBERO 自带）；无显示环境需要 `MUJOCO_GL=egl`
- 依赖清单：`requirements.txt`（本项目）、`requirements-extra.txt`（录像/规划相关）

```bash
conda create -n darwin python=3.10 -y && conda activate darwin
pip install -e .                    # 安装本项目（darwin 包）
pip install mujoco h5py numpy pyyaml  # 运行所需的基础依赖
# LIBERO 及其依赖按其官方说明安装，然后：
export PYTHONPATH=$PWD:/path/to/LIBERO
export MUJOCO_GL=egl
```

## 快速开始

```bash
# 1) 单任务：常驻仿真进程 + 反思学习（默认 30 次尝试上限）
PYTHONPATH=$PWD:/path/to/LIBERO MUJOCO_GL=egl python -u scripts/ipc_learn.py \
    libero_spatial:0 --max-attempts 30 --sock /tmp/libero.sock

# 2) 直接跑 agent runner：按台账断点续跑，自动跳过已通过的任务
PYTHONPATH=$PWD:/path/to/LIBERO MUJOCO_GL=egl python -u -m darwin.agents.libero_runner \
    --suites libero_spatial,libero_object,libero_goal

# 3) 30 任务全量验收
bash scripts/verify_30.sh 8 verify      # 参数：每任务尝试次数、结果标签
```

常用参数：

| 参数 | 默认 | 含义 |
|---|---|---|
| `--suites` / `--tasks` | `libero_spatial,libero_object,libero_goal` | 跑哪些 suite，或显式 `suite:idx` 列表 |
| `--attempt-steps` | `7000` | 单条轨迹仿真步数上限（**主判据**，确定性） |
| `--attempt-timeout` | `300` | 挂钟兜底秒数（只在"不走步但卡死"时可能先触发） |
| `--no-video` | — | 不录像 |

台账在 `logs/libero_agent_progress.json`，键为 `libero:<suite>:<idx>`；已通过的下一轮自动跳过。

## 任务定义：形式化规格与 LLM 分解

每个任务的定义是一份**自描述 YAML**，冻结在 `darwin/skills/configs/task_specs/`：

```yaml
language: put the bowl on top of the cabinet
source: llm                     # llm | deterministic
formal:                         # 形式化规格（LLM 与校验器都读这一份）
  items:                        # BDDL goal 谓词，带稳定 id
    - {id: p0, predicate: On, kind: place, object: akita_black_bowl_1,
       target: wooden_cabinet_1_top_side}
  implicit_candidates:          # 允许插入的隐式前置
    - {id: 'open:wooden_cabinet_1_top_side', predicate: Open, target: ...}
  vocabulary: {objects: [...], targets: [...], fixtures: {...}}
  skills: [...]                 # 可用技能约束
  reference_order: [...]        # 确定性拓扑排序（参考序，用于 diff/回归）
llm:                            # 这次分解的原始记录
  order: [p0]
  add: []
  rationale: 把碗放到柜子顶面，不涉及打开柜门或抽屉，无需插入隐式前置。
subgoals: [...]                 # 最终产物：runner 直接消费
```

**重新生成**（需要 `DEEPSEEK_API_KEY`；不设则回退确定性解析）：

```bash
python scripts/gen_task_specs.py                     # 全部 30 个任务
python scripts/gen_task_specs.py --tasks libero_goal:7
python scripts/gen_task_specs.py --no-llm            # 只写确定性规格
python scripts/gen_task_specs.py --model deepseek-v4-pro --force
```

## 项目结构

```
darwin/
  agents/       libero_runner（attempt 闭环 / 台账）、libero_planner（子目标→技能）、
                task_spec（BDDL 解析 / 形式化规格 / LLM 分解 / 校验）、
                libero_skills（技能实现）、objectives、predicates
  envs/         libero_adapter（LIBERO/robosuite → 统一动作与语义目标语义；
                horizon 回卷、AttemptStepLimit 确定性截断）
  physics/      失败机制判别、Θ 割退路、参数推导、后验估计
  skills/       perception/（YOLO/SAM/GraspNet）、primitives/（ik_servo、motion、control）
                configs/task_specs/  ← 冻结的任务定义（形式化规格 + LLM 分解）
  benchmarks/   LIBERO 动态工厂（suite, idx → env_id / goal / spec）
  ipc/          常驻仿真进程 + 反思学习进程（scripts/ipc_learn.py 用它）
  memory/       经验库（LIBERO 路径当前不写）
  policies/     步级决策点（换参 / 规则 / 参数空间）
  llm/          OpenAI 兼容客户端（供离线生成脚本用）
  utils/        权重下载、MuJoCo 树工具、EGL 录像
scripts/        gen_task_specs.py（离线生成任务定义）、verify_30.sh（验收）、
                ipc_learn.py（单任务）、learn_libero_all.py、calibrate_from_logs.py、
                bench_grasp_models.py、dev_probes/（排障取证探针，归档不删）
docs/           设计文档 07–12
reports/        标定报告与台账快照
```

约定：`logs/`、`videos/`、`*.mp4`、`**/snapshots/` 是运行产物，不入库（见 `.gitignore`）。

## 设计文档

| 编号 | 文档 | 核心内容 |
|---|---|---|
| 07 | [感知技能](docs/07_perception.md) | YOLO/SAM/GraspNet 双模式、真值采样路径、随机源全景、GPU 前向与不可复现性 |
| 08 | [规划与执行栈](docs/08_libero_planning_stack.md) | BDDL→有序子目标、**两条分解路径（LLM 冻结 / 确定性参考序）**、attempt 闭环与双判据超时 |
| 09 | [技能库](docs/09_libero_skills.md) | `serve`/`goto` 底座、grasp 候选枚举与 13 道门、`mem` 字典、流形跟随 |
| 10 | [物理机制与退路](docs/10_physics_mechanisms.md) | 10 种失败机制及判别依据、Θ 割与几何指令两类退路 |
| 11 | [环境适配层](docs/11_envs_libero_adapter.md) | 固定初始状态、horizon 回卷、`servo_step` 动作语义、`AttemptStepLimit` |
| 12 | [排障纪要 2026-09-30](docs/12_libero_debug_2026-09-30.md) | 一次实际排障全记录：失败归因、`turnon` 端点 bug 取证链、不可复现性排查 |

## 当前状态与可复现性

**30 个任务中 23 个通过**（台账 `logs/libero_agent_progress.json`，2026-09-30 快照）。未通过：`spatial:4`、`object:1/6/7/9`、`goal:5/9`。

已修的一个代表性问题：`goal:7`（`turn_on_the_stove`）的 `articulation_info` 原先取关节量程的 **max** 侧，给出的目标角是 LIBERO 判定阈值的 3.6 倍；改取 **min** 侧后该任务首次尝试即通过。取证链见 [docs/12](docs/12_libero_debug_2026-09-30.md)。

> ⚠️ **结果目前不完全可复现**：同一任务、同一固定初始状态下重跑，结论会变。已排除**挂钟超时**、**CPU 线程数**、**RRT 种子**；剩余噪声来自感知侧 GraspNet 的 **GPU 前向**（未播种的全局 `np.random` 已修，GPU 前向残差仍在）。详见 [docs/07](docs/07_perception.md) 与 [docs/12](docs/12_libero_debug_2026-09-30.md) §4。
>
> **报告通过率时请注意：单次通过 ≠ 稳定通过。**

## 已知限制

- **可复现性只做到一半**（见上）。
- **LLM 分解尚未在基准上做 A/B**：形式化规格让 LLM 能纠正名字启发式的误判（例：`wooden_cabinet_1_top_side` 是**顶面**，不该补隐式 `Open`），但"省下这一步"能带来多少收益**尚未测量**。
- **冻结产物是快照而非自动同步**：BDDL 变了需要重跑 `gen_task_specs.py`；尚无 CI 校验 YAML 与当前 BDDL 是否一致。
- **`unsupported` 谓词（如 `NextTo`）只记录不执行。**

## 引用与许可

许可：**Apache-2.0**（见 [LICENSE](LICENSE)）。

```bibtex
@misc{evobot,
  title  = {EvoBot: BDDL formal spec + reusable skills for the LIBERO benchmark},
  author = {李飞达},
  year   = {2026},
  note   = {https://github.com/ABC12345anouys/EvoBot}
}
```
