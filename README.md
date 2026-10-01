<br />

# EvoBot · LIBERO 实验栈

用 **BDDL 形式化规格 + 可复用技能** 跑 LIBERO 机器人操作基准——**不用 RL、不用 VLA、不用世界模型，也不用 LLM 猜子目标**。

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[English](README.en.md) · [设计文档](docs/README.md) · [排障纪要](docs/12_libero_debug_2026-09-30.md)

***

## 这是什么

这是一个跑 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 基准的确定性执行栈。核心取舍是**不靠 LLM 猜子目标**：

- 任务定义（BDDL）被当作**形式化规格**直接解析成有序子目标，而不是让模型去规划；
- 每个子目标映射到一组可复用的**参数化技能**（`serve` / `goto` / `grasp` / `place` / `articulate` …）；
- 技能失败时返回**失败机制**（`friction_slip`、`contact_blocked`、`ik_unreachable` …），执行器据此换参、换候选或换退路，而不是盲目重试；
- 单条轨迹的截断判据是**仿真步数**（确定性），挂钟超时只作异常卡死的兜底。

```
BDDL ──► task_spec ──► libero_planner ──► libero_skills ──► envs/libero_adapter ──► LIBERO/robosuite
             │              │                  │                    │
             │              │                  └── 失败机制 ──► physics 判别与退路
             └──────────────┴──► libero_runner：attempt 闭环 / 双判据超时 / 台账 / 断点续跑
```

## 设计文档

`docs/` 下有 6 篇，只写**能从代码读出来的事实**，关键论断带 `file:line` 证据，每篇末尾单列"已核对的落差与限制"。索引见 **[docs/README.md](docs/README.md)**。

| 编号 | 文档 | 核心内容 |
|---|---|---|
| 07 | [感知技能](docs/07_perception.md) | YOLO/SAM/GraspNet 双模式、真值采样路径、随机源全景、GPU 前向与不可复现性 |
| 08 | [规划与执行栈](docs/08_libero_planning_stack.md) | BDDL→有序子目标、谓词→技能映射、析取规划、attempt 闭环与双判据超时 |
| 09 | [技能库](docs/09_libero_skills.md) | `serve`/`goto` 底座、grasp 候选枚举与 13 道门、`mem` 字典、流形跟随 |
| 10 | [物理机制与退路](docs/10_physics_mechanisms.md) | 10 种失败机制及判别依据、Θ 割与几何指令两类退路 |
| 11 | [环境适配层](docs/11_envs_libero_adapter.md) | 固定初始状态、horizon 回卷、`servo_step` 动作语义、`AttemptStepLimit` |
| 12 | [排障纪要 2026-09-30](docs/12_libero_debug_2026-09-30.md) | 一次实际排障全记录：失败归因、`turnon` 端点 bug 取证链、不可复现性排查 |

## 快速开始

依赖：conda 环境（python 3.10）+ [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 源码，无显示环境需要 `MUJOCO_GL=egl`。

```bash
# 单任务（默认 30 次尝试上限）：常驻仿真进程 + 反思学习
PYTHONPATH=$PWD:/path/to/LIBERO MUJOCO_GL=egl python -u scripts/ipc_learn.py \
    libero_spatial:0 --max-attempts 30 --sock /tmp/libero.sock

# 直接跑 agent runner：按台账断点续跑、自动跳过已通过的任务
PYTHONPATH=$PWD:/path/to/LIBERO MUJOCO_GL=egl python -u -m darwin.agents.libero_runner \
    --suites libero_spatial,libero_object,libero_goal

# 30 任务全量回归（验收脚本）
bash scripts/verify_30.sh 8 verify
```

常用参数：`--attempt-steps 7000`（单条轨迹步数上限，主判据）、`--attempt-timeout 300`（挂钟兜底秒数）。

## 当前状态

**30 个任务中 23 个通过**（台账 `logs/libero_agent_progress.json`，2026-09-30 快照）。未通过：`spatial:4`、`object:1/6/7/9`、`goal:5/9`。

本轮修掉了 `goal:7`（`turn_on_the_stove`）的目标端点 bug：`articulation_info` 原先取关节量程的 max 侧，给出的目标角是 LIBERO 判定阈值的 3.6 倍；改取 min 侧后该任务首次尝试即通过。取证链见 [docs/12](docs/12_libero_debug_2026-09-30.md)。

> ⚠️ **结果目前不完全可复现**：同一任务、同一固定初始状态下重跑，结论会变。已排除挂钟超时、CPU 线程数、RRT 种子；剩余噪声来自感知侧 GraspNet 的 GPU 前向（未播种的全局 `np.random` 已修，GPU 前向残差仍在）。详见 [docs/07](docs/07_perception.md) 与 [docs/12](docs/12_libero_debug_2026-09-30.md) §4。
>
> **报告通过率时请注意：单次通过 ≠ 稳定通过。**

## 架构

```
darwin/
  agents/     libero_runner（attempt 闭环 / 台账）、libero_planner（BDDL→子目标）、
              task_spec（BDDL 解析）、libero_skills（技能实现）、objectives / predicates
  envs/       libero_adapter（LIBERO/robosuite → 统一动作与语义目标语义）
  physics/    失败机制判别、Θ 割退路、参数推导、后验估计
  skills/     perception/（YOLO/SAM/GraspNet）、primitives/（ik_servo、motion、control）
  benchmarks/ LIBERO 动态工厂（suite, task_idx → env_id / goal / spec）
  memory/     经验库（LIBERO 路径当前不写）
  ipc/        常驻仿真进程 + 反思学习进程（scripts/ipc_learn.py 用它）
  policies/   步级决策点（换参 / 规则 / 参数空间）
  utils/      权重下载、MuJoCo 树工具、EGL 录像
scripts/      LIBERO 入口（verify_30.sh、ipc_learn.py、learn_libero_all.py）
              + dev_probes/（排障取证探针，归档不删）
docs/         本文档目录（07–12）
reports/      标定报告与台账快照
```

### 目录约定

- `logs/`、`videos/`、`data/videos/`、`*.mp4`、`**/snapshots/` **不入库**（见 `.gitignore`）——运行产物。
- `scripts/dev_probes/` 是**一次性排障探针**，归档不删：文档里引用的取证脚本仍可复现结论。

## License

MIT
