<p align="center">
  <img src="docs/figs/evobot.png" alt="EvoBot" width="620">
</p>

<div align="left">

# EvoBot

**Agent + Skill：Agent 决定做什么、按什么顺序做，技能库负责执行**

在 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 基准的 30 个任务上经过多轮迭代，成功率 **83%（25 / 30）**。

不用强化学习 · 不用视觉-语言-动作模型 · 不用世界模型 · 运行时不需要调用大模型

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Benchmark](https://img.shields.io/badge/benchmark-LIBERO-orange)](https://github.com/Lifelong-Robot-Learning/LIBERO)

[English](README.en.md) · [设计文档](docs/README.md)

![架构一览](docs/figs/architecture.png)

</div>

---

## 目录

- [这是什么](#这是什么)
- [核心思路](#核心思路)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [任务定义与生成](#任务定义与生成)
- [项目结构](#项目结构)
- [设计文档](#设计文档)
- [评测结果](#评测结果)
- [许可](#许可)

## 这是什么

EvoBot 是一个跑 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 机器人操作基准的框架。它的做法是把机器人操作拆成 **Agent 挑选技能、技能负责执行** 两层，技能是可以反复使用、带参数的动作单元：

- **Agent 层**负责看当前状态、决定下一个子目标用哪个技能、失败后怎么调整；
- **技能层**负责把一次动作做完，包括抓取、放置、推动、开合抽屉柜门与炉灶旋钮等；
- 技能失败时不只返回"失败"，还会说明**为什么失败**，Agent 据此换参数、换抓取位置或换做法。

任务的执行顺序不是让模型每次现场发挥：框架把 LIBERO 每个任务自带的官方任务定义解析成结构化数据，在**离线阶段**让模型读一遍并决定执行顺序，结果写成 YAML 提交进仓库；**运行时只读这份 YAML，不调用大模型**，因此同一任务的执行顺序每次都完全一样。

## 核心思路

**任务定义先解析成结构。** 任务定义文件里已经写清了最终要达到什么状态，框架把它解析成有序子目标，而不是让模型去猜目标。

**执行顺序离线决定一次。** 模型只做一件事：读结构化任务定义，排列子目标的执行顺序、判断是否需要插入"先开抽屉"这类前置动作。结果写成 YAML 存进仓库，之后不再调用模型。模型的能力有明确边界——只能排列已有条目的顺序、只能从给定候选里挑要插入的前置动作，不能新增或改写目标、不能指定技能、不能给参数；结果会经过校验，不通过就退回按依赖关系自动排序。

**子目标映射到技能，与任务名无关。** 技能失败时不只返回"失败"，还会给出失败原因（被挡住、够不到、打滑等），Agent 按原因类别决定是换参数、换抓取位置，还是换一种做法。

**一条轨迹的上限用仿真步数，不用实际耗时。** 这样结果不受机器负载影响。每个任务都从示范轨迹的初始状态开始，保证不同次运行可以对比。

## 环境要求

- **Python 3.10**（conda 环境；3.12 无法运行 robosuite 及其依赖）
- **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) 源码**，需要加到 `PYTHONPATH`
- MuJoCo + robosuite（LIBERO 自带）；无显示器的机器上需要设置 `MUJOCO_GL=egl`
- 依赖清单见 `requirements.txt` 与 `requirements-extra.txt`（录像、轨迹规划）

```bash
conda create -n darwin python=3.10 -y && conda activate darwin
pip install -e .                        # 安装本项目
pip install mujoco h5py numpy pyyaml    # 基础运行依赖
# 按 LIBERO 官方说明安装 LIBERO 及其依赖，然后：
export PYTHONPATH=$PWD:/path/to/LIBERO
export MUJOCO_GL=egl
```

## 快速开始

```bash
# 1) 跑单个任务（常驻仿真进程 + 逐步学习，默认最多尝试 30 次）
PYTHONPATH=$PWD:/path/to/LIBERO MUJOCO_GL=egl python -u scripts/ipc_learn.py \
    libero_spatial:0 --max-attempts 30 --sock /tmp/libero.sock

# 2) 跑 agent runner：会读取进度记录，自动跳过已经通过的任务
PYTHONPATH=$PWD:/path/to/LIBERO MUJOCO_GL=egl python -u -m darwin.agents.libero_runner \
    --suites libero_spatial,libero_object,libero_goal

# 3) 跑全部 30 个任务做验收
bash scripts/verify_30.sh 8 verify      # 参数：每个任务的尝试次数、结果标签
```

常用参数：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--suites` / `--tasks` | `libero_spatial,libero_object,libero_goal` | 要跑哪些任务集，或直接列出 `任务集:编号` |
| `--attempt-steps` | `7000` | 单条轨迹的仿真步数上限，主要的中断标准 |
| `--attempt-timeout` | `300` | 墙钟时间上限（秒），只在程序卡住不动时才可能先触发 |
| `--no-video` | — | 不录制视频 |

每个任务的尝试结果会记录在 `logs/libero_agent_progress.json`，键名形如 `libero:<任务集>:<编号>`；已经通过的任务下次会跳过。

## 任务定义与生成

每个任务的定义是一份自描述 YAML，存放在 `darwin/skills/configs/task_specs/`：

```yaml
language: put the bowl on top of the cabinet
source: llm                     # llm 表示由模型分解，deterministic 表示自动排序
formal:                         # 结构化的任务定义
  items:                        # 任务里要求达成的每一条，带固定编号
    - {id: p0, predicate: On, kind: place, object: akita_black_bowl_1,
       target: wooden_cabinet_1_top_side}
  implicit_candidates:          # 允许插入的前置动作
    - {id: 'open:wooden_cabinet_1_top_side', predicate: Open, target: ...}
  vocabulary: {objects: [...], targets: [...], fixtures: {...}}
  skills: [...]                 # 可用技能的说明
  reference_order: [...]        # 自动排序给出的参考顺序，用于对比
llm:                            # 这次分解的原始记录
  order: [p0]
  add: []
  rationale: 把碗放到柜子顶面，不需要打开柜门或抽屉。
subgoals: [...]                 # 最终结果，runner 直接使用
```

**重新生成**（需要 `DEEPSEEK_API_KEY`；没有设置时会自动退回确定性排序）：

```bash
python scripts/gen_task_specs.py                     # 全部 30 个任务
python scripts/gen_task_specs.py --tasks libero_goal:7
python scripts/gen_task_specs.py --no-llm            # 只用自动排序，不调用模型
python scripts/gen_task_specs.py --model deepseek-v4-pro --force
```

## 项目结构

```
darwin/
  agents/       libero_runner（尝试循环与进度记录）、libero_planner（子目标 → 技能）、
                task_spec（任务定义解析、结构化物化、模型分解与校验）、
                libero_skills（技能实现）、objectives、predicates
  envs/         libero_adapter（把 LIBERO/robosuite 包装成统一接口：
                动作约定、关节目标、任务步数上限的接管、按步数截断）
  physics/      失败原因判别、恢复建议、参数推导、状态估计
  skills/       perception/（YOLO / SAM / GraspNet）、primitives/（逆运动学与伺服、运动、控制）
                configs/task_specs/  ← 每个任务的定义（结构化任务定义 + 分解结果）
  benchmarks/   任务集工厂（任务集与编号 → 环境 id、目标、任务定义）
  ipc/          常驻仿真进程 + 学习进程（scripts/ipc_learn.py 使用）
  memory/       经验记录（LIBERO 路径下暂未写入）
  policies/     每步的参数决策（换参数 / 规则 / 参数范围）
  llm/          OpenAI 兼容客户端（供离线生成脚本使用）
  utils/        权重下载、MuJoCo 工具、录像
scripts/        gen_task_specs.py（离线生成任务定义）、verify_30.sh（全量验收）、
                ipc_learn.py（单任务）、learn_libero_all.py、calibrate_from_logs.py、
                bench_grasp_models.py、dev_probes/（排障用的探针脚本）
docs/           设计文档 01–06
```

约定：`logs/`、`videos/`、`*.mp4`、`**/snapshots/` 是运行产物，不提交到仓库（见 `.gitignore`）。

## 设计文档

| 编号 | 文档 | 内容 |
|---|---|---|
| 01 | [感知](docs/01_perception.md) | 物体位姿与抓取候选的来源：真值采样与模型推理两条路径 |
| 02 | [任务定义与规划](docs/02_task_and_planning.md) | 任务定义 → 有序子目标 → 技能序列，以及离线的一次分解 |
| 03 | [技能库](docs/03_skills.md) | 技能清单与统一约定 |
| 04 | [失败分类与恢复](docs/04_failure_and_recovery.md) | 失败原因分类与对应的处理方式 |
| 05 | [环境适配层](docs/05_env_adapter.md) | 统一动作接口、固定初始状态、任务步数上限的接管与截断 |
| 06 | [逆运动学、末端伺服与运动控制](docs/06_ik_servo_motion.md) | 末端目标位姿如何变成关节动作 |

## 评测结果

在 LIBERO 的 `libero_spatial`、`libero_object`、`libero_goal` 三个任务集共 30 个任务上完成全量评测，**25 / 30 通过**（每个任务单独运行，每次最多 8 次尝试）。

复现方式：

```bash
bash scripts/verify_30.sh 8 verify
```

## 许可

本项目使用 **Apache-2.0** 许可，详见 [LICENSE](LICENSE)。
