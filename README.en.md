<div align="left">

# EvoBot

**Agent + Skill: the Agent decides what to do and in what order, the skill library executes**

Across several rounds of iteration on the 30 tasks of the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) benchmark, **83% (25 / 30)** now succeed.

No RL · No VLA · No world model · No LLM calls at runtime

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Benchmark](https://img.shields.io/badge/benchmark-LIBERO-orange)](https://github.com/Lifelong-Robot-Learning/LIBERO)

[中文](README.md) · [Design Docs](docs/README.md)

![Architecture](docs/figs/architecture.png)

</div>

---

## Contents

- [What is this](#what-is-this)
- [Core ideas](#core-ideas)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Task definitions](#task-definitions)
- [Project layout](#project-layout)
- [Design docs](#design-docs)
- [Results](#results)
- [License](#license)

## What is this

EvoBot is a framework for the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) manipulation benchmark. Work is split into two layers: an **Agent** that picks which skill to use, and a **skill library** of reusable, parameterized action units.

- The **Agent** looks at the current state, decides which subgoal to tackle next, and adjusts when something fails.
- **Skills** carry out one action each — grasping, placing, pushing, opening drawers and cabinet doors, turning stove knobs.
- When a skill fails, it does not just report "failed": it reports **why**, and the Agent responds by changing parameters, trying another grasp position, or taking a different approach.

The execution order is not improvised at run time. Each LIBERO task ships with an official task definition; the framework parses it into a structured form, lets a model decide the execution order **once, offline**, and stores the result as YAML committed to the repository. **At run time it only reads that YAML — no model calls** — so the same task always follows the same plan.

## Core ideas

**Task definitions are parsed into structure first.** The task definition already states the final state to reach, so the framework turns it into ordered subgoals instead of asking a model to guess goals.

**The execution order is decided once, offline.** The model does exactly one thing: read the structured task definition, order the subgoals, and decide whether prerequisites like "open the drawer first" are needed. The result is written to YAML and committed; the model is not called again. Its freedom is bounded — it can only reorder the given items and pick prerequisites from a given candidate list; it cannot add or rewrite goals, choose skills, or set parameters. Results are validated, and if validation fails the framework falls back to ordering derived from the dependency structure.

**Subgoals map to skills, with no coupling to task names.** When a skill fails it does not just report "failed": it reports why (blocked, unreachable, slipped, …), and the Agent responds per category — change parameters, try another grasp position, or take a different approach.

**A trajectory's limit is measured in simulation steps, not wall-clock time,** so results do not depend on machine load. Every task starts from a demonstration's initial state, so different runs can be compared.

## Requirements

- **Python 3.10** (conda env; 3.12 cannot run robosuite and its dependencies)
- **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) source**, on `PYTHONPATH`
- MuJoCo + robosuite (as provided by LIBERO); set `MUJOCO_GL=egl` on headless machines
- Dependency lists: `requirements.txt` and `requirements-extra.txt` (video recording, trajectory planning)

```bash
conda create -n darwin python=3.10 -y && conda activate darwin
pip install -e .                        # install this project
pip install mujoco h5py numpy pyyaml    # base runtime dependencies
# install LIBERO and its dependencies per its own instructions, then:
export PYTHONPATH=$PWD:/path/to/LIBERO
export MUJOCO_GL=egl
```

## Quick start

```bash
# 1) Single task (resident sim process + step-by-step learning; up to 30 attempts)
PYTHONPATH=$PWD:/path/to/LIBERO MUJOCO_GL=egl python -u scripts/ipc_learn.py \
    libero_spatial:0 --max-attempts 30 --sock /tmp/libero.sock

# 2) Run the agent runner: reads the progress record and skips tasks that already passed
PYTHONPATH=$PWD:/path/to/LIBERO MUJOCO_GL=egl python -u -m darwin.agents.libero_runner \
    --suites libero_spatial,libero_object,libero_goal

# 3) Run all 30 tasks as acceptance
bash scripts/verify_30.sh 8 verify      # args: attempts per task, result tag
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--suites` / `--tasks` | `libero_spatial,libero_object,libero_goal` | which task sets to run, or an explicit `set:index` list |
| `--attempt-steps` | `7000` | per-trajectory simulation-step limit (the main stopping criterion) |
| `--attempt-timeout` | `300` | wall-clock limit in seconds (only fires if the process stops making progress) |
| `--no-video` | — | disable recording |

Per-task attempts are recorded in `logs/libero_agent_progress.json`, keyed as `libero:<set>:<index>`; tasks that already passed are skipped next time.

## Task definitions

Each task's definition is a self-describing YAML stored under `darwin/skills/configs/task_specs/`:

```yaml
language: put the bowl on top of the cabinet
source: llm                     # llm = decomposed by a model, deterministic = ordered automatically
formal:                         # the structured task definition
  items:                        # every condition the task requires, with a stable id
    - {id: p0, predicate: On, kind: place, object: akita_black_bowl_1,
       target: wooden_cabinet_1_top_side}
  implicit_candidates:          # prerequisites that may be inserted
    - {id: 'open:wooden_cabinet_1_top_side', predicate: Open, target: ...}
  vocabulary: {objects: [...], targets: [...], fixtures: {...}}
  skills: [...]                 # description of available skills
  reference_order: [...]        # order produced by automatic sorting, for comparison
llm:                            # raw record of this decomposition
  order: [p0]
  add: []
  rationale: Placing the bowl on top of the cabinet needs no door or drawer opened.
subgoals: [...]                 # the result, consumed directly by the runner
```

**Regenerate** (needs `DEEPSEEK_API_KEY`; without it, falls back to automatic ordering):

```bash
python scripts/gen_task_specs.py                     # all 30 tasks
python scripts/gen_task_specs.py --tasks libero_goal:7
python scripts/gen_task_specs.py --no-llm            # automatic ordering only
python scripts/gen_task_specs.py --model deepseek-v4-pro --force
```

## Project layout

```
darwin/
  agents/       libero_runner (attempt loop + progress record), libero_planner (subgoal → skill),
                task_spec (task-definition parsing, structuring, model decomposition + validation),
                libero_skills (skill implementations), objectives, predicates
  envs/         libero_adapter (wraps LIBERO/robosuite behind a uniform interface:
                action conventions, joint targets, taking over the task step limit, step-based truncation)
  physics/      failure-reason discrimination, recovery suggestions, parameter derivation, state estimation
  skills/       perception/ (YOLO / SAM / GraspNet), primitives/ (IK and servoing, motion, control)
                configs/task_specs/  ← per-task definitions (structured definition + decomposition result)
  benchmarks/   task-set factory (set + index → env id, goal, task definition)
  ipc/          resident sim process + learning process (used by scripts/ipc_learn.py)
  memory/       experience records (not written on the LIBERO path yet)
  policies/     per-step parameter decisions (retry params / rules / parameter ranges)
  llm/          OpenAI-compatible client (used by the offline generator)
  utils/        weight download, MuJoCo helpers, video recording
scripts/        gen_task_specs.py (offline task-definition generation), verify_30.sh (full acceptance),
                ipc_learn.py (single task), learn_libero_all.py, calibrate_from_logs.py,
                bench_grasp_models.py, dev_probes/ (diagnostic probes)
docs/           design docs 01–06
```

Convention: `logs/`, `videos/`, `*.mp4`, `**/snapshots/` are run artifacts and are not committed (see `.gitignore`).

## Design docs

| # | Doc | Contents |
|---|-----|----------|
| 01 | [Perception](docs/01_perception.md) | Where object poses and grasp candidates come from: ground-truth sampling vs. model inference |
| 02 | [Task Definitions & Planning](docs/02_task_and_planning.md) | Task definition → ordered subgoals → skill calls, and the one-off offline decomposition |
| 03 | [Skill Library](docs/03_skills.md) | The skill list and the common conventions |
| 04 | [Failures & Recovery](docs/04_failure_and_recovery.md) | Failure categories and how each is handled |
| 05 | [Environment Adapter](docs/05_env_adapter.md) | Uniform action interface, fixed initial states, keeping the environment from ending a task early, step-based truncation |
| 06 | [IK, Servoing & Motion Control](docs/06_ik_servo_motion.md) | How an end-effector target pose becomes joint actions |

## Results

Full evaluation over the 30 tasks of LIBERO's `libero_spatial`, `libero_object` and `libero_goal` sets: **25 / 30 pass** (each task run separately, up to 8 attempts each).

Reproduce with:

```bash
bash scripts/verify_30.sh 8 verify
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
