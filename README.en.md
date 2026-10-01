<div align="center">

# EvoBot · LIBERO Stack

### Robot Manipulation with a BDDL Formal Spec + Reusable Skills.

**No RL · No VLA · No WAM · No LLM subgoal guessing**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[中文](README.md) · [Design Docs](docs/README.md) · [Debug Notes](docs/12_libero_debug_2026-09-30.md)

</div>

---

## What is this?

A deterministic execution stack for the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) manipulation benchmark. The core trade-off is that it **does not ask an LLM to guess subgoals**:

- Task definitions (BDDL) are treated as a **formal spec** and parsed into ordered subgoals directly;
- Each subgoal maps to a set of reusable **parameterized skills** (`serve` / `goto` / `grasp` / `place` / `articulate` …);
- When a skill fails it returns a **failure mechanism** (`friction_slip`, `contact_blocked`, `ik_unreachable`, …); the runner reacts to the mechanism — retune parameters, switch grasp candidate, or take a geometric fallback — instead of blind retrying;
- A trajectory is truncated by **simulation steps** (deterministic); the wall-clock timeout is only a backstop for hard hangs.

```
BDDL ──► task_spec ──► libero_planner ──► libero_skills ──► envs/libero_adapter ──► LIBERO/robosuite
             │              │                  │                    │
             │              │                  └── failure mechanism ──► physics discrimination & fallbacks
             └──────────────┴──► libero_runner: attempt loop / dual-criterion timeout / ledger / resume
```

## Design Docs

Six docs under `docs/`. They state only **facts readable from the code**, cite `file:line` for load-bearing claims, and each ends with an explicit "verified gaps and limitations" section. Index: **[docs/README.md](docs/README.md)**.

| # | Doc | Highlights |
|---|-----|-----------|
| 07 | [Perception](docs/07_perception.md) | YOLO/SAM/GraspNet dual mode, ground-truth sampling path, random-source survey, GPU non-determinism |
| 08 | [Planning & Execution Stack](docs/08_libero_planning_stack.md) | BDDL→ordered subgoals, predicate→skill map, disjunctive planning, attempt loop with dual-criterion timeout |
| 09 | [Skills Library](docs/09_libero_skills.md) | `serve`/`goto` base, grasp candidate enumeration and 13 gates, `mem` dictionary, manifold following |
| 10 | [Physics Mechanisms](docs/10_physics_mechanisms.md) | 10 failure mechanisms and discriminators, Θ-cut and geometric fallbacks |
| 11 | [Env Adapter](docs/11_envs_libero_adapter.md) | fixed initial states, horizon rewind, `servo_step` action semantics, `AttemptStepLimit` |
| 12 | [Debug Notes 2026-09-30](docs/12_libero_debug_2026-09-30.md) | a full troubleshooting record: failure triage, the `turnon` endpoint bug, non-reproducibility investigation |

## Quick Start

Requirements: a conda env (Python 3.10) plus the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) source tree; set `MUJOCO_GL=egl` on headless machines.

```bash
# Single task (up to 30 attempts): resident sim process + reflection learning
PYTHONPATH=$PWD:/path/to/LIBERO MUJOCO_GL=egl python -u scripts/ipc_learn.py \
    libero_spatial:0 --max-attempts 30 --sock /tmp/libero.sock

# Run the agent runner directly: resumes from the ledger, skips passed tasks
PYTHONPATH=$PWD:/path/to/LIBERO MUJOCO_GL=egl python -u -m darwin.agents.libero_runner \
    --suites libero_spatial,libero_object,libero_goal

# Full 30-task regression (acceptance script)
bash scripts/verify_30.sh 8 verify
```

Useful flags: `--attempt-steps 7000` (per-trajectory step budget, the primary criterion), `--attempt-timeout 300` (wall-clock backstop in seconds).

## Status

**23 of 30 tasks passing** (ledger `logs/libero_agent_progress.json`, snapshot 2026-09-30). Still failing: `spatial:4`, `object:1/6/7/9`, `goal:5/9`.

This round fixed `goal:7` (`turn_on_the_stove`): `articulation_info` used the max side of the joint range, demanding a target angle 3.6× LIBERO's success threshold; switching to the min side made the task pass on the first attempt. Evidence chain: [docs/12](docs/12_libero_debug_2026-09-30.md).

> ⚠️ **Results are not fully reproducible yet**: re-running the same task from the same fixed initial state can change the verdict. Wall-clock timeout, CPU thread count, and RRT seed have been ruled out; the residual noise comes from GraspNet's GPU forward pass in perception (the unseeded global `np.random` is fixed; the GPU-forward residue remains). See [docs/07](docs/07_perception.md) and [docs/12](docs/12_libero_debug_2026-09-30.md) §4.
>
> **When reporting pass rates: a single pass ≠ a stable pass.**

## Architecture

```
darwin/
  agents/     libero_runner (attempt loop / ledger), libero_planner (BDDL→subgoals),
              task_spec (BDDL parsing), libero_skills (skill implementations), objectives / predicates
  envs/       libero_adapter (LIBERO/robosuite → unified action and semantic-goal semantics)
  physics/    failure-mechanism discrimination, Θ-cut fallbacks, parameter derivation, posterior
  skills/     perception/ (YOLO/SAM/GraspNet), primitives/ (ik_servo, motion, control)
  benchmarks/ LIBERO dynamic factory (suite, task_idx → env_id / goal / spec)
  memory/     experience store (not written on the LIBERO path yet)
  ipc/        resident sim process + reflection learner (used by scripts/ipc_learn.py)
  policies/   step-level decision points (retry params / rules / parameter spaces)
  utils/      weight download, MuJoCo tree helpers, EGL recording
scripts/      LIBERO entry points (verify_30.sh, ipc_learn.py, learn_libero_all.py)
              + dev_probes/ (diagnostic probes, archived rather than deleted)
docs/         this documentation set (07–12)
reports/      calibration reports and ledger snapshots
```

### Directory Conventions

- `logs/`, `videos/`, `data/videos/`, `*.mp4`, `**/snapshots/` are **not tracked** (see `.gitignore`) — run artifacts.
- `scripts/dev_probes/` holds **one-off diagnostic probes**, archived rather than deleted: evidence scripts cited by the docs stay reproducible.

## License

MIT
