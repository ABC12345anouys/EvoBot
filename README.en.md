<div align="left">

# EvoBot

**BDDL formal spec + reusable skills for the LIBERO manipulation benchmark**

No RL · No VLA · No world model · No LLM calls at runtime

[![Python](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Benchmark](https://img.shields.io/badge/benchmark-LIBERO-orange)](https://github.com/Lifelong-Robot-Learning/LIBERO)

[中文](README.md) · [Design Docs](docs/README.md) · [Debug Notes](docs/12_libero_debug_2026-09-30.md)

</div>

---

## Contents

- [What is this](#what-is-this)
- [Core design](#core-design)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Task definitions: formal spec + LLM decomposition](#task-definitions-formal-spec--llm-decomposition)
- [Project layout](#project-layout)
- [Design docs](#design-docs)
- [Status & reproducibility](#status--reproducibility)
- [Known limitations](#known-limitations)
- [Citation & license](#citation--license)

## What is this

An execution stack for the [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) manipulation benchmark. It treats each task's BDDL definition as a **formal spec** instead of letting a model improvise:

1. **BDDL → ordered subgoals.** `On / In / Open / Close / Turnon / Turnoff` are parsed into structured subgoals. An LLM reads this spec **once, offline**, to perform a predicate-level decomposition (decide the execution order, decide whether prerequisites like "open the drawer first" are needed); the result is frozen as YAML committed to the repo, so **runtime costs zero tokens and is fully deterministic**.
2. **Subgoals → skills.** Predicate kinds map to parameterized skills (`place → grasp + place_at`, `articulate → articulate`, `toggle → toggle`), with zero coupling to task names.
3. **Skill failure → mechanism.** A failing skill returns a **failure mechanism** (`friction_slip`, `contact_blocked`, `ik_unreachable`, `reach_limit`, …) instead of a boolean; the runner reacts by retuning parameters, switching grasp candidates, or taking a geometric fallback.
4. **Deterministic truncation.** A trajectory is cut by **simulation steps** (`--attempt-steps`, default 7000); the wall-clock timeout is only a backstop for hard hangs — so machine load cannot change the verdict.

```
BDDL ──► task_spec ──► libero_planner ──► libero_skills ──► envs/libero_adapter ──► LIBERO/robosuite
             │              │                  │                    │
             │              │                  └── failure mechanism ──► physics discrimination & fallbacks
             └──────────────┴──► libero_runner: attempt loop / dual-criterion timeout / ledger / resume
```

## Core design

| Decision | Why |
|---|---|
| **Decompose once with an LLM, freeze into the repo** | The order is no longer a hardcoded heuristic (no more "the name contains `cabinet`, so it must need opening"), while runtime stays token-free and the plan stays reproducible. `validate_order()` is a pure-function check (ids complete & unique, `add` only from candidates); if it fails, we fall back to the deterministic topological sort. |
| **Keep the deterministic sort as reference order** | It doubles as fallback, as the validator's basis, and as the `reference_order` written into the YAML (for diffing and regression). |
| **Failures return a mechanism, not a boolean** | Different failures need different responses (blocked → softer and slower; unreachable → switch candidate / add time). A boolean throws that away. |
| **Step budget beats wall clock** | Results should not depend on how loaded the shared machine is. |
| **The spec is an artifact** | Each task's definition lands in `darwin/skills/configs/task_specs/<suite>_<idx>.yaml` — readable, diffable, replayable. |

## Requirements

- **Python 3.10** (conda env; tested on 3.10 — 3.12 cannot run robosuite / MinkowskiEngine)
- **[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) source tree** on `PYTHONPATH`
- MuJoCo + robosuite (as provided by LIBERO); `MUJOCO_GL=egl` on headless machines
- `requirements.txt` (this project), `requirements-extra.txt` (recording / planning)

```bash
conda create -n darwin python=3.10 -y && conda activate darwin
pip install -e .                      # install this project (the darwin package)
pip install mujoco h5py numpy pyyaml  # base runtime deps
# install LIBERO and its deps per its own instructions, then:
export PYTHONPATH=$PWD:/path/to/LIBERO
export MUJOCO_GL=egl
```

## Quick start

```bash
# 1) Single task: resident sim process + reflection learning (up to 30 attempts)
PYTHONPATH=$PWD:/path/to/LIBERO MUJOCO_GL=egl python -u scripts/ipc_learn.py \
    libero_spatial:0 --max-attempts 30 --sock /tmp/libero.sock

# 2) Run the agent runner directly: resumes from the ledger, skips passed tasks
PYTHONPATH=$PWD:/path/to/LIBERO MUJOCO_GL=egl python -u -m darwin.agents.libero_runner \
    --suites libero_spatial,libero_object,libero_goal

# 3) Full 30-task acceptance
bash scripts/verify_30.sh 8 verify      # args: attempts per task, result tag
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--suites` / `--tasks` | `libero_spatial,libero_object,libero_goal` | which suites to run, or an explicit `suite:idx` list |
| `--attempt-steps` | `7000` | per-trajectory simulation-step budget (**primary**, deterministic) |
| `--attempt-timeout` | `300` | wall-clock backstop in seconds (only fires for non-stepping hangs) |
| `--no-video` | — | disable recording |

The ledger lives at `logs/libero_agent_progress.json`, keyed by `libero:<suite>:<idx>`; passed tasks are skipped on the next cycle.

## Task definitions: formal spec + LLM decomposition

Each task's definition is a **self-describing YAML**, frozen under `darwin/skills/configs/task_specs/`:

```yaml
language: put the bowl on top of the cabinet
source: llm                     # llm | deterministic
formal:                         # the formal spec (read by both the LLM and the validator)
  items:                        # BDDL goal predicates, with stable ids
    - {id: p0, predicate: On, kind: place, object: akita_black_bowl_1,
       target: wooden_cabinet_1_top_side}
  implicit_candidates:          # prerequisites allowed to be inserted
    - {id: 'open:wooden_cabinet_1_top_side', predicate: Open, target: ...}
  vocabulary: {objects: [...], targets: [...], fixtures: {...}}
  skills: [...]                 # available skill vocabulary
  reference_order: [...]        # deterministic topological sort (for diff/regression)
llm:                            # the raw record of this decomposition
  order: [p0]
  add: []
  rationale: Placing the bowl on top of the cabinet requires no drawer/door to be opened.
subgoals: [...]                 # the product: consumed directly by the runner
```

**Regenerate** (needs `DEEPSEEK_API_KEY`; without it, falls back to the deterministic parse):

```bash
python scripts/gen_task_specs.py                     # all 30 tasks
python scripts/gen_task_specs.py --tasks libero_goal:7
python scripts/gen_task_specs.py --no-llm            # deterministic spec only
python scripts/gen_task_specs.py --model deepseek-v4-pro --force
```

## Project layout

```
darwin/
  agents/       libero_runner (attempt loop / ledger), libero_planner (subgoal→skill),
                task_spec (BDDL parsing / formal spec / LLM decomposition / validation),
                libero_skills (skill implementations), objectives, predicates
  envs/         libero_adapter (LIBERO/robosuite → unified actions and semantic goals;
                horizon rewind, AttemptStepLimit deterministic truncation)
  physics/      failure-mechanism discrimination, Θ-cut fallbacks, parameter derivation, posterior
  skills/       perception/ (YOLO/SAM/GraspNet), primitives/ (ik_servo, motion, control)
                configs/task_specs/  ← frozen task definitions (formal spec + LLM decomposition)
  benchmarks/   LIBERO dynamic factory (suite, idx → env_id / goal / spec)
  ipc/          resident sim process + reflection learner (used by scripts/ipc_learn.py)
  memory/       experience store (not written on the LIBERO path yet)
  policies/     step-level decision points (retry params / rules / parameter spaces)
  llm/          OpenAI-compatible client (used by the offline generator)
  utils/        weight download, MuJoCo tree helpers, EGL recording
scripts/        gen_task_specs.py (offline task-definition generator), verify_30.sh (acceptance),
                ipc_learn.py (single task), learn_libero_all.py, calibrate_from_logs.py,
                bench_grasp_models.py, dev_probes/ (diagnostic probes, archived not deleted)
docs/           design docs 07–12
reports/        calibration reports and ledger snapshots
```

Convention: `logs/`, `videos/`, `*.mp4`, `**/snapshots/` are run artifacts and are not tracked (see `.gitignore`).

## Design docs

| # | Doc | Highlights |
|---|-----|-----------|
| 07 | [Perception](docs/07_perception.md) | YOLO/SAM/GraspNet dual mode, ground-truth sampling path, random-source survey, GPU non-determinism |
| 08 | [Planning & Execution Stack](docs/08_libero_planning_stack.md) | BDDL→ordered subgoals, **two decomposition paths (LLM-frozen / deterministic reference)**, attempt loop with dual-criterion timeout |
| 09 | [Skills Library](docs/09_libero_skills.md) | `serve`/`goto` base, grasp candidate enumeration and 13 gates, `mem` dictionary, manifold following |
| 10 | [Physics Mechanisms](docs/10_physics_mechanisms.md) | 10 failure mechanisms and discriminators, Θ-cut and geometric fallbacks |
| 11 | [Env Adapter](docs/11_envs_libero_adapter.md) | fixed initial states, horizon rewind, `servo_step` action semantics, `AttemptStepLimit` |
| 12 | [Debug Notes 2026-09-30](docs/12_libero_debug_2026-09-30.md) | a full troubleshooting record: failure triage, the `turnon` endpoint bug, non-reproducibility investigation |

## Status & reproducibility

**23 of 30 tasks passing** (ledger `logs/libero_agent_progress.json`, snapshot 2026-09-30). Still failing: `spatial:4`, `object:1/6/7/9`, `goal:5/9`.

One representative fix: `goal:7` (`turn_on_the_stove`) — `articulation_info` used the **max** side of the joint range, demanding a target angle 3.6× LIBERO's success threshold; switching to the **min** side made the task pass on the first attempt. Evidence chain: [docs/12](docs/12_libero_debug_2026-09-30.md).

> ⚠️ **Results are not fully reproducible yet**: re-running the same task from the same fixed initial state can change the verdict. **Wall-clock timeout**, **CPU thread count**, and **RRT seed** have been ruled out; the residual noise comes from GraspNet's **GPU forward pass** (the unseeded global `np.random` is fixed; the GPU-forward residue remains). See [docs/07](docs/07_perception.md) and [docs/12](docs/12_libero_debug_2026-09-30.md) §4.
>
> **When reporting pass rates: a single pass ≠ a stable pass.**

## Known limitations

- **Reproducibility is only half solved** (see above).
- **The LLM decomposition has not been A/B-tested on the benchmark.** The formal spec does let the LLM correct a name-heuristic mistake (e.g. `wooden_cabinet_1_top_side` is the cabinet's **top surface**, so no implicit `Open` should be inserted), but how much that saves **has not been measured**.
- **Frozen artifacts are snapshots, not auto-synced**: if the BDDL changes, re-run `gen_task_specs.py`; there is no CI check that the YAMLs still match the current BDDL.
- **`unsupported` predicates (e.g. `NextTo`) are recorded but not executed.**

## Citation & license

License: **Apache-2.0** (see [LICENSE](LICENSE)).

```bibtex
@misc{evobot,
  title  = {EvoBot: BDDL formal spec + reusable skills for the LIBERO benchmark},
  author = {Li Feida},
  year   = {2026},
  note   = {https://github.com/ABC12345anouys/EvoBot}
}
```
