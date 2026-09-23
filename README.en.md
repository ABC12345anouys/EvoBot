<div align="center">

![EvoBot](figs/EvoBot.png)

# EvoBot

### Robot Manipulation with Agent + Skills. Nothing Else.

**No RL · No VLA · No WAM**

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[中文](README.md) · [Architecture Docs](docs/)

</div>

---

## What is EvoBot?

EvoBot is a **self-evolving framework for robot manipulation**. It does **not** use reinforcement learning, vision-language-action models, or world models. It solves tasks with an **agent that composes reusable skills** and remembers what works—so it needs **fewer LLM calls and fewer attempts** over time.

It has two layers:

**Upper layer — self-evolution loop:**

| Component | What it does |
|-----------|--------------|
| **Skills** | Parameterized primitives + forged strategies the agent composes to solve tasks |
| **RAG Memory** | Injects past successes/failures into each decision; skips known-bad offsets, reuses known-good ones |
| **Evolution** | MAP-Elites search over configs; successful trajectories are distilled into code skills |

**Lower layer — manipulation capabilities:**

| Capability | What it does |
|------------|--------------|
| **IK Control** | CARTIK joint-space control for stable long-range motion (replaces unstable CARTIMP) |
| **Force Control** | `guarded_move`, `impedance_push`, `spiral_search` — terminated by `force_ee` sensor |
| **Visual Servoing** | `servo_align` — closed-loop alignment on reprojection error for final cm-level precision |
| **MuJoCo** | Physics engine providing environment, contact dynamics, ground-truth perception |

## The Problem It Solves

LLM-driven robot manipulation has three recurring costs:

1. **Token waste** — the LLM re-plans the same pick-and-place from scratch every episode.
2. **Repeated failures** — no memory of "grasping from this offset slipped last time," so the robot retries the same bad action.
3. **No generalization** — a policy learned on one object or robot rarely transfers to another.

EvoBot addresses all three by treating **experience as a first-class, queryable resource**.

## How It Works: The Self-Evolution Loop

![EvoBot architecture](figs/Evobotarc.png)

**Before execution**, RAG filters out recently-failed grasp offsets and surfaces the best proven offset. **After execution**, the result is written back—successes become reusable strategies, failures become temporary pitfalls (TTL-based, not permanent).

### Skill hierarchy

```
Strategy Skills (forged, auto-generated from success)
        │  calls
Control / Motion Primitives (servo_align, guarded_move, path_plan, ...)
        │  calls
Perception Skills (detect, segment, grasp_pose)
```

## Quick Start

### Install

Core package (RAG memory only, zero ML dependencies):

```bash
pip install -e .
```

### Minimal usage

```python
from ragbot import RAGMemory

rag = RAGMemory()
feat = {"shape": "box", "size": [0.05, 0.05, 0.05]}

# Before action: learn from the past
best = rag.best_success("pick", feat)             # proven grasp offset
bad  = rag.is_failed("pick", feat, [0.2, 0, 0])  # did this fail recently?

# After action: write back experience
rag.add_success("pick", feat, rel_offset=[0.05, 0, 0], score_band="mid", attempts=2)
rag.add_failure("pick", feat, rel_offset=[0.2, 0, 0],  score_band="mid", fail_phase="descend")
rag.tick()    # advance failure-TTL clock
rag.decay()   # purge expired failures
```

### Run an evolution episode (simulation)

```bash
python scripts/run_evolution.py --task pickplace --episodes 20
```

## Architecture

```
ragbot/                  Public package — RAG memory (numpy + pyyaml only)
  memory/
    store.py             YAML-frontmatter store, evidence merge, confidence upgrade
    rag.py               normalized offset, TTL, semantic retrieval, UCB ranking
    experience/          seed experiences (.md, git-trackable)

darwin/                  Simulation & evolution harness (for validating RAG)
  agents/                ManipulationAgent (5-step loop), EpisodeRunner
  skills/                base.py, primitives/ (control, motion), perception/, forged/
  evolution/             MAP-Elites, skill_forge, EvolutionLoop
  robot/                 RobotBackend ABC + MujocoRobopalBackend + YAML profiles
  llm/                   OpenAI-compatible client (Ark / OpenAI), rule-based fallback
```

## Key Design Decisions

- **Normalized grasp offset** `rel_offset = (grasp − center) / half_size` — experiences transfer across objects of different sizes.
- **Failure TTL (20 episodes)** instead of permanent blacklisting — failures expire as the scene changes.
- **CARTIK over CARTIMP** — robopal's CARTIMP is unstable; use IK + joint impedance with a force sensor for termination.
- **Dispatcher-free execution** — the agent holds `Dict[str, Skill]` directly; forged skills reuse primitives via `bind()`.
- **Robot-agnostic motion primitives** — `motion.py` calls only `env.backend`; adding a robot = one YAML profile (URDF robots) or one `RobotBackend` subclass.

## Extending

| Goal | What to do |
|------|------------|
| Add a URDF-based robot | Write `darwin/assets/profiles/<name>.yaml` |
| Add a ROS/SDK robot | Subclass `RobotBackend`, register it, add a YAML profile |
| Add a control primitive | Subclass `Skill` in `darwin/skills/primitives/` |
| Use a custom perception stack | Provide `{shape, size, center, rel_offset}` — `ragbot` needs no ML deps |

## License

MIT
