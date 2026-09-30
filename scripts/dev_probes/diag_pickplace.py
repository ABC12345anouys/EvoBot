"""诊断 pickplace：用 EpisodeRunner.run 跑一个 episode，打印 trajectory 每步 result。"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from darwin.benchmarks import get_benchmark
from darwin.skills import build_registry
from darwin.memory import RAGMemory
from darwin.agents.runner import EpisodeRunner

entry = get_benchmark("pickplace")
registry = build_registry()
rag = RAGMemory()
print(f"初始记忆: {rag.n_failure} fail / {rag.n_success} succ")

runner = EpisodeRunner(entry, rag=rag, registry=registry, max_attempts=6,
                       record_dir=None, seed="diag", verbose=True)

cfg = {"hover": 0.12, "k_descend": 2.0, "lift_height": 0.52, "jit": 0.01}
result = runner.run(cfg)

print(f"\n=== RESULT ===")
print(f"success={result['success']} attempts={result['metrics']['attempts']} steps={result['metrics']['steps']}")
traj = result.get("trajectory", [])
print(f"trajectory ({len(traj)} steps):")
for i, t in enumerate(traj):
    r = t.get("result", {})
    print(f"  [{i}] {t['action']}: success={r.get('success')} reason={r.get('reason')} steps={r.get('steps')}")
print(f"fail_phases: {result['info']['fail_phases']}")
