"""无渲染探针：验证 runner_dynamic 的条件驱动闭环 + EpisodeLogger。

跑 drawer_place（与视频相同的安全 home / 障碍配置），打印动态规划事件，
最后检查 JSONL 日志的事件构成、采样密度与体积。

用法: python3 scripts/probe_dynamic_runner.py
"""
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

from darwin.benchmarks import get_benchmark
from darwin.agents.runner_dynamic import (
    DynamicEpisodeRunner, objectives_from_entry, select_method,
)
from darwin.memory import RAGMemory

entry = dict(get_benchmark("drawer_place"))
# 与 JARVIS 视频脚本一致的确定性配置
entry["safe_home_qpos"] = [-0.614, -0.2586, -0.7121, 2.2855,
                           0.2737, -0.6727, -0.2971]
entry["obstacles"] = ["cupboard", "drawer"]

log_dir = tempfile.mkdtemp(prefix="probe_dyn_logs_")
rag = RAGMemory(root=tempfile.mkdtemp(prefix="probe_rag_"))
runner = DynamicEpisodeRunner(entry, rag=rag, max_attempts=2,
                              verbose=True, log_dir=log_dir, sample_every=10)

# 先看条件/方法解析（不连环境执行）
from darwin.agents.runner_dynamic import _get_env
env = _get_env(entry["env_id"], entry["robot"], entry.get("controller"))
env.reset()
conds = objectives_from_entry(entry, env)
print("[probe] objectives:")
for c in conds:
    print(f"  - {c.describe():38s} via {select_method(c, entry).name}")

cfg = {"hover": 0.12, "k_descend": 2.0, "lift_height": 0.52, "jit": 0.0}
res = runner.run(cfg)

print("\n[probe] ===== result =====")
print(f"success={res['success']} metrics={res['metrics']}")
print(f"fail_phases={res['info']['fail_phases']}")
log_path = Path(res["info"]["log"])
print(f"log: {log_path} ({log_path.stat().st_size / 1024:.1f} KB)")

# 日志构成分析
rows = [json.loads(l) for l in log_path.read_text().splitlines()]
ev_count = Counter(r["ev"] for r in rows)
print(f"events: {dict(ev_count)} total={len(rows)}")
sim_samples = ev_count.get("sample", 0)
print(f"sim samples: {sim_samples} (every 10 sim steps -> "
      f"约 {sim_samples * 10} 控制步)")

print("\n[probe] 规划/条件事件序列：")
for r in rows:
    if r["ev"] == "plan":
        chain = " -> ".join(s["skill"] for s in r["steps"])
        print(f"  plan  {r['goal']:32s} via {r['method']:13s}: {chain}")
    elif r["ev"] == "condition":
        print(f"  cond  {r['name']:38s} satisfied={r['satisfied']} {r['measures']}")
    elif r["ev"] == "verify":
        print(f"  verify ok={r['ok']} {r['measures']}")
    elif r["ev"] == "episode_end":
        print(f"  episode_end success={r['success']} sim_steps={r['sim_steps']} "
              f"attempts={r['attempts']}")

# 抽查一条 sample 的字段完整性
sample = next(r for r in rows if r["ev"] == "sample")
print("\n[probe] sample 字段示例：")
print(json.dumps({k: sample[k] for k in
                  ("sim_step", "tcp", "arm_qpos", "bodies", "joints",
                   "clear_mm", "coll_pair") if k in sample},
                 ensure_ascii=False, indent=2))
print("[probe] done")
