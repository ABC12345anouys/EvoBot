"""goal:5 push 策略冒烟：验证路由（plate 厚度 → libero_push 优先于
ik_libero_transfer）+ 单 attempt 端到端执行推盘链。

跑: MUJOCO_GL=egl python scripts/probe_goal5_push.py
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

sys.path.insert(0, ".")

from darwin.benchmarks import get_libero_benchmark
from darwin.agents.env_utils import get_env as _get_env
from darwin.agents.runner_dynamic import DynamicEpisodeRunner, objectives_from_entry
from darwin.agents.chain_registry import reset_default_registry

entry = get_libero_benchmark("libero_goal", 5)
env = _get_env(entry["env_id"], entry["robot"], entry.get("controller"))
env.reset()

# ① 方法路由验证
b = env.object_bounds(entry["body"])
th = float(b["z_top"]) - float(b["z_bottom"])
print(f"[probe] {entry['body']} thickness={th:.4f} "
      f"(push 阈值 0.020 → pushable={th < 0.02})", flush=True)

registry = reset_default_registry()
runner = DynamicEpisodeRunner(entry, rag=None, max_attempts=1, verbose=True)
conds = objectives_from_entry(entry, env)
print(f"[probe] objectives: {[c.describe() for c in conds]}", flush=True)
for c in conds:
    try:
        m = registry.select(c, entry)
        print(f"[probe] select: {c.describe()[:60]} → {m.name}", flush=True)
    except Exception as e:
        print(f"[probe] select failed for {c.describe()[:60]}: {e}", flush=True)

# ② 单 attempt 端到端
res = runner.run({"hover": 0.08, "k": 5.0, "vcap": 1.0, "timeout_scale": 1.0})
print(f"[probe] result: success={res.get('success')} "
      f"fail_phases={res.get('info', {}).get('fail_phases')}", flush=True)
print("[probe] done", flush=True)
