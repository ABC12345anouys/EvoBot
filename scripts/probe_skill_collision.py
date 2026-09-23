"""无渲染探针：复现视频 take 流程，打印每个 skill 的完整结果与最小间隙。

用法: python3 scripts/probe_skill_collision.py
"""
import sys
import tempfile

import numpy as np

sys.path.insert(0, ".")

from darwin.benchmarks import get_benchmark
from darwin.agents.runner import EpisodeRunner, _get_env
from darwin.memory import RAGMemory

entry = get_benchmark("drawer_place")
rag = RAGMemory(root=tempfile.mkdtemp(prefix="probe_rag_"))
runner = EpisodeRunner(entry, rag=rag, verbose=False)
env = _get_env(entry["env_id"], entry["robot"], entry.get("controller"))

env.reset()
import mujoco as _mj
SAFE_HOME_QPOS = np.array(
    [-0.614, -0.2586, -0.7121, 2.2855, 0.2737, -0.6727, -0.2971])
env.mj_data.qpos[:7] = SAFE_HOME_QPOS
env.mj_data.qvel[:7] = 0.0
_mj.mj_forward(env.mj_model, env.mj_data)
env.init_pos[entry["actor"]] = np.array(env.get_site_pos(entry["grip_site"]))
env.obstacle_bodies = ("cupboard", "drawer")
from darwin.skills.primitives.collision import CollisionMonitor
mon = CollisionMonitor(env, actor=entry["actor"])

# 与真实脚本对齐：用 hold 帧让 CARTIK 内部目标收敛到安全 home（否则前几步朝旧目标跳）
from darwin.skills.primitives import step_env, p_action
site0 = entry["grip_site"]
for _ in range(40):
    cur = env.get_site_pos(site0)
    step_env(env, entry["actor"], p_action(env, site0, cur, gripper=+1, k=2.0))
d, pr = mon.clearance()
print(f"[probe] settled clear={d*1000:.1f}mm {pr}")
print(f"[probe] monitor enabled={mon.enabled} arm_geoms={len(mon._arm)} "
      f"obs_geoms={len(mon._obs)}")

center = np.array(env.get_body_pos(entry["body"]), float)
cand = {"position": center.tolist()}
plan = runner._plan_for(cand, {}, None, env)
print(f"[probe] plan: {[p['action'] for p in plan]}")

skills = runner.agent.skills
boost = {"home": 150, "move_to": 220, "move_above": 200, "descend": 200,
         "move_to_xy_top": 200, "place": 380}
for i, p in enumerate(plan):
    ex = dict(p["params"])
    if p["action"] in boost:
        ex["timeout"] = boost[p["action"]]
    res = skills[p["action"]].execute(env=env, **ex)
    d, pr = mon.clearance()
    mc = res.get("min_clearance")
    mc = f"{mc*1000:.1f}mm" if isinstance(mc, float) else str(mc)
    print(f"[{i:02d}] {p['action']:14s} -> success={res.get('success')} "
          f"reason={res.get('reason','-')} min_clear={mc} "
          f"pair={res.get('coll_pair','-')} steps={res.get('steps','-')} "
          f"| now={d*1000:.1f}mm {pr}")
    if not res.get("success"):
        break
print("[probe] done")
