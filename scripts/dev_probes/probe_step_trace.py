"""慢速(vcap)追踪 move_to 逐步 TCP 位移与 clearance。"""
import sys, tempfile
import numpy as np
sys.path.insert(0, ".")

from darwin.benchmarks import get_benchmark
from darwin.agents.runner import EpisodeRunner, _get_env
from darwin.memory import RAGMemory
from darwin.skills.primitives import step_env, p_action
from darwin.skills.primitives.collision import CollisionMonitor

entry = get_benchmark("drawer_place")
rag = RAGMemory(root=tempfile.mkdtemp(prefix="probe_rag_"))
runner = EpisodeRunner(entry, rag=rag, verbose=False)
env = _get_env(entry["env_id"], entry["robot"], entry.get("controller"))

env.reset()
import mujoco as _mj
SAFE_HOME_QPOS = np.array([-0.614, -0.2586, -0.7121, 2.2855, 0.2737, -0.6727, -0.2971])
env.mj_data.qpos[:7] = SAFE_HOME_QPOS
env.mj_data.qvel[:7] = 0.0
_mj.mj_forward(env.mj_model, env.mj_data)
env.init_pos[entry["actor"]] = np.array(env.get_site_pos(entry["grip_site"]))
env.obstacle_bodies = ("cupboard", "drawer")

mon = CollisionMonitor(env, actor=entry["actor"])
site = entry["grip_site"]
d0, p0 = mon.clearance()
print(f"[trace] start clear={d0*1000:.1f}mm {p0}")

tgt = np.array([0.44, 0.0, 0.62])
for t in range(120):
    prev = np.asarray(env.get_site_pos(site), float)
    step_env(env, entry["actor"], p_action(env, site, tgt, gripper=0.0, k=2.5, vcap=0.05))
    end = np.asarray(env.get_site_pos(site), float)
    d, pr = mon.clearance()
    flag = "  <<<< ABORT" if d < 0.004 else ""
    print(f"  t={t:03d} move={np.linalg.norm(end-prev)*1000:5.1f}mm "
          f"clear={d*1000:7.1f}mm {pr:26s} tcp=[{end[0]:.3f},{end[1]:.3f},{end[2]:.3f}]{flag}")
    if d < 0.004 or np.linalg.norm(end - tgt) < 0.012:
        break
