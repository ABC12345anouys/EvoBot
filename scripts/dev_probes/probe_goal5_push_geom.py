"""goal:5 push 起点可达性探针：打印推送起点/目标区/TCP 轨迹，定位 above 段 stall。

跑: MUJOCO_GL=egl python scripts/probe_goal5_push_geom.py
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

sys.path.insert(0, ".")

import numpy as np

from darwin.benchmarks import get_libero_benchmark
from darwin.agents.env_utils import get_env as _get_env
from darwin.agents.libero_tasks import resolve_site_name
from darwin.physics.derives import push_start_xy, push_z
from darwin.skills.primitives import HomeSkill, IkServoSkill

entry = get_libero_benchmark("libero_goal", 5)
env = _get_env(entry["env_id"], entry["robot"], entry.get("controller"))
env.reset()
body = entry["body"]
ACTOR, SITE = "agent0", "gripper0_grip_site"

b = env.object_bounds(body)
body_c = np.asarray(b["center"], float)
half_max = max(float(b["half_x"]), float(b["half_y"]))
z_mid = 0.5 * (float(b["z_top"]) + float(b["z_bottom"]))

csite = resolve_site_name(env.mj_model, "main_table_stove_front_region")
sid = int(env.mj_model.site_name2id(csite))
tgt_c = np.asarray(env.get_site_pos(csite), float)
xmat = np.asarray(env.mj_data.site_xmat[sid], float).reshape(3, 3)
half = np.abs(xmat @ np.asarray(env.mj_model.site_size[sid], float))

start_xy = push_start_xy(body_c[:2], tgt_c[:2], half_max)
z_push = push_z(z_mid)
print(f"[probe] body={np.round(body_c,3)} half_max={half_max:.3f} z_mid={z_mid:.4f}",
      flush=True)
print(f"[probe] region={csite} c={np.round(tgt_c,3)} half={np.round(half,3)}",
      flush=True)
print(f"[probe] start_xy={np.round(start_xy,3)} z_push={z_push:.4f}", flush=True)

home = HomeSkill()
ik = IkServoSkill()
for trial in range(2):
    env.reset()
    home.execute(env, actor=ACTOR, grip_site=SITE)
    pt = [float(start_xy[0]), float(start_xy[1]), z_push + 0.08]
    print(f"[probe] trial {trial}: above → {np.round(pt, 3)}", flush=True)
    r = ik.execute(env, point=pt, mode="above", hover=0.08, actor=ACTOR,
                   grip_site=SITE, k=5.0, timeout=120)
    print(f"[probe] above result: {r}", flush=True)
print("[probe] done", flush=True)
