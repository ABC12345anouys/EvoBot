"""诊断v3：逐步复现 _clr / _off_for / 方向选择，带打印。"""
import os
import sys
import tempfile

os.environ.setdefault("MUJOCO_GL", "egl")
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

sys.path.insert(0, ".")
sys.path.insert(0, "/home/lifd/Public/LIBERO")

import numpy as np

from darwin.ipc.sim_worker import parse_task, task_id_of, config_env_of
from darwin.agents.runner_dynamic import DynamicEpisodeRunner
from darwin.agents.chain_registry import default_registry
from darwin.agents.experience_store import ExperienceStore
from darwin.skills.skill_config import SkillConfigStore

entry = parse_task("libero_spatial:1")
runner = DynamicEpisodeRunner(
    entry, rag=None, max_attempts=1, verbose=False,
    log_dir=tempfile.mkdtemp(prefix="probe_c3_"),
    registry=default_registry(), experience=ExperienceStore(),
    blacklist_path=os.path.join(tempfile.gettempdir(), "probe_bl.json"),
    blacklist_write=False)
store = SkillConfigStore.load("ik_servo", config_env_of(entry),
                              task=task_id_of(entry))
handle = runner.prepare_episode(record_dir=None,
                                initial_cfg=dict(store.params))
env = handle.env
cfg = dict(store.params)

body = entry["body"]
b = env.object_bounds(body)
cx, cy = float(b["center"][0]), float(b["center"][1])
z_top = float(b["z_top"])
print(f"[c3] bowl center=({cx:.6f},{cy:.6f}) z_top={z_top:.4f}")
print(f"[c3] cfg ratio={cfg.get('grasp_container_offset_ratio')} "
      f"z_delta={cfg.get('grasp_container_z_delta')} "
      f"hover={cfg.get('hover')}")
print(f"[c3] env.obstacle_bodies={getattr(env,'obstacle_bodies',None) is not None}"
      f" n={len(getattr(env,'obstacle_bodies',None) or ())}")

from darwin.skills.perception.grasp import sample_object_point_cloud
from darwin.physics.derives import straddle_offset_search
from darwin.agents.runner_dynamic import DynamicEpisodeRunner as _DynR

cloud = sample_object_point_cloud(env, body, n_points=2048)
_gp = str(entry.get("grip_site", "gripper0_grip_site")).split("_grip_site")[0]
m, d = env.mj_model, env.mj_data
p1 = d.body_xpos[m.body_name2id(f"{_gp}_leftfinger")]
p2 = d.body_xpos[m.body_name2id(f"{_gp}_rightfinger")]
print(f"[c3] gripper prefix={_gp} leftfinger={np.round(p1,4)} "
      f"rightfinger={np.round(p2,4)}")

z_delta = float(cfg.get("grasp_container_z_delta", 0.030))
hover = float(cfg.get("hover", 0.08))


def off_for(direction):
    hs = abs(float((np.asarray(p1) - np.asarray(p2)) @ np.asarray(
        [direction[0], direction[1], 0.0]))) / 2.0
    off = straddle_offset_search(
        cloud, [cx, cy], direction,
        z_top - z_delta, z_top - z_delta + hover, hs, _DynR.FINGER_R_M)
    print(f"[c3]   off_for dir={direction} hs={hs:.4f} -> off={off}")
    return off


def clr(p):
    obs_names = getattr(env, "obstacle_bodies", None)
    if not (obs_names and hasattr(env, "object_bounds")):
        print(f"[c3]   clr: NO OBSTACLES -> 1.0")
        return 1.0
    z_low = float(z_top - z_delta - 0.02)
    z_high = float(z_top + 0.30)
    c = 1.0
    best = ("", 1.0)
    for nm in obs_names:
        if nm == body:
            continue
        try:
            ob = env.object_bounds(nm)
        except Exception:
            continue
        hx, hy = float(ob["half_x"]), float(ob["half_y"])
        if hx <= 0.0 or hy <= 0.0:
            continue
        if float(ob["z_top"]) < z_low or float(ob["z_bottom"]) > z_high:
            continue
        cc = ob["center"]
        dx = max(float(cc[0]) - hx - p[0], 0.0, p[0] - float(cc[0]) - hx)
        dy = max(float(cc[1]) - hy - p[1], 0.0, p[1] - float(cc[1]) - hy)
        dist = float(np.hypot(dx, dy))
        if dist < c:
            c = dist
            best = (nm, dist)
    print(f"[c3]   clr(p={np.round(p[:2],4)}) = {c:.4f} nearest={best}")
    return c


y_dirs = [np.array([0.0, -1.0]), np.array([0.0, 1.0])]
y_clr = []
for dd in y_dirs:
    off = off_for(dd)
    p = [cx + (off or 0.0) * dd[0], cy + (off or 0.0) * dd[1],
         z_top - z_delta]
    y_clr.append(clr(p))
print(f"[c3] y_clr={y_clr} argmax={int(np.argmax(y_clr))}")
direction = y_dirs[int(np.argmax(y_clr))] if max(y_clr) >= 0.02 else None
print(f"[c3] chosen direction={direction}")
