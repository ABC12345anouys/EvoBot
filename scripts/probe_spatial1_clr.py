"""诊断：spatial:1 容器 straddle 方向选择的净空计算复现。

打印 bowl bounds、各障碍 bounds、z 过滤结果、两个 y 方向的 _clr 值，
解释为什么最终选了朝向 ramekin 的 -y 方向。

用法: MUJOCO_GL=egl python scripts/probe_spatial1_clr.py
"""
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
    log_dir=tempfile.mkdtemp(prefix="probe_clr_"),
    registry=default_registry(), experience=ExperienceStore(),
    blacklist_path=os.path.join(tempfile.gettempdir(), "probe_bl.json"),
    blacklist_write=False)
store = SkillConfigStore.load("ik_servo", config_env_of(entry),
                              task=task_id_of(entry))
handle = runner.prepare_episode(record_dir=None,
                                initial_cfg=dict(store.params))
env = handle.env

body = "akita_black_bowl_1"
b = env.object_bounds(body)
print(f"[clr] bowl bounds: center={np.round(b['center'],4)} "
      f"half=({b['half_x']:.4f},{b['half_y']:.4f}) "
      f"z=[{b['z_bottom']:.4f},{b['z_top']:.4f}]")
cx, cy = float(b["center"][0]), float(b["center"][1])

obs_names = getattr(env, "obstacle_bodies", None)
print(f"[clr] obstacles: {obs_names}")
z_top = float(b["z_top"])
z_delta = 0.030
z_low = z_top - z_delta - 0.02
z_high = z_top + 0.30
print(f"[clr] z filter: [{z_low:.4f}, {z_high:.4f}]")


def clr(p):
    c = 1.0
    best = ("", 1.0)
    for nm in obs_names:
        if nm == body:
            continue
        try:
            ob = env.object_bounds(nm)
        except Exception as e:
            print(f"[clr]   {nm}: bounds fail {e}")
            continue
        hx, hy = float(ob["half_x"]), float(ob["half_y"])
        if hx <= 0.0 or hy <= 0.0:
            continue
        if float(ob["z_top"]) < z_low or float(ob["z_bottom"]) > z_high:
            print(f"[clr]   {nm}: z-filtered (z=[{ob['z_bottom']:.3f},"
                  f"{ob['z_top']:.3f}])")
            continue
        cc = ob["center"]
        print(f"[clr]   {nm}: center={np.round(cc,3)} half=({hx:.3f},"
              f"{hy:.3f}) z=[{ob['z_bottom']:.3f},{ob['z_top']:.3f}]")
        dx = max(float(cc[0]) - hx - p[0], 0.0, p[0] - float(cc[0]) - hx)
        dy = max(float(cc[1]) - hy - p[1], 0.0, p[1] - float(cc[1]) - hy)
        d = float(np.hypot(dx, dy))
        if d < c:
            c = d
            best = (nm, d)
    return c, best


# straddle_offset_search 实算（与 rules.py 同款调用）
from darwin.skills.perception.grasp import sample_object_point_cloud
from darwin.physics.derives import straddle_offset_search
from darwin.agents.runner_dynamic import DynamicEpisodeRunner as _DynR

cloud = sample_object_point_cloud(env, body, n_points=2048)
m, d = env.mj_model, env.mj_data
gp = "gripper0"
p1 = d.body_xpos[m.body_name2id(f"{gp}_leftfinger")]
p2 = d.body_xpos[m.body_name2id(f"{gp}_rightfinger")]
print(f"[clr] leftfinger={np.round(p1,4)} rightfinger={np.round(p2,4)}")

for tag, direction in (("-y", np.array([0.0, -1.0])),
                       ("+y", np.array([0.0, 1.0]))):
    hs = abs(float((np.asarray(p1) - np.asarray(p2)) @ np.asarray(
        [direction[0], direction[1], 0.0]))) / 2.0
    off = straddle_offset_search(
        cloud, [cx, cy], direction,
        z_top - z_delta, z_top - z_delta + 0.03, hs, _DynR.FINGER_R_M)
    p = [cx + (off or 0.0) * direction[0], cy + (off or 0.0) * direction[1]]
    c, best = clr(p)
    print(f"[clr] dir {tag}: half_spread={hs:.4f} off={off} "
          f"grasp_p={np.round(p,4)} clr={c:.4f} (nearest={best})")
    # 外指位置（偏置 + half_spread 继续向外）的净空
    p_out = [p[0] + hs * direction[0], p[1] + hs * direction[1]]
    c2, best2 = clr(p_out)
    print(f"[clr] dir {tag}: outer-finger p={np.round(p_out,4)} "
          f"clr={c2:.4f} (nearest={best2})")
