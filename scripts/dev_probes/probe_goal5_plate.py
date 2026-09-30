"""goal:5 plate 抓取探针：实证 GraspNet 0 候选时不同 xy 偏移的下降-闭合-提升
结果，为"边缘包夹候选"（rules.py rim 候选）选定几何参数。

跑: MUJOCO_GL=egl python scripts/probe_goal5_plate.py
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
from darwin.skills.primitives import (
    HomeSkill, IkServoSkill, CloseGripperSkill, LiftSkill,
)

entry = get_libero_benchmark("libero_goal", 5)
print("[probe] entry:", {k: entry[k] for k in ("env_id", "body", "mode")}, flush=True)
env = _get_env(entry["env_id"], entry["robot"], entry.get("controller"))
env.reset()
body = entry["body"]
ACTOR = "agent0"
SITE = "gripper0_grip_site"

b = env.object_bounds(body)
cx, cy = float(b["center"][0]), float(b["center"][1])
hx, hy = float(b["half_x"]), float(b["half_y"])
z_top, z_bot = float(b["z_top"]), float(b["z_bottom"])
thickness = z_top - z_bot
print(f"[probe] plate bounds: c=({cx:.3f},{cy:.3f}) half=({hx:.3f},{hy:.3f}) "
      f"z=[{z_bot:.3f},{z_top:.3f}] thickness={thickness:.4f}", flush=True)

# 夹爪几何：开爪宽 / 指尖位置
try:
    gg = env.gripper_geometry(ACTOR)
    print("[probe] gripper_geometry:", gg, flush=True)
except Exception as e:
    print("[probe] gripper_geometry unavailable:", e, flush=True)

home = HomeSkill()
ik = IkServoSkill()
close = CloseGripperSkill()
lift = LiftSkill()

# 候选: 沿 +x 偏移 f*half_max, z = z_top - 2mm（grasp_z_flat 实证值）
factors = [0.0, 0.5, 0.75, 0.9, 1.0, 1.1]
for f in factors:
    env.reset()
    home.execute(env, actor=ACTOR, grip_site=SITE)
    gx = cx + f * hx
    gz = z_top - 0.002
    pt = [float(gx), float(cy), float(gz)]
    r1 = ik.execute(env, point=pt, mode="above", hover=0.08, actor=ACTOR, grip_site=SITE)
    r2 = ik.execute(env, point=pt, mode="descend", body=body, actor=ACTOR, grip_site=SITE,
                    timeout=90, contact_stop_band=0.05)
    end_z = None
    try:
        end_z = float(env.get_site_pos(SITE)[2])
    except Exception:
        pass
    r3 = close.execute(env, actor=ACTOR, grip_site=SITE)
    r4 = lift.execute(env, height=0.52, body=body, actor=ACTOR, grip_site=SITE)
    print(f"[probe] f={f:.2f} xy=({gx:.3f},{cy:.3f}) z_goal={gz:.3f} | "
          f"above={'✓' if r1.get('success') else r1.get('reason')} "
          f"descend={'✓' if r2.get('success') else r2.get('reason')}"
          f"(end_z={end_z and round(end_z,4)}) "
          f"close ✓ lift={'✓' if r4.get('success') else r4.get('reason')} "
          f"liftF={r4.get('max_force') or r4.get('force') or r4.get('f_max')}",
          flush=True)
print("[probe] done", flush=True)
