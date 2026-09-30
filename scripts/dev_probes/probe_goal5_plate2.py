"""goal:5 plate 抓取探针 v2：生产参数精确复现 + 手指位置/接触力逐帧打印。

v1 发现：z_goal=z_top-2mm 时全偏移（含中心）lift ✓；生产 z_goal=z_top-22mm
（grasp_z_flat 厚支路 + stop_above=-0.01）8/8 lift_no_grip。本探针打印
指尖 z 定位机制差异。

跑: MUJOCO_GL=egl python scripts/probe_goal5_plate2.py
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
from darwin.skills.primitives import HomeSkill, IkServoSkill, CloseGripperSkill, LiftSkill

entry = get_libero_benchmark("libero_goal", 5)
env = _get_env(entry["env_id"], entry["robot"], entry.get("controller"))
env.reset()
body = entry["body"]
ACTOR, SITE = "agent0", "gripper0_grip_site"

FINGER_CAND = ("left_finger", "right_finger", "finger1", "finger2",
               "gripper0_leftfinger", "gripper0_rightfinger",
               "left_finger_pad", "right_finger_pad")


def finger_zs():
    out = {}
    m, d = env.mj_model, env.mj_data
    for n in FINGER_CAND:
        try:
            bid = m.body_name2id(n)
            out[n] = float(d.xpos[bid][2])
        except Exception:
            pass
    # 退化：列全部带 finger 的 body
    if not out:
        for i in range(m.nbody):
            nm = m.body(i).name
            if "finger" in nm.lower():
                out[nm] = float(d.xpos[i][2])
    return out


b = env.object_bounds(body)
cx, cy = float(b["center"][0]), float(b["center"][1])
z_top = float(b["z_top"])
print(f"[probe] plate c=({cx:.3f},{cy:.3f}) z_top={z_top:.4f} "
      f"thickness={z_top - float(b['z_bottom']):.4f}", flush=True)

home, ik, close, lift = HomeSkill(), IkServoSkill(), CloseGripperSkill(), LiftSkill()

TRIALS = [
    ("生产复现 z=0.907 stop_above=-0.01 k=5", 0.907, -0.01, 5.0),
    ("v1 参数 z=0.917 stop_above=0 k=默认", 0.917, 0.0, 2.0),
    ("中间 z=0.912 stop_above=-0.005 k=5", 0.912, -0.005, 5.0),
]

for name, gz, sa, kk in TRIALS:
    env.reset()
    home.execute(env, actor=ACTOR, grip_site=SITE)
    pt = [cx, cy, gz]
    r1 = ik.execute(env, point=pt, mode="above", hover=0.08, actor=ACTOR,
                    grip_site=SITE, k=kk)
    r2 = ik.execute(env, point=pt, mode="descend", body=body, actor=ACTOR,
                    grip_site=SITE, timeout=90, k=kk, stop_above=sa,
                    contact_stop_band=0.05)
    endz = float(env.get_site_pos(SITE)[2])
    fz = finger_zs()
    r3 = close.execute(env, actor=ACTOR, grip_site=SITE)
    fz2 = finger_zs()
    bodyz0 = float(env.get_body_pos(body)[2])
    r4 = lift.execute(env, height=bodyz0 + 0.10, body=body, actor=ACTOR,
                      grip_site=SITE)
    fz_s = {k: round(v, 4) for k, v in fz.items()}
    fz2_s = {k: round(v, 4) for k, v in fz2.items()}
    print(f"[probe] {name}\n"
          f"   descend={'✓' if r2.get('success') else r2.get('reason')} "
          f"end_z={endz:.4f} 手指z={fz_s}\n"
          f"   close 后手指z={fz2_s} body_z0={bodyz0:.4f}\n"
          f"   lift={'✓' if r4.get('success') else r4.get('reason')} "
          f"liftF={r4.get('grip_contact_force_n')} "
          f"rise={r4.get('lift_rise_m')}", flush=True)
print("[probe] done", flush=True)
