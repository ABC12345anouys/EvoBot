"""goal:5 plate 抓取探针 v3：测指尖-TCP 偏置，验证边缘包夹几何。

v2 结论修正：lift(0.52) 在 v1 是假成功（桌面物体 z≈0.9>0.52 即刻判真）；
全参数 lift_no_grip liftF=0——TCP 停在盘顶时指尖整体在盘底下方夹空。
本探针：①量指尖最低几何点相对 grip_site 的偏置 ②边缘候选（TCP 盘沿
正上方，指尖对齐盘厚中部）descend→close→lift。

跑: MUJOCO_GL=egl python scripts/probe_goal5_plate3.py
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

sys.path.insert(0, ".")

import numpy as np
import mujoco

from darwin.benchmarks import get_libero_benchmark
from darwin.agents.env_utils import get_env as _get_env
from darwin.skills.primitives import HomeSkill, IkServoSkill, CloseGripperSkill, LiftSkill

entry = get_libero_benchmark("libero_goal", 5)
env = _get_env(entry["env_id"], entry["robot"], entry.get("controller"))
env.reset()
body = entry["body"]
ACTOR, SITE = "agent0", "gripper0_grip_site"
m, d = env.mj_model, env.mj_data

b = env.object_bounds(body)
cx, cy = float(b["center"][0]), float(b["center"][1])
z_top, z_bot = float(b["z_top"]), float(b["z_bottom"])
z_mid = 0.5 * (z_top + z_bot)
print(f"[probe] plate c=({cx:.3f},{cy:.3f}) z=[{z_bot:.4f},{z_top:.4f}] "
      f"thickness={z_top - z_bot:.4f}", flush=True)

# ① 指尖偏置：reset 开爪位姿下，gripper0 手指 geom 最低点到 grip_site 的 z 差
site_z = float(env.get_site_pos(SITE)[2])
low = None
for i in range(m.ngeom):
    nm = m.geom(i).name or ""
    if "finger" not in nm.lower():
        continue
    gid = i
    gz = float(d.geom_xpos[gid][2])
    half = float(m.geom_size[gid][2])
    lowest = gz - half
    print(f"[probe] geom {nm}: z={gz:.4f} half_z={half:.4f} lowest={lowest:.4f}",
          flush=True)
    low = lowest if low is None else min(low, lowest)
if low is not None:
    finger_off = site_z - low
    print(f"[probe] site_z={site_z:.4f} 指尖最低={low:.4f} → 指尖在 TCP 下方 "
          f"{finger_off:.4f} m", flush=True)
else:
    finger_off = 0.056
    print("[probe] 未找到 finger geom，用默认偏置 0.056", flush=True)

home, ik, close, lift = HomeSkill(), IkServoSkill(), CloseGripperSkill(), LiftSkill()

# ② 边缘候选：f=TCP xy 在盘沿外 f*half_x；z 使指尖对齐盘厚中部
TRIALS = []
for f in (0.95, 1.05, 1.15):
    for zoff_name, zoff in (("mid", 0.0), ("upper", 0.004)):
        gz = z_mid + zoff + finger_off
        TRIALS.append((f"f={f} z={zoff_name}(TCP={gz:.3f})", cx + f * float(b["half_x"]), gz))

for name, gx, gz in TRIALS:
    env.reset()
    home.execute(env, actor=ACTOR, grip_site=SITE)
    pt = [float(gx), cy, float(gz)]
    r1 = ik.execute(env, point=pt, mode="above", hover=0.08, actor=ACTOR,
                    grip_site=SITE, k=5.0)
    r2 = ik.execute(env, point=pt, mode="descend", body=body, actor=ACTOR,
                    grip_site=SITE, timeout=90, k=5.0, stop_above=-0.01,
                    contact_stop_band=0.05)
    r3 = close.execute(env, actor=ACTOR, grip_site=SITE)
    bodyz0 = float(env.get_body_pos(body)[2])
    r4 = lift.execute(env, height=bodyz0 + 0.10, body=body, actor=ACTOR,
                      grip_site=SITE)
    print(f"[probe] {name}\n"
          f"   above={'✓' if r1.get('success') else r1.get('reason')} "
          f"descend={'✓' if r2.get('success') else r2.get('reason')} "
          f"end={np.round(r2.get('end'), 4) if r2.get('end') else None}\n"
          f"   lift={'✓' if r4.get('success') else r4.get('reason')} "
          f"liftF={r4.get('grip_contact_force_n')} rise={r4.get('lift_rise_m')}",
          flush=True)
print("[probe] done", flush=True)
