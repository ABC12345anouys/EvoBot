"""诊断探针：libero_spatial:4 descend 楔停接触分析。

复用 SimWorker 的 env 构造（DynamicEpisodeRunner.prepare_episode +
ik_servo.libero_spatial_4.yaml 参数），手动驱动：
  1. 张爪伺服到抓取点正上方 hover=0.03
  2. 竖直下降到日志同款目标（grasp_pt + straddle offset, z_goal=0.9060）
  3. 全程 dump mj_data.contact（dist<1cm 的接触对 + 法向力），
     定位楔停时手指到底压在哪。

用法:
    MUJOCO_GL=egl PYTHONPATH=$PWD:$PWD/../LIBERO \
      python scripts/probe_spatial1_descend.py
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
from darwin.skills.physics_profile import PhysicsProfile
from darwin.skills.skill_config import SkillConfigStore
from darwin.skills.primitives import servo_step

entry = parse_task("libero_spatial:4")
registry = default_registry()
runner = DynamicEpisodeRunner(
    entry, rag=None, max_attempts=1, verbose=False,
    log_dir=tempfile.mkdtemp(prefix="probe_s1_"),
    registry=registry, experience=ExperienceStore(),
    blacklist_path=os.path.join(tempfile.gettempdir(), "probe_bl.json"),
    blacklist_write=False)
store = SkillConfigStore.load("ik_servo", config_env_of(entry),
                              task=task_id_of(entry))
handle = runner.prepare_episode(record_dir=None,
                                initial_cfg=dict(store.params))
env = handle.env
print("[probe] env ready, cfg:", {k: store.params[k] for k in
      ("hover", "k", "k_descend", "vcap") if k in store.params})

# ---- 日志 142127 回放目标（LIBERO reset 确定性，逐值照搬）----
BOWL_LOG = np.array([0.081, -0.1432, 1.0626])
GRASP_LOG = np.array([0.07864812515051293, -0.09935404402306933,
                      1.0752051778662068])
OFF_LOG = np.array([0.09284262646582415, -0.10420338752602187,
                    1.0752051778662068]) - GRASP_LOG   # straddle offset
STOP_ABOVE = 0.0

bowl_now = np.asarray(env.get_body_pos("akita_black_bowl_1"), float)
grasp = GRASP_LOG + (bowl_now - BOWL_LOG)
pt = grasp + OFF_LOG
z_goal = pt[2] + STOP_ABOVE
print(f"[probe] bowl_now={np.round(bowl_now,4)} grasp={np.round(grasp,4)} "
      f"pt={np.round(pt,4)} z_goal={z_goal:.4f}")

site = "gripper0_grip_site"
m = getattr(env.mj_model, "_model", env.mj_model)
d = getattr(env.mj_data, "_data", env.mj_data)


def contacts(tag):
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        if c.dist < 0.01:
            g1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
            g2 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
            out.append(f"{g1}|{g2} d={c.dist:.4f}")
    if out:
        print(f"[probe] {tag} " + " ; ".join(out))
    return out


import mujoco  # noqa: E402  (须在 mujoco 环境引导后)

# ---- 1. 到正上方 hover ----
top = np.array([pt[0], pt[1], pt[2] + 0.03])
for i in range(300):
    end = np.asarray(env.get_site_pos(site), float)
    if np.linalg.norm(end - top) < 0.008:
        print(f"[probe] above ok @{i} end={np.round(end,4)}")
        break
    servo_step(env, site, top, gripper=-1, k=4.25, vcap=0.85, actor="agent0")
else:
    print("[probe] above TIMEOUT end=", np.round(end, 4))

# ---- 2. descend + 接触 dump ----
last_z = None
for t in range(300):
    end = np.asarray(env.get_site_pos(site), float)
    if t % 10 == 0 or (end[2] < 0.945 and t % 2 == 0):
        print(f"[probe] t={t:3d} tcp={np.round(end,4)}")
        contacts(f"t={t}")
    if abs(end[2] - z_goal) < 0.006:
        print(f"[probe] z reached t={t}")
        break
    servo_step(env, site, [pt[0], pt[1], z_goal], gripper=-1,
               k=6.7734, vcap=0.85, actor="agent0")

end = np.asarray(env.get_site_pos(site), float)
print(f"[probe] FINAL tcp={np.round(end,4)} z_goal={z_goal:.4f} "
      f"dz={end[2]-z_goal:.4f} xy_err={np.linalg.norm(end[:2]-pt[:2]):.4f}")
contacts("FINAL")
for fb in ("gripper0_leftfinger", "gripper0_rightfinger"):
    try:
        print(f"[probe] {fb} xpos=",
              np.round(np.asarray(env.get_body_pos(fb), float), 4))
    except Exception as e:
        print(f"[probe] {fb}: {e}")

# ---- 3. 楔停点手部几何：谁挡住了手掌？----
import mujoco  # noqa: E402
m = getattr(env.mj_model, "_model", env.mj_model)
d = getattr(env.mj_data, "_data", env.mj_data)
mujoco.mj_forward(m, d)

def geom_aabb(g):
    c = np.asarray(d.geom_xpos[g], float)
    R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
    sz = np.asarray(m.geom_size[g], float)
    gt = int(m.geom_type[g])
    if gt == mujoco.mjtGeom.mjGEOM_BOX:
        ext = sz
    elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
        ext = np.array([sz[0]] * 3)
    elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
        ext = np.array([sz[0], sz[0], sz[1]])
    elif gt == mujoco.mjtGeom.mjGEOM_MESH:
        # mesh AABB：取局部顶点包围盒（近似，仅诊断用）
        ext = np.array([0.02, 0.02, 0.02])
    else:
        ext = sz
    corners = c[:, None] + R @ np.diag(ext) @ np.array(
        [[sx, sy, sz2] for sx in (-1, 1) for sy in (-1, 1)
         for sz2 in (-1, 1)]).T
    return corners.min(axis=1), corners.max(axis=1)

hand_gs = []
for g in range(m.ngeom):
    nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    if nm.startswith("gripper0_") and "collision" in nm:
        lo, hi = geom_aabb(g)
        hand_gs.append((g, nm, lo, hi))
        print(f"[probe] geom {nm:42s} z[{lo[2]:.4f},{hi[2]:.4f}] "
              f"x[{lo[0]:.3f},{hi[0]:.3f}] y[{lo[1]:.3f},{hi[1]:.3f}]")

# 手掌（hand_collision）footprint 内的碗沿点云 → 挡住手掌的沿口 z
from darwin.skills.perception.grasp import sample_object_point_cloud
cloud = sample_object_point_cloud(env, "akita_black_bowl_1", 8192)
print(f"[probe] cloud {cloud.shape} z range "
      f"[{cloud[:,2].min():.4f},{cloud[:,2].max():.4f}]")
for g, nm, lo, hi in hand_gs:
    if "hand" not in nm:
        continue
    inside = (np.abs(cloud[:, 0] - 0.5 * (lo[0] + hi[0]))
              < 0.5 * (hi[0] - lo[0]) + 0.002) \
        & (np.abs(cloud[:, 1] - 0.5 * (lo[1] + hi[1]))
           < 0.5 * (hi[1] - lo[1]) + 0.002) \
        & (cloud[:, 2] > lo[2] - 0.01)
    if inside.any():
        zz = cloud[inside, 2]
        print(f"[probe] {nm}: footprint 内碗沿点 n={inside.sum()} "
              f"z_max={zz.max():.4f} (手掌底 lo.z={lo[2]:.4f})")

# ---- 4. 楔停高度直接试抓：闭合→抬升→看碗是否跟爪 ----
print("[probe] 楔停点试抓 close+lift...")
for i in range(40):
    servo_step(env, site, [end[0], end[1], end[2]], gripper=+1,
               k=4.0, vcap=0.5, actor="agent0")
bowl_z0 = float(np.asarray(env.get_body_pos("akita_black_bowl_1"), float)[2])
for i in range(120):
    e = np.asarray(env.get_site_pos(site), float)
    servo_step(env, site, [e[0], e[1], e[2] + 0.10], gripper=+1,
               k=4.0, vcap=0.4, actor="agent0")
bowl_z1 = float(np.asarray(env.get_body_pos("akita_black_bowl_1"), float)[2])
tcp_z1 = float(np.asarray(env.get_site_pos(site), float)[2])
print(f"[probe] 试抓: bowl_z {bowl_z0:.4f}->{bowl_z1:.4f} "
      f"(Δ={bowl_z1-bowl_z0:+.4f}) tcp_z1={tcp_z1:.4f}")
contacts("AFTER-LIFT")
