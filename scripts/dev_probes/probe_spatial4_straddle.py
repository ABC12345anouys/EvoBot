"""诊断探针：libero_spatial:6 ik_servo(above) 撞 wooden_cabinet_1_base。

现象（logs/episodes/libero_spatial_06_20260925_143320_48416.jsonl）：
  抓取候选 = 碗心(0.143,-0.077) 沿 -y 偏置 ~0.199 → (0.143,-0.275)，
  正好贴 wooden_cabinet_1 立面；above 腿 tcp 停在 z≈1.133 磨停，
  coll_pair = gripper0_right_gripperseg / wooden_cabinet_1_base。

本探针：
  1. 复现 rules.py 容器分支的决策内部量（云、搜索网格、各方向净空），
     弄清 0.199 偏置怎么来的、为何 _clr 没拦住 -y 向；
  2. 打印碗/柜的 AABB 与点云 z 带；
  3. 驱动 ik_servo(above) 复现撞柜并 dump 接触。

用法:
    MUJOCO_GL=egl PYTHONPATH=$PWD:/home/lifd/Public/LIBERO \
      python scripts/probe_spatial6_straddle.py
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
from darwin.skills.primitives import servo_step
from darwin.physics.derives import straddle_offset, straddle_offset_search
from darwin.skills.perception.grasp import sample_object_point_cloud

import mujoco  # noqa: E402

entry = parse_task("libero_spatial:4")
registry = default_registry()
runner = DynamicEpisodeRunner(
    entry, rag=None, max_attempts=1, verbose=False,
    log_dir=tempfile.mkdtemp(prefix="probe_s6_"),
    registry=registry, experience=ExperienceStore(),
    blacklist_path=os.path.join(tempfile.gettempdir(), "probe_bl.json"),
    blacklist_write=False)
store = SkillConfigStore.load("ik_servo", config_env_of(entry),
                              task=task_id_of(entry))
handle = runner.prepare_episode(record_dir=None,
                                initial_cfg=dict(store.params))
env = handle.env
cfg = dict(store.params)
print("[probe] cfg 关键项:", {k: cfg.get(k) for k in
      ("hover", "grasp_container_offset_ratio", "grasp_container_z_delta",
       "k", "vcap")})

m = getattr(env.mj_model, "_model", env.mj_model)
d = getattr(env.mj_data, "_data", env.mj_data)

body_name = entry["body"]
b = env.object_bounds(body_name)
cx, cy = float(b["center"][0]), float(b["center"][1])
half_max = max(float(b["half_x"]), float(b["half_y"]))
z_top = float(b["z_top"])
ratio = float(cfg.get("grasp_container_offset_ratio", 0.70))
z_delta = float(cfg.get("grasp_container_z_delta", 0.030))
hover = float(cfg.get("hover", 0.08))
print(f"[probe] bowl bounds: center=({cx:.4f},{cy:.4f}) "
      f"half=({b['half_x']:.4f},{b['half_y']:.4f}) "
      f"z[{b['z_bottom']:.4f},{z_top:.4f}]")

# ---- 1. 全部障碍的 AABB ----
print("[probe] --- obstacle AABBs ---")
obs = list(getattr(env, "obstacle_bodies", None) or ())
for nm in obs:
    try:
        ob = env.object_bounds(nm)
        print(f"  {nm:36s} c=({ob['center'][0]:+.3f},{ob['center'][1]:+.3f}) "
              f"h=({ob['half_x']:.3f},{ob['half_y']:.3f}) "
              f"z[{ob['z_bottom']:.3f},{ob['z_top']:.3f}]")
    except Exception as e:
        print(f"  {nm:36s} bounds失败: {e}")

# ---- 2. 云采样复现（同 rules.py）----
p_z = z_top - z_delta
clouds = {}
own_cloud = np.zeros((0, 3), float)
try:
    own_cloud = sample_object_point_cloud(env, body_name, 2048)
    clouds[body_name] = own_cloud
    for nm in obs:
        if nm == body_name:
            continue
        try:
            clouds[nm] = sample_object_point_cloud(env, nm, 1024)
        except Exception as e:
            print(f"[probe] 云采样失败 {nm}: {e}")
    cloud = np.vstack([c for c in clouds.values() if len(c)])
    near = cloud[np.hypot(cloud[:, 0] - cx, cloud[:, 1] - cy) < 3.0 * half_max]
    cloud = near if len(near) else cloud
except Exception as e:
    cloud = np.zeros((0, 3))
    print(f"[probe] 云整体失败: {e}")

print("[probe] --- 云构成（裁剪后总 n=%d）---" % len(cloud))
band_all = cloud[(cloud[:, 2] > p_z) & (cloud[:, 2] < p_z + hover)] \
    if len(cloud) else cloud
print(f"[probe] z 带 [{p_z:.4f},{p_z + hover:.4f}] 内总点数: {len(band_all)}")
for nm, cl in clouds.items():
    bd = cl[(cl[:, 2] > p_z) & (cl[:, 2] < p_z + hover)]
    if len(bd):
        r = np.hypot(bd[:, 0] - cx, bd[:, 1] - cy)
        print(f"  {nm:36s} n={len(bd):5d} radial[min,max]=[{r.min():.3f},"
              f"{r.max():.3f}] x[{bd[:,0].min():.3f},{bd[:,0].max():.3f}] "
              f"y[{bd[:,1].min():.3f},{bd[:,1].max():.3f}]")

# ---- 3. 搜索决策复现 ----
gp = str(entry.get("grip_site", "gripper0_grip_site")).split("_grip_site")[0]


def finger_half_spread(direction):
    mw, dw = env.mj_model, env.mj_data   # robosuite 包装（同 rules.py）
    p1 = dw.body_xpos[mw.body_name2id(f"{gp}_leftfinger")]
    p2 = dw.body_xpos[mw.body_name2id(f"{gp}_rightfinger")]
    return abs(float((np.asarray(p1) - np.asarray(p2))
                     @ np.asarray([direction[0], direction[1], 0.0]))) / 2.0


def search_dbg(direction, hs):
    """真实 straddle_offset_search（own=目标自身云）+ 诊断打印。"""
    off = straddle_offset_search(
        cloud, [cx, cy], direction, p_z, p_z + hover,
        hs, runner.FINGER_R_M, own=own_cloud)
    # 复算分数曲线（与 derives 同款语义）用于诊断
    pts = np.asarray(cloud, float).reshape(-1, 3)
    band = pts[(pts[:, 2] > p_z) & (pts[:, 2] < p_z + hover)]
    c2 = np.asarray([cx, cy])
    u = np.asarray(direction, float)
    u /= max(np.linalg.norm(u), 1e-9)
    rel = band[:, :2] - c2
    along = rel @ u
    perp = rel - np.outer(along, u)
    perp_n = np.linalg.norm(perp, axis=1)
    ob = own_cloud[(own_cloud[:, 2] > p_z) & (own_cloud[:, 2] < p_z + hover)]
    oalong = (ob[:, :2] - c2) @ u if len(ob) else None
    best = -np.inf
    for off_t in np.linspace(0, hs + float(np.hypot(rel[:,0], rel[:,1]).max()), 64):
        if oalong is not None and not (
                ((oalong > off_t - hs) & (oalong < off_t + hs)).any()):
            continue
        d_in = float(np.sqrt(((off_t - hs) - along) ** 2 + perp_n ** 2).min())
        d_out = float(np.sqrt(((off_t + hs) - along) ** 2 + perp_n ** 2).min())
        best = max(best, min(d_in, d_out) - runner.FINGER_R_M)
    print(f"  dir=({direction[0]:+.1f},{direction[1]:+.1f}) hs={hs:.4f} "
          f"带内点={len(band)} → off={off}  best_score={best:.4f}")
    for q in (straddle_offset(float(b['half_y']), ratio),
              0.7 * float(b['half_y'])):
        print(f"    参考 off={q:.4f} (ratio 路径)")
    return off


def clr(p):
    z_low, z_high = float(p[2]), float(p[2] + 0.30)
    c = 1.0
    best = ("", 1.0)
    for nm in obs:
        if nm == body_name:
            continue
        try:
            ob = env.object_bounds(nm)
        except Exception:
            continue
        hx, hy = float(ob["half_x"]), float(ob["half_y"])
        if hx <= 0 or hy <= 0:
            continue
        if float(ob["z_top"]) < z_low or float(ob["z_bottom"]) > z_high:
            continue
        cc = ob["center"]
        dx = max(float(cc[0]) - hx - p[0], 0.0, p[0] - float(cc[0]) - hx)
        dy = max(float(cc[1]) - hy - p[1], 0.0, p[1] - float(cc[1]) - hy)
        dd = float(np.hypot(dx, dy))
        if dd < c:
            c, best = dd, (nm, dd)
    return c, best


print("[probe] --- 方向/偏置决策 ---")
y_dirs = [np.array([0.0, -1.0]), np.array([0.0, 1.0])]
decisions = []
for dd in y_dirs:
    hs = finger_half_spread(dd)
    off = search_dbg(dd, hs) if len(cloud) else None
    if off is None:
        off = straddle_offset(float(b["half_y"]), ratio)
    pos = [cx + off * dd[0], cy + off * dd[1], p_z]
    c, who = clr(pos)
    print(f"  → dir=({dd[0]:+.1f},{dd[1]:+.1f}) off={off:.4f} "
          f"pos=({pos[0]:.4f},{pos[1]:.4f}) clr={c:.4f} 最近={who[0]}")
    decisions.append((dd, off, c))

# 按 rules.py 同款规则选方向：净空大者胜（≥2cm 直接定，否则 12 向搜）
best_i = int(np.argmax([dc[2] for dc in decisions]))
if decisions[best_i][2] < 0.02:
    for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        dd = np.array([np.cos(ang), np.sin(ang)])
        hs = finger_half_spread(dd)
        off = search_dbg(dd, hs) if len(cloud) else None
        if off is None:
            off = straddle_offset(float(b["half_y"]), ratio)
        pos = [cx + off * dd[0], cy + off * dd[1], p_z]
        c, who = clr(pos)
        if c > decisions[best_i][2] + 1e-6:
            decisions[best_i] = (dd, off, c)
            print(f"  ↺ 12向搜索改选 ang={ang:.2f} off={off:.4f} clr={c:.4f}")
order = [best_i, 1 - best_i]
print(f"[probe] 选定方向 dir={decisions[best_i][0]} off={decisions[best_i][1]:.4f}")

# ---- 4. ik_servo(above) 复现 ----
site = "gripper0_grip_site"


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


for tag, di in zip(("CHOSEN", "OTHER"), order):
    dd, off, c = decisions[di]
    pos = np.array([cx + off * dd[0], cy + off * dd[1], p_z])
    tgt = np.array([pos[0], pos[1], pos[2] + hover])
    print(f"[probe] === servo test {tag}: target={np.round(tgt,4)} ===")
    for i in range(400):
        end = np.asarray(env.get_site_pos(site), float)
        if i % 40 == 0:
            print(f"  t={i:3d} tcp={np.round(end,4)}")
            contacts(f"t={i}")
        if np.linalg.norm(end - tgt) < 0.006:
            print(f"  [probe] {tag} reached @t={i} tcp={np.round(end,4)}")
            break
        servo_step(env, site, tgt, gripper=-1, k=float(cfg.get("k", 5.0)),
                   vcap=float(cfg.get("vcap", 1.0)), actor="agent0")
    else:
        end = np.asarray(env.get_site_pos(site), float)
        print(f"  [probe] {tag} TIMEOUT/stall tcp={np.round(end,4)} "
              f"dz={end[2]-tgt[2]:+.4f}")
        contacts("FINAL")
    # 复位臂：抬回起始高度再回中，避免两测试互相污染
    for i in range(200):
        end = np.asarray(env.get_site_pos(site), float)
        home = np.array([-0.217, -0.003, 1.16])
        if np.linalg.norm(end - home) < 0.01:
            break
        servo_step(env, site, home, gripper=-1, k=4.0, vcap=1.0,
                   actor="agent0")
