"""Drawer-v1 / DrawerBox-v1 / LockedCabinet-v1 纯控制器验证（诚实判据：官方 flag）。

运行时证据确立的事实链：
1. RethinkGripper env 归一化闭合位 -0.01 → 指尖条间隙 2.65cm，夹不住 2cm 把手 → 必打滑。
   补丁：apply_action 闭合指令改发 -0.020833（机械全闭合）→ 指尖条间隙 1.45cm，对 2cm 把手
   形成 5.5mm 过盈楔紧（kp=1000 → ~11N/指，μ≈1 → 摩擦 ~20N > 抽屉 frictionloss 10N）。
2. 指尖条（实测）z 跨度 [eef-0.0315, eef+0.0055]、指杆 [eef-0.0305, eef+0.0445]。
   抽屉把手=沿世界 x 长条 8cm×2cm(y)×1cm(z)，z∈[0.463,0.473]，中心 x=drawer_site+0.02
   → 抓取点 z=0.482（指尖条中心 0.469 恰对把手中心，容 ~1.4cm 接触下沉）。
   梁把手 z∈[0.55,0.58]（3cm 高）；门把手 C 竖杆 2×2×6cm（x[0.54,0.56] y[0.04,0.06] z[0.54,0.60]）。
3. P 速度控制的 desired 积分在 eef 被反顶时会失控漂移 → 一切推进带 lag 门控：
   lag 用 reset 时一次性捕获的基座→世界平移 off 计算（旧版每步重算 base_offset → lag≡0 → 门控全失效）；
   lag>5cm 冻结 desired（vel=0）让其追平后再推进。
4. close 阶段：全冻结 desired（vel=0），楔紧靠阻抗自稳。z 伺服（vel_z=clip(k·err)）是失控
   积分器——探针实测 desired 80 步漂 25cm 把手臂拽走，已弃用。
   抓取高度 0.462=指杆楔紧区（指杆间隙 1.8cm 对 2cm 把手 2mm 过盈，探针实测双指 10+ 接触、
   80 步 naive 拉出 drawer_q=0.098），容差 ±4cm 覆盖 descend 过冲。
5. 拉取一律用相对蠕进（target=当前 eef+4mm·dir，desired ~0.8mm/步匀速前移，lag 上限 0.12m
   冻结保险）：旧门控版 lag=0.05 即冻结 → 力恒低于摩擦loss 突破阈值 → 死锁
   （220 步 drawer 不动且 stalled=False 实证）。门拉方向每步沿 left_handle→opened 弦线重算。
运行：MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/check_cabinet_family.py   （REC=1 同时录像）
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

import numpy as np
import robopal
from robopal.envs.base import MujocoEnv
from robopal.robots.grippers import RethinkGripper

MujocoEnv.close = lambda self: self.renderer.close()

GRIP_CLOSE = -0.020833  # 机械全闭合（env 归一化 -0.01 间隙 2.65cm 夹不住 2cm 把手）
_orig_apply = RethinkGripper.apply_action
RethinkGripper.apply_action = lambda self, a: _orig_apply(self, GRIP_CLOSE if a < 0 else a)

GripSite = "0_grip_site"
REC = os.environ.get("REC") == "1"
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "videos"))


def p_action(env, target, gripper=0.0, k=4.0):
    end = env.get_site_pos(GripSite)
    vel = np.clip(k * (np.asarray(target, dtype=float) - end), -1.0, 1.0)
    return np.append(vel, 1.0 if gripper > 0 else (-1.0 if gripper < 0 else 0.0))


def snap(rec, env):
    if rec is not None:
        rec.snap(env)


def make_recorder(env, path):
    import cv2
    import mujoco
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.55, 0.0, 0.50]
    cam.distance, cam.azimuth, cam.elevation = 1.6, 135.0, -25.0
    renderer = mujoco.Renderer(env.mj_model, height=480, width=640)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 20, (640, 480))

    class _R:
        def snap(self, e):
            if writer.isOpened():
                renderer.update_scene(e.mj_data, camera=cam)
                writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

        def close(self, keep=True):
            writer.release()
            renderer.close()
            if not keep and os.path.exists(path):
                os.remove(path)

    return _R()


def lag_of(env, off):
    """跟踪滞后：eef(世界) 偏离 desired(基座)+off 的距离。off=reset 时捕获的基座→世界平移。"""
    return float(np.linalg.norm((env.get_site_pos(GripSite) - env.desired_position) - off))


def finger_qpos(env):
    return env.mj_data.joint("0_l_finger_joint").qpos[0]


def grasp_from_above(env, point, off, z_above=0.10, max_steps=400, descend_k=1.5, close_steps=30,
                     rec=None, verbose=False):
    """上方对准 point（z=夹持高度）→缓降→全闭合楔紧 + z 伺服回高度。返回 (是否完成, 步数)。"""
    point = np.asarray(point, dtype=float)
    phase, t, t2 = "above", 0, 0
    while t < max_steps:
        t += 1
        end = env.get_site_pos(GripSite)
        lag = lag_of(env, off)
        if phase == "above":
            act = p_action(env, point + [0, 0, z_above], gripper=+1, k=2.0)
            if np.linalg.norm((end - point)[[0, 1]]) < 0.005 and end[2] > point[2] + z_above - 0.02 and lag < 0.02:
                phase, t2 = "descend", t
        elif phase == "descend":
            # k=1.5 慢降 + lag<0.015 严门控（probe4 实证）：eef 停在把手上缘 +6mm 处，
            # 指尖条跨骑把手下缘成"钩"——快降(k=2)会让 desired 过冲、eef 追落到把手下 2cm 失楔
            act = p_action(env, point, gripper=+1, k=1.5)
            if end[2] < point[2] + 0.006 and lag < 0.015:
                phase, t2 = "close", t
            elif t - t2 > 150:  # 兜底：锥面卡阻推不动也强制闭合
                phase, t2 = "close", t
        elif phase == "close":
            # 全冻结 desired（vel=0）：楔紧靠阻抗自稳。z 伺服已被探针证伪——
            # vel_z=clip(k·err) 配 desired 积分是失控积分器（desired 漂 25cm 把手臂拽走）
            act = np.array([0.0, 0.0, 0.0, -1.0])
            if t - t2 > close_steps:
                if verbose:
                    print("    夹持 finger_qpos=%.4f eef_z=%.3f（目标 %.3f）" % (
                        finger_qpos(env), end[2], point[2]), flush=True)
                return True, t
        env.step(act)
        snap(rec, env)
    return False, t


def ramp_pull(env, step_fn, off, flag_fn, max_steps=420, d_cap=0.12, mode="creep", z_stop=None,
              gap_ok=0.025, rec=None, verbose=False, tag="pull"):
    """拉取双模式（探针实证）：
    creep: target = 当前 eef + step_fn(env)，desired ~0.8mm/步匀速前移，滞后增长突破摩擦；
           step 带 +z 分量即形成持续上偏钩载——夹持不打滑的关键（探针 4/4 成功）。
    gate:  target 每步前进 step_fn(env)（仅 lag<gap_ok 时），lag>0.05 冻结——适合自由滑移对象（横梁）。
    lag 超 d_cap 或 eef_z 超 z_stop 冻结 desired。flag 连续 10 步 → 成功。返回 (成功, 步数)。"""
    target = env.get_site_pos(GripSite).copy()
    ok_cnt, armed, calm = 0, False, 0
    t = 0
    for t in range(max_steps):
        end = env.get_site_pos(GripSite)
        lag = lag_of(env, off)
        if lag > 0.03:  # 已建立拉力 → armed；armed 后 lag 崩回 <8mm 连续 15 步 = 打滑，早退重抓
            armed, calm = True, 0
        elif armed and lag < 0.008:
            calm += 1
            if calm >= 10:
                if verbose:
                    print("    %s 打滑早退（t=%d lag=%.3f）" % (tag, t + 1, lag), flush=True)
                return False, t
        else:
            calm = 0
        if lag > d_cap or (z_stop is not None and end[2] > z_stop):
            act = np.array([0.0, 0.0, 0.0, -1.0])  # 冻结 desired，力保持在当前滞后水平
        elif mode == "creep":
            act = p_action(env, end + np.asarray(step_fn(env), dtype=float), gripper=-1, k=2.0)
        else:  # gate
            if lag < gap_ok:
                target = target + np.asarray(step_fn(env), dtype=float)
            act = p_action(env, target, gripper=-1) if lag <= 0.05 else \
                p_action(env, end, gripper=-1, k=1.0)
        _, _, _, _, info = env.step(act)
        snap(rec, env)
        ok_cnt = ok_cnt + 1 if flag_fn(info) else 0
        if ok_cnt >= 10:
            return True, t
        if verbose and (t + 1) % 40 == 0:
            print("      %s t=%d lag=%.3f" % (tag, t + 1, lag), flush=True)
    if verbose:
        print("    %s 未触发 flag（lag=%.3f）" % (tag, lag_of(env, off)), flush=True)
    return False, t


def drawer_episode(env, recorder=None, verbose=False):
    env.reset()
    env.robot.pos_min_bound = np.array([0.3, -0.2, 0.0])
    env.robot.pos_max_bound = np.array([0.72, 0.25, 0.45])
    off = env.get_site_pos(GripSite) - env.desired_position  # reset 后滞后≈0，即基座→世界平移
    site0 = env.get_site_pos("drawer")
    handle = np.array([site0[0] + 0.02, 0.0, 0.462])
    flag = lambda info: bool(info.get("is_success", 0))
    if verbose:
        print("    drawer_goal=%s handle=%s" % (
            np.round(env.get_site_pos("drawer_goal"), 3), np.round(handle, 3)), flush=True)
    got, t = grasp_from_above(env, handle, off, rec=recorder, verbose=verbose)
    if not got:
        return False, t, "grasp"
    zc = env.desired_position[2]  # z 限高：+z 爬顶会让指杆滑出把手，闭合平面 +5mm 后停 +z 进给
    ok, tp = ramp_pull(env, lambda e: np.array([-0.004, -0.003 if e.desired_position[1] > 0.003 else 0.0, 0.0015 if e.desired_position[2] < zc + 0.08 else 0.0]), off, flag,
                       z_stop=0.52, rec=recorder, verbose=verbose, tag="drawer")
    return ok, t + tp, "pull" if ok else "pull_no_flag"


def pick_drop_cube(env, rec=None, verbose=False):
    """抓 green_block → 移动到箱口上方 → 悬空松爪落入。"""
    goal = env.get_site_pos("cube_goal")
    phase, t, t2, hold = "above", 0, 0, 0
    while t < 360:
        t += 1
        end = env.get_site_pos(GripSite)
        blk = env.get_body_pos("green_block")
        if phase == "above":
            act = p_action(env, blk + [0, 0, 0.12], gripper=+1)
            if np.linalg.norm((end - blk)[[0, 1]]) < 0.010 and end[2] > blk[2] + 0.09:
                phase, t2 = "descend", t
        elif phase == "descend":
            act = p_action(env, blk + [0, 0, 0.0], gripper=+1, k=2.0)
            if end[2] - blk[2] < 0.012:
                phase, t2 = "grasp", t
        elif phase == "grasp":
            act = p_action(env, end, gripper=-1, k=1.0)
            if t - t2 > 20:
                phase, t2 = "lift", t
        elif phase == "lift":
            act = p_action(env, end + [0, 0, 0.1], gripper=-1, k=2.0)
            if blk[2] > 0.50:
                phase, t2 = "move", t
            elif t - t2 > 80:
                if verbose:
                    print("    lift 失败 blk_z=%.3f" % blk[2], flush=True)
                return False, t
        elif phase == "move":
            act = p_action(env, [goal[0], goal[1], 0.60], gripper=-1, k=2.5)
            if np.linalg.norm((end - goal)[[0, 1]]) < 0.012:
                phase, t2 = "drop", t
        elif phase == "drop":
            act = p_action(env, [goal[0], goal[1], 0.53], gripper=-1, k=1.5)
            if end[2] < 0.535:
                phase, t2 = "release", t
        elif phase == "release":
            act = p_action(env, end, gripper=+1, k=1.0)
            hold += 1
            if hold > 12:
                for _ in range(30):
                    env.step(p_action(env, [goal[0], goal[1], 0.62], gripper=+1, k=2.0))
                    snap(rec, env)
                return True, t
        env.step(act)
        snap(rec, env)
    return False, t


def drawerbox_episode(env, recorder=None, verbose=False):
    env.reset()
    env.robot.pos_min_bound = np.array([0.3, -0.2, 0.0])
    env.robot.pos_max_bound = np.array([0.72, 0.25, 0.45])
    off = env.get_site_pos(GripSite) - env.desired_position
    site0 = env.get_site_pos("drawer")
    handle = np.array([site0[0] + 0.02, 0.0, 0.437])
    flag = lambda info: bool(info.get("is_drawer_success", 0))
    got, t = grasp_from_above(env, handle, off, rec=recorder, verbose=verbose)
    if not got:
        return False, t, "grasp_drawer"
    zc = env.desired_position[2]
    ok, tp = ramp_pull(env, lambda e: np.array([-0.004, -0.003 if e.desired_position[1] > 0.003 else 0.0, 0.0015 if e.desired_position[2] < zc + 0.08 else 0.0]), off, flag,
                       z_stop=0.52, rec=recorder, verbose=verbose, tag="drawer")
    if not ok:
        return False, t + tp, "pull_no_flag"
    for attempt in range(3):
        ok2, t2 = pick_drop_cube(env, rec=recorder, verbose=verbose)
        if env._get_info().get("is_place_success", 0):
            return True, t + tp + t2, "place"
        if verbose:
            print("    place attempt%d: cube=%s goal=%s" % (
                attempt + 1, np.round(env.get_body_pos("green_block"), 3),
                np.round(env.get_site_pos("cube_goal"), 3)), flush=True)
    return False, t + tp, "place"


def cabinet_episode(env, recorder=None, verbose=False):
    env.reset()
    env.robot.pos_min_bound = np.array([0.3, -0.32, 0.05])
    env.robot.pos_max_bound = np.array([0.70, 0.30, 0.45])
    off = env.get_site_pos(GripSite) - env.desired_position
    # ---- 阶段1：解锁横梁 ----
    beam_left = env.get_site_pos("beam_left")
    handle = beam_left + np.array([-0.06, -0.23, 0.0])
    got, t = grasp_from_above(env, handle, off, z_above=0.08, rec=recorder, verbose=verbose)
    if not got:
        return False, t, "grasp_beam"
    flag_u = lambda info: bool(info.get("is_unlock_success", 0))
    ok, tp = ramp_pull(env, lambda e: np.array([0.0, -0.004, 0.0]), off, flag_u, max_steps=160,
                       mode="gate", rec=recorder, verbose=verbose, tag="beam")
    if verbose:
        print("    unlock=%s beam_left=%s" % (ok, np.round(env.get_site_pos("beam_left"), 3)), flush=True)
    if not ok:
        return False, t, "unlock"
    lift = env.get_site_pos(GripSite)
    for _ in range(35):  # 松爪退避到梁上方（固定目标，防 relative-target 失控上升）
        env.step(p_action(env, [lift[0], lift[1], 0.66], gripper=+1, k=2.0))
        snap(recorder, env)
    t2 = tp + 35  # 松爪退避步数计入耗时
    # ---- 阶段2：夹门把手 C 竖杆，弦线 lead 递增拉向 opened ----
    opened = env.get_site_pos("cabinet_left_opened")
    d_bid = int(env.mj_model.site("left_handle").bodyid[0])
    jid = next(j for j in range(env.mj_model.njnt) if env.mj_model.jnt_bodyid[j] == d_bid)
    jnt_local = env.mj_model.jnt_pos[jid].copy()

    def door_step(e):
        # 纯切向蠕进：切向 = z×(把手−铰链)，⊥闭合轴 y（与抽屉同款力传输，弦线 y 分量 95% 会挤出楔缝）
        site = env.get_site_pos("left_handle")
        xmat = env.mj_data.xmat[d_bid].reshape(3, 3)
        hinge = env.mj_data.xpos[d_bid] + xmat @ jnt_local
        tan = np.cross([0.0, 0.0, 1.0], site - hinge)
        n = float(np.linalg.norm(tan))
        if n < 1e-6:
            return np.zeros(3)
        tan = tan / n
        if float(np.dot(tan, opened - site)) < 0:
            tan = -tan
        return tan * 0.004 + np.array([0.0, 0.0, -0.0008])  # −z 微偏：滑向杆底支架卡紧

    flag_d = lambda info: bool(info.get("is_door_success", 0))
    t3 = 0
    for attempt in range(4):
        # 每次尝试把门推进一步（铰链 frictionloss 锁角），打滑即重新抓、从新角度继续拉
        rod = env.get_site_pos("left_handle") + np.array([0.0, 0.0, -0.035])
        got, tg = grasp_from_above(env, rod, off, z_above=0.10, rec=recorder, verbose=verbose)
        if not got:
            return False, t + t2 + t3 + tg, "grasp_door"
        ok, tp = ramp_pull(env, door_step, off, flag_d, max_steps=300, z_stop=0.615,
                           rec=recorder, verbose=verbose, tag="door#%d" % (attempt + 1))
        t3 += tg + tp
        if ok:
            return True, t + t2 + t3, "door"
        if verbose:
            print("    door 尝试%d 后 hinge=%.2frad" % (
                attempt + 1, env.mj_data.qpos[env.mj_model.jnt_qposadr[jid]]), flush=True)
    if verbose:
        info = env._get_info()
        m, d = env.mj_model, env.mj_data
        bid = m.site("left_handle").bodyid[0]
        gadr, gnum = int(m.body_geomadr[bid]), int(m.body_geomnum[bid])
        door_g = set(range(gadr, gadr + gnum))
        finger_g = {g for g in range(m.ngeom) if "finger" in m.geom(g).name}
        hinge = env.mj_data.qpos[m.jnt_qposadr[jid]]
        print("    door 超时: left_handle=%s opened=%s door=%s unlock=%s" % (
            np.round(env.get_site_pos("left_handle"), 3), np.round(opened, 3),
            info.get("is_door_success"), info.get("is_unlock_success")), flush=True)
        print("      finger_qpos=%.4f lag=%.3f hinge=%.2frad eef=%s" % (
            finger_qpos(env), lag_of(env, off), hinge,
            np.round(env.get_site_pos(GripSite), 3)), flush=True)
        for c in d.contact:
            if c.dist > -0.004 and (c.geom1 in door_g or c.geom2 in door_g or
                                    c.geom1 in finger_g or c.geom2 in finger_g):
                print("      contact g%d(%s)<->g%d(%s) %.4f" % (
                    c.geom1, m.body(m.geom_bodyid[c.geom1]).name,
                    c.geom2, m.body(m.geom_bodyid[c.geom2]).name, c.dist), flush=True)
    return False, t + t2 + t3, "door_timeout"


def main():
    tasks = [
        ("Drawer-v1", "DianaDrawer", drawer_episode),
        ("DrawerBox-v1", "DianaDrawerCube", drawerbox_episode),
        ("LockedCabinet-v1", "DianaCabinet", cabinet_episode),
    ]
    task_filter = [k.strip() for k in os.environ.get("TASKS", "").split(",") if k.strip()]
    for name, robot, ep_fn in tasks:
        if task_filter and not any(k in name for k in task_filter):
            continue
        env = robopal.make(name, robot=robot, render_mode=None, control_freq=20)
        env.is_randomize_end = False  # 用任务自带 init_qpos（构造器不透传该参）
        results = []
        for ep in range(3):
            rec = None
            if REC:
                os.makedirs(OUT_DIR, exist_ok=True)
                rec = make_recorder(env, os.path.join(OUT_DIR, "%s-ep%d.mp4" % (name, ep + 1)))
            ok, t, where = ep_fn(env, recorder=rec, verbose=True)
            if rec:
                rec.close(keep=ok)
            results.append(ok)
            print("[%s] ep%d: %s steps=%d stopped_at=%s" % (
                name, ep + 1, "SUCCESS" if ok else "FAIL", t, where), flush=True)
        env.close()
        print("%s 成功率: %d/%d" % (name, sum(results), len(results)), flush=True)


if __name__ == "__main__":
    main()
