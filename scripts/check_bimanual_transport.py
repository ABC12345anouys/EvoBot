"""BimanualTransport-v0 纯控制器验证：经桌面交接把锤子放进纸箱。

- 布局：agent0 基座 world(1.1,0,0.3) 朝 -x（基座系旋转180°）；agent1 基座 (-0.3,0,0.3) 朝 +x。
  锤子 spawn (0.1,0,0.5) 只有 agent1 够得着；纸箱 (0.8,0,0.49) 只有 agent0 够得着 -> 必须交接。
- 策略：agent1 抓锤（候选抓点枚举+提升测试）-> 桌面中转点 (0.42,0) 放下 -> agent0 再抓 -> 
  移到纸箱上方缓放，锤子落入箱内（箱为开口容器，官方判据 hammer 距 carton 中心 < 4cm）。
- 诚实判据：env 官方 info['is_success']，无重置过滤。
运行：MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/check_bimanual_transport.py
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

import numpy as np
import mujoco
import robopal
from robopal.envs.base import MujocoEnv

MujocoEnv.close = lambda self: self.renderer.close()

BOUNDS_MIN = {"agent0": np.array([0.15, -0.30, 0.02]), "agent1": np.array([0.15, -0.30, 0.02])}
BOUNDS_MAX = {"agent0": np.array([0.75, 0.30, 0.68]), "agent1": np.array([0.75, 0.35, 0.68])}
HANDOFF = np.array([0.42, 0.0])


_BASE_CACHE = {}


def _base_frame(env, agent):
    if agent not in _BASE_CACHE:
        bid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, env.robot.base_link_name[agent])
        _BASE_CACHE[agent] = (env.mj_data.xpos[bid].copy(), env.mj_data.xmat[bid].reshape(3, 3).copy())
    return _BASE_CACHE[agent]


def p_action(env, agent, target, gripper=0.0, k=4.0):
    """误差在臂基座系计算（agent0 基座旋转 180°，世界系速度会被镜像）。"""
    end = env.get_site_pos("%s_grip_site" % agent[-1])
    bp, bm = _base_frame(env, agent)
    err_b = bm.T @ (np.asarray(target, dtype=float) - end)
    vel = np.clip(k * err_b, -1.0, 1.0)
    return np.append(vel, 1.0 if gripper > 0 else (-1.0 if gripper < 0 else 0.0))


def grasp_candidates(env):
    """hammer 局部长短轴上的候选抓点（世界系偏移），手柄处优先。"""
    quat = env.get_body_quat("hammer")  # wxyz
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    cands = []
    for d in (0.06, -0.06, 0.0, 0.10, -0.10):
        cands.append(R[:, 0] * d)
    for d in (0.06, -0.06, 0.10, -0.10):
        cands.append(R[:, 1] * d)
    return cands


def try_grasp(env, agent, acts, t, point, lift_z=0.56, verbose=False):
    """对某候选抓点执行 下降->闭合->提升测试；成功返回 (True, t)。"""
    t2 = t
    while t - t2 < 80:  # 对准
        t += 1
        acts[agent] = p_action(env, agent, point + [0, 0, 0.10], gripper=+1, k=3.0)
        env.step(acts)
        if np.linalg.norm(env.get_site_pos("%s_grip_site" % agent[-1]) - (point + [0, 0, 0.10])) < 0.02:
            break
    while t - t2 < 140:  # 下探
        t += 1
        acts[agent] = p_action(env, agent, point + [0, 0, 0.0], gripper=+1, k=1.5)
        env.step(acts)
        if env.get_site_pos("%s_grip_site" % agent[-1])[2] < point[2] + 0.015:
            break
    for _ in range(25):  # 闭合
        t += 1
        acts[agent] = p_action(env, agent, env.get_site_pos("%s_grip_site" % agent[-1]), gripper=-1, k=1.0)
        env.step(acts)
    z0 = env.get_body_pos("hammer")[2]
    ok_steps, zmax = 0, z0
    while t - t2 < 260:  # 提升测试（需连续保持）
        t += 1
        acts[agent] = p_action(env, agent, env.get_site_pos("%s_grip_site" % agent[-1]) + [0, 0, 0.08], gripper=-1, k=2.0)
        env.step(acts)
        zmax = max(zmax, env.get_body_pos("hammer")[2])
        if env.get_body_pos("hammer")[2] > z0 + 0.06:
            ok_steps += 1
            if ok_steps >= 10:
                return True, t
        else:
            ok_steps = 0
    if verbose:
        print("      grasp[%.2f,%.2f,%.2f] align_d=%.3f zmax-z0=%.3f" % (
            point[0], point[1], point[2],
            np.linalg.norm(env.get_site_pos("%s_grip_site" % agent[-1]) - point), zmax - z0), flush=True)
    for _ in range(10):  # 失败松开
        t += 1
        acts[agent] = p_action(env, agent, env.get_site_pos("%s_grip_site" % agent[-1]), gripper=+1, k=1.0)
        env.step(acts)
    return False, t


def carry(env, agent, acts, t, xy, z_via, z_end, verbose=False, vmax=0.35):
    """提起移动到 xy 上方 z_via，再缓降到 z_end（持物限速防甩脱）。"""

    def _hold(target, k):
        a = p_action(env, agent, target, gripper=-1, k=k)
        a[:3] = np.clip(a[:3], -vmax, vmax)
        return a

    end = env.get_site_pos("%s_grip_site" % agent[-1])
    while end[2] < z_via - 0.01 and t < 100000:
        t += 1
        acts[agent] = _hold([end[0], end[1], z_via], 2.0)
        env.step(acts)
        end = env.get_site_pos("%s_grip_site" % agent[-1])
        if verbose and t % 20 == 0:
            print("      lift t=%d grip=%s hammer=%s" % (t, end.round(3), env.get_body_pos("hammer").round(3)), flush=True)
    while np.linalg.norm(end[[0, 1]] - xy) > 0.012 and t < 100000:
        t += 1
        acts[agent] = _hold([xy[0], xy[1], z_via], 2.0)
        env.step(acts)
        end = env.get_site_pos("%s_grip_site" % agent[-1])
        if verbose and t % 20 == 0:
            print("      move t=%d grip=%s hammer=%s" % (t, end.round(3), env.get_body_pos("hammer").round(3)), flush=True)
    while end[2] > z_end + 0.008 and t < 100000:
        t += 1
        acts[agent] = _hold([xy[0], xy[1], z_end], 1.2)
        env.step(acts)
        end = env.get_site_pos("%s_grip_site" % agent[-1])
        if verbose and t % 20 == 0:
            print("      desc t=%d grip=%s hammer=%s" % (t, end.round(3), env.get_body_pos("hammer").round(3)), flush=True)
    return t


def episode(env, max_steps=1600, verbose=False):
    env.reset()
    _BASE_CACHE.clear()
    env.desired_positions = {ag: env.init_pos[ag].copy() for ag in env.agents}
    env.pos_min_bound = {ag: BOUNDS_MIN[ag].copy() for ag in env.agents}
    env.pos_max_bound = {ag: BOUNDS_MAX[ag].copy() for ag in env.agents}
    acts = {"agent0": np.zeros(4), "agent1": np.zeros(4)}
    t = 0
    for _ in range(150):  # 等锤子从空中 spawn 沉降到桌面
        env.step(acts)
        t += 1

    # ---- agent1：抓锤子 ----
    agent = "agent1"
    ok = False
    for c in grasp_candidates(env):
        point = env.get_body_pos("hammer") + c
        if point[2] > 0.55 or point[2] < 0.40:
            continue
        ok, t = try_grasp(env, agent, acts, t, point, verbose=verbose)
        if ok:
            if verbose: print("    agent1 抓住 hammer@offset=%s (t=%d)" % (c.round(3), t), flush=True)
            break
    if not ok:
        return False, t, "agent1_grasp"
    t = carry(env, agent, acts, t, HANDOFF, 0.60, 0.475, verbose)
    for _ in range(20):  # 放下锤子
        t += 1
        acts[agent] = p_action(env, agent, env.get_site_pos("1_grip_site"), gripper=+1, k=1.0)
        env.step(acts)
    while t < max_steps:  # agent1 退到旁边
        t += 1
        acts[agent] = p_action(env, agent, [0.32, 0.20, 0.64], gripper=+1, k=2.0)
        env.step(acts)
        if np.linalg.norm(env.get_site_pos("1_grip_site") - [0.32, 0.20, 0.64]) < 0.03:
            break

    # ---- agent0：抓锤子放入纸箱 ----
    agent = "agent0"
    ok = False
    for c in grasp_candidates(env):
        point = env.get_body_pos("hammer") + c
        if point[2] > 0.55 or point[2] < 0.40:
            continue
        ok, t = try_grasp(env, agent, acts, t, point, verbose=verbose)
        if ok:
            if verbose: print("    agent0 抓住 hammer@offset=%s (t=%d)" % (c.round(3), t), flush=True)
            break
    if not ok:
        return False, t, "agent0_grasp"
    carton = env.get_body_pos("carton")
    t = carry(env, agent, acts, t, carton[:2], 0.64, 0.505, verbose)  # 伸入箱口内松爪
    ok_steps = 0
    while t < max_steps:  # 松爪投放，判官方成功
        t += 1
        acts[agent] = p_action(env, agent, env.get_site_pos("0_grip_site"), gripper=+1, k=1.0)
        env.step(acts)
        d = float(np.linalg.norm(env.get_body_pos("hammer") - env.get_body_pos("carton")))
        if d < 0.04:
            ok_steps += 1
            if ok_steps >= 10:
                return True, t, "placed(d=%.3f)" % d
        else:
            ok_steps = 0
    d = float(np.linalg.norm(env.get_body_pos("hammer") - env.get_body_pos("carton")))
    return False, t, "final(d=%.3f, hammer=%s)" % (d, env.get_body_pos("hammer").round(3))


def main():
    env = robopal.make("BimanualTransport-v0", robot="DualPandaTransport",
                       render_mode=None, control_freq=20)
    results = []
    for ep in range(3):
        ok, t, where = episode(env, verbose=True)
        results.append(ok)
        print("[BimanualTransport] ep%d: %s steps=%d at=%s" % (
            ep + 1, "SUCCESS" if ok else "FAIL", t, where), flush=True)
    env.close()
    print("成功率: %d/%d" % (sum(results), len(results)), flush=True)


if __name__ == "__main__":
    main()
