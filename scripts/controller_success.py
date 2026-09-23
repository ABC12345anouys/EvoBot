"""纯控制器（脚本化策略，无 LLM）真正完成任务：验证 + 录制。

诚实化：PickAndPlace 出生点离目标 <8cm 重开；成功要求 is_success 且方块曾抬离桌面。
Drawer 从模型现场读取滑动关节世界轴向与把手位置，抓取尝试两种朝向。
运行：MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/controller_success.py
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

import numpy as np
import cv2
import mujoco
import robopal
from robopal.envs.base import MujocoEnv

MujocoEnv.close = lambda self: self.renderer.close()  # 绕过 robopal close() 内的 os._exit(0)

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "videos"))
W, H, FPS = 640, 480, 20
GripSite = "0_grip_site"


def p_action(env, target, gripper=0.0, k=4.0):
    end = env.get_site_pos(GripSite)
    vel = np.clip(k * (np.asarray(target, dtype=float) - end), -1.0, 1.0)
    return np.append(vel, 1.0 if gripper > 0 else (-1.0 if gripper < 0 else 0.0))


def make_recorder(env, path):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.45, 0.0, 0.40]
    cam.distance, cam.azimuth, cam.elevation = 2.0, 135.0, -22.0
    renderer = mujoco.Renderer(env.mj_model, height=H, width=W)
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))

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


# ================= PickAndPlace =================
def pick_place_episode(env, recorder=None, max_steps=700, max_attempts=4, verbose=False):
    env.reset()
    # 诚实化：出生点离目标 >= 8cm 才算有效 episode
    for _ in range(30):
        block = env.get_body_pos("green_block")
        goal = env.get_site_pos("goal_site")
        if np.linalg.norm(block - goal) >= 0.08:
            break
        env.reset()
    phase, t, t2, hold, attempt = "above", 0, 0, 0, 0
    max_block_z = 0.0
    while t < max_steps:
        t += 1
        end = env.get_site_pos(GripSite)
        block = env.get_body_pos("green_block")
        max_block_z = max(max_block_z, block[2])
        if phase == "above":
            act = p_action(env, block + [0, 0, 0.12], gripper=+1)
            if np.linalg.norm((end - block)[[0, 1]]) < 0.012 and end[2] > block[2] + 0.09:
                phase, t2 = "descend", t
        elif phase == "descend":
            act = p_action(env, block + [0, 0, 0.002], gripper=+1, k=2.0)
            if end[2] - block[2] < 0.012:
                phase, t2 = "grasp", t
        elif phase == "grasp":
            act = p_action(env, end, gripper=-1, k=1.0)
            if t - t2 > 18:
                phase, t2 = "lift", t
        elif phase == "lift":
            act = p_action(env, end + [0, 0, 0.1], gripper=-1, k=2.0)
            if block[2] > 0.52:
                phase, t2 = "move", t
            elif t - t2 > 80:
                attempt += 1
                if verbose: print("    lift 失败(重试%d)" % attempt, flush=True)
                if attempt >= max_attempts:
                    return False, t, phase
                phase, t2 = "above", t
        elif phase == "move":
            goal = env.get_site_pos("goal_site")
            act = p_action(env, [goal[0], goal[1], max(0.60, goal[2] + 0.14)], gripper=-1, k=2.5)
            if np.linalg.norm((end - goal)[[0, 1]]) < 0.012:
                phase, t2 = "place", t
        elif phase == "place":
            goal = env.get_site_pos("goal_site")
            act = p_action(env, goal + [0, 0, 0.002], gripper=-1, k=1.2)
            if block[2] - goal[2] < 0.022 or t - t2 > 100:
                phase, t2 = "release", t
        elif phase == "release":
            act = p_action(env, end, gripper=+1, k=1.0)
            hold += 1
            if hold > 15:
                phase = "retreat"
        elif phase == "retreat":
            act = p_action(env, end + [0, 0, 0.08], gripper=+1, k=2.0)
        _, _, _, _, info = env.step(act)
        if recorder: recorder.snap(env)
        if info.get("is_success", 0) > 0 and max_block_z > 0.52:
            return True, t, phase
    return False, max_steps, phase


def run_pick_place(n_episodes=5):
    from robopal.robots.panda import PandaPickAndPlace
    env = robopal.make("PickAndPlace-v1", robot=PandaPickAndPlace,
                       render_mode=None, control_freq=FPS, is_randomize_object=True)
    results, video_done = [], False
    for ep in range(n_episodes):
        path = os.path.join(OUT_DIR, "PickAndPlace-v1-success.mp4")
        rec = None if (video_done or ep >= 3) else make_recorder(env, path)
        ok, t, phase = pick_place_episode(env, recorder=rec, verbose=True)
        if rec:
            rec.close(keep=ok and not video_done)
            if ok and not video_done:
                video_done = True
                print("  成功视频: %s" % path, flush=True)
        results.append(ok)
        print("[PickAndPlace] ep%d: %s steps=%d last_phase=%s" % (ep+1, "SUCCESS" if ok else "FAIL", t, phase), flush=True)
    env.close()
    return results


# ================= Drawer =================
def drawer_geometry(env):
    """运行时读取：抽屉滑动世界轴向、把手世界坐标。"""
    model, data = env.mj_model, env.mj_data
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer:joint")
    bid = model.jnt_bodyid[jid]
    R = data.xmat[bid].reshape(3, 3)
    axis_w = R @ model.jnt_axis[jid]
    drawer_site = env.get_site_pos("drawer")
    goal_site = env.get_site_pos("drawer_goal")
    knob = drawer_site + R @ np.array([0.0, 0.02, 0.0])  # 把手相对 drawer site 局部偏移
    s = 1.0 if np.dot(axis_w, goal_site - drawer_site) > 0 else -1.0
    return axis_w * s, knob, drawer_site, goal_site


def drawer_episode(env, recorder=None, max_steps=500, verbose=False):
    env.reset()
    pull_dir, knob, _, _ = drawer_geometry(env)
    t = 0
    for yaw in [0.0, np.pi / 2]:
        pre = knob - pull_dir * 0.07 + [0, 0, 0.06]
        while t < 120:
            t += 1
            env.step(p_action(env, pre, gripper=+1, k=3.0))
            if recorder: recorder.snap(env)
            if np.linalg.norm(env.get_site_pos(GripSite) - pre) < 0.02:
                break
        while t < 200:
            t += 1
            env.step(p_action(env, knob + [0, 0, 0.002], gripper=+1, k=1.2))
            if recorder: recorder.snap(env)
            if np.linalg.norm(env.get_site_pos(GripSite) - knob) < 0.02:
                break
        if verbose: print("    对准把手 d=%.3f" % np.linalg.norm(env.get_site_pos(GripSite) - knob), flush=True)
        for _ in range(15):
            t += 1
            env.step(p_action(env, env.get_site_pos(GripSite), gripper=-1, k=1.0))
            if recorder: recorder.snap(env)
        d0 = env.get_site_pos("drawer")
        for _ in range(12):
            t += 1
            env.step(p_action(env, env.get_site_pos(GripSite) + pull_dir * 0.04, gripper=-1, k=2.0))
            if recorder: recorder.snap(env)
        moved = np.linalg.norm(env.get_site_pos("drawer") - d0)
        if verbose: print("    yaw=%.2f 试拉位移=%.1fmm" % (yaw, moved * 1000), flush=True)
        if moved > 0.005:  # 抓住了 -> 持续拉出
            while t < max_steps:
                t += 1
                act = p_action(env, env.get_site_pos(GripSite) + pull_dir * 0.03, gripper=-1, k=2.0)
                _, _, _, _, info = env.step(act)
                if recorder: recorder.snap(env)
                if info.get("is_success", 0) > 0:
                    for _ in range(25):
                        env.step(p_action(env, env.get_site_pos(GripSite), gripper=-1, k=1.0))
                        if recorder: recorder.snap(env)
                    return True, t
            break
        for _ in range(8):  # 没抓住：松开退开换朝向
            t += 1
            env.step(p_action(env, env.get_site_pos(GripSite) - pull_dir * 0.02 + [0, 0, 0.04], gripper=+1, k=2.0))
            if recorder: recorder.snap(env)
    return False, t


def run_drawer(n_episodes=3):
    from robopal.robots.diana_med import DianaDrawer
    env = robopal.make("Drawer-v1", robot=DianaDrawer, render_mode=None, control_freq=FPS)
    pull, knob, d0, g0 = drawer_geometry(env)
    print("[Drawer] 轴向=%s 把手=%s drawer=%s goal=%s" % (pull.round(3), knob.round(3), d0.round(3), g0.round(3)), flush=True)
    results, video_done = [], False
    for ep in range(n_episodes):
        path = os.path.join(OUT_DIR, "Drawer-v1-success.mp4")
        rec = None if (video_done or ep >= 2) else make_recorder(env, path)
        ok, t = drawer_episode(env, recorder=rec, verbose=True)
        if rec:
            rec.close(keep=ok and not video_done)
            if ok and not video_done:
                video_done = True
                print("  成功视频: %s" % path, flush=True)
        results.append(ok)
        print("[Drawer] ep%d: %s steps=%d" % (ep + 1, "SUCCESS" if ok else "FAIL", t), flush=True)
    env.close()
    return results


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    r1 = run_pick_place(n_episodes=5)
    r2 = run_drawer(n_episodes=3)
    print("\n===== 汇总 =====", flush=True)
    print("PickAndPlace 成功率: %d/%d" % (sum(r1), len(r1)), flush=True)
    print("Drawer        成功率: %d/%d" % (sum(r2), len(r2)), flush=True)


if __name__ == "__main__":
    main()
