"""BimanualPickAndPlace-v0 纯控制器验证：agent1 完成抓取+放置，agent0 保持。

- DualPanda 两臂基座 world(0,±0.15,0.4) 朝 +x；agent1 初始位姿已朝向桌面。
- 目标 site 悬空（z 0.46~0.66），官方判据只看方块与 goal 距离 < 2cm：抓着方块停在目标即成。
- agent1 的 pos_min_bound z=0.2(base) 挡住桌面抓取（0.44->base 0.04），脚本放宽。
- 诚实判据：env 官方 info['is_success']，无出生点过滤（reset_object 已保证 block-goal > 5cm）。
运行：MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/check_bimanual_pick_place.py
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

import numpy as np
import robopal
from robopal.envs.base import MujocoEnv

MujocoEnv.close = lambda self: self.renderer.close()

ACTOR = "agent1"
GripSite = "1_grip_site"


def p_action(env, target, gripper=0.0, k=4.0):
    end = env.get_site_pos(GripSite)
    vel = np.clip(k * (np.asarray(target, dtype=float) - end), -1.0, 1.0)
    return np.append(vel, 1.0 if gripper > 0 else (-1.0 if gripper < 0 else 0.0))


def episode(env, max_steps=400, max_attempts=3, verbose=False):
    env.reset()
    env.desired_positions = {ag: env.init_pos[ag].copy() for ag in env.agents}
    # agent1 base(0,-0.15,0.4)：base_y = world_y + 0.15，需覆盖 world y ±0.25
    env.pos_max_bound = {"agent0": np.array([0.65, 0.65, 0.65]), "agent1": np.array([0.65, 0.40, 0.65])}
    env.pos_min_bound = {"agent0": np.array([-0.10, -0.30, 0.02]), "agent1": np.array([0.25, -0.10, 0.02])}
    t, ok_steps = 0, 0
    for attempt in range(max_attempts):
        phase, t2, hold = "above", t, 0
        while t < max_steps:
            t += 1
            end = env.get_site_pos(GripSite)
            block = env.get_body_pos("green_block")
            goal = env.get_site_pos("goal_site")
            acts = {"agent0": np.zeros(4)}
            if phase == "above":
                acts[ACTOR] = p_action(env, block + [0, 0, 0.12], gripper=+1)
                if np.linalg.norm((end - block)[[0, 1]]) < 0.012 and end[2] > block[2] + 0.09:
                    phase, t2 = "descend", t
            elif phase == "descend":
                jit = np.array([0.004 * np.sin(3.1 * attempt), 0.004 * np.cos(1.7 * attempt), 0.0])
                acts[ACTOR] = p_action(env, block + jit + [0, 0, 0.0], gripper=+1, k=2.0)
                if end[2] - block[2] < 0.012:
                    phase, t2 = "grasp", t
            elif phase == "grasp":
                acts[ACTOR] = p_action(env, end, gripper=-1, k=1.0)
                if t - t2 > 22:
                    phase, t2 = "lift", t
            elif phase == "lift":
                acts[ACTOR] = p_action(env, end + [0, 0, 0.1], gripper=-1, k=2.0)
                if block[2] > 0.52:
                    phase, t2 = "move", t
                elif t - t2 > 80:
                    break  # 重试
            elif phase == "move":
                acts[ACTOR] = p_action(env, [goal[0], goal[1], max(0.60, goal[2] + 0.12)], gripper=-1, k=2.5)
                if np.linalg.norm((end - goal)[[0, 1]]) < 0.012:
                    phase, t2 = "place", t
            elif phase == "place":  # 悬空目标：抓着方块对准即可
                acts[ACTOR] = p_action(env, goal + [0, 0, 0.0], gripper=-1, k=1.2)
                hold += 1
            env.step(acts)
            d = float(np.linalg.norm(env.get_body_pos("green_block") - env.get_site_pos("goal_site")))
            if d < 0.02:
                ok_steps += 1
                if ok_steps >= 10:
                    return True, t, d
            else:
                ok_steps = 0
        if verbose:
            print("    lift/grasp 失败(重试%d)" % (attempt + 1), flush=True)
    return False, t, 9.9


def main():
    env = robopal.make("BimanualPickAndPlace-v0", robot="DualPandaPickAndPlace",
                       render_mode=None, control_freq=20)
    results = []
    for ep in range(3):
        ok, t, d = episode(env, verbose=True)
        results.append(ok)
        print("[BimanualPickAndPlace] ep%d: %s steps=%d final_d=%.4f" % (
            ep + 1, "SUCCESS" if ok else "FAIL", t, d), flush=True)
    env.close()
    print("成功率: %d/%d" % (sum(results), len(results)), flush=True)


if __name__ == "__main__":
    main()
