"""MultiCubeStack-v1 纯控制器验证：红->绿->蓝 依次堆到固定目标 [0.55,0.1]。

- 方块 4cm，桌面 0.42，静止中心 0.44；目标 red/green/blue = 0.44/0.48/0.52。
- DianaTrippleStack 的 pos_min_bound z=0.03(base) 会把抓取高度 0.44 挡住，脚本放宽到 0。
- 诚实判据：env 官方 info['is_success']（三块同时距各自 goal < 2cm），无出生点过滤。
运行：MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/check_multi_stack.py
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

GripSite = "0_grip_site"
ORDER = [("red_block", "red_goal"), ("green_block", "green_goal"), ("blue_block", "blue_goal")]


def p_action(env, target, gripper=0.0, k=4.0):
    end = env.get_site_pos(GripSite)
    vel = np.clip(k * (np.asarray(target, dtype=float) - end), -1.0, 1.0)
    return np.append(vel, 1.0 if gripper > 0 else (-1.0 if gripper < 0 else 0.0))


def move_block(env, block, goal_name, max_steps=250, max_attempts=3, verbose=False):
    """抓取 block 放到 goal site（PickAndPlace 同款状态机）。"""
    t_local = 0
    for attempt in range(max_attempts):
        phase, t2, hold = "above", t_local, 0
        lifted = False
        while t_local < max_steps:
            t_local += 1
            end = env.get_site_pos(GripSite)
            blk = env.get_body_pos(block)
            goal = env.get_site_pos(goal_name)
            if phase == "above":
                act = p_action(env, blk + [0, 0, 0.12], gripper=+1)
                if np.linalg.norm((end - blk)[[0, 1]]) < 0.012 and end[2] > blk[2] + 0.09:
                    phase, t2 = "descend", t_local
            elif phase == "descend":
                act = p_action(env, blk + [0, 0, 0.0], gripper=+1, k=2.0)
                if end[2] - blk[2] < 0.012:
                    phase, t2 = "grasp", t_local
            elif phase == "grasp":
                act = p_action(env, end, gripper=-1, k=1.0)
                if t_local - t2 > 18:
                    phase, t2 = "lift", t_local
            elif phase == "lift":
                act = p_action(env, end + [0, 0, 0.1], gripper=-1, k=2.0)
                if blk[2] > 0.52:
                    lifted = True
                    phase, t2 = "move", t_local
                elif t_local - t2 > 80:
                    break  # 没抓住，重试
            elif phase == "move":
                act = p_action(env, [goal[0], goal[1], max(0.58, goal[2] + 0.12)], gripper=-1, k=2.5)
                if np.linalg.norm((end - goal)[[0, 1]]) < 0.012:
                    phase, t2 = "place", t_local
            elif phase == "place":
                act = p_action(env, goal + [0, 0, 0.002], gripper=-1, k=1.0)
                if blk[2] - goal[2] < 0.018 or t_local - t2 > 100:
                    phase, t2 = "release", t_local
            elif phase == "release":
                act = p_action(env, end, gripper=+1, k=1.0)
                hold += 1
                if hold > 15:
                    phase = "retreat"
            elif phase == "retreat":
                act = p_action(env, end + [0, 0, 0.08], gripper=+1, k=2.0)
                if end[2] > goal[2] + 0.10:
                    return True, t_local
            env.step(act)
        if verbose and not lifted:
            print("    %s lift 失败(重试%d)" % (block, attempt + 1), flush=True)
    return False, t_local


def stack_episode(env, verbose=False):
    env.reset()
    env.robot.pos_min_bound = np.array([0.3, -0.2, 0.0])  # 放宽抓取高度
    total = 0
    for block, goal_name in ORDER:
        ok, t = move_block(env, block, goal_name, verbose=verbose)
        total += t
        info = env._get_info()
        if verbose:
            print("    %s -> %s: %s (%d steps, is_success=%s)" % (
                block, goal_name, "OK" if ok else "FAIL", t, info["is_success"]), flush=True)
        if not ok:
            return False, total, block
    # 最终官方判据
    for _ in range(10):
        obs, r, term, trunc, info = env.step(np.array([0, 0, 0, 0]))
    return bool(info.get("is_success", 0)), total, "final"


def main():
    env = robopal.make("MultiCubeStack-v1", robot="DianaTripleStack",
                       render_mode=None, control_freq=20, is_randomize_object=True)
    results = []
    for ep in range(3):
        ok, t, where = stack_episode(env, verbose=True)
        results.append(ok)
        print("[MultiCubeStack] ep%d: %s steps=%d stopped_at=%s" % (
            ep + 1, "SUCCESS" if ok else "FAIL", t, where), flush=True)
    env.close()
    print("成功率: %d/%d" % (sum(results), len(results)), flush=True)


if __name__ == "__main__":
    main()
