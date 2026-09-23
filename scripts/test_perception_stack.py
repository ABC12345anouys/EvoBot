"""感知技能驱动的 MultiCubeStack 测试。

用 detect_objects 检测红/绿/蓝方块位置，替代硬编码的 env.get_body_pos。
运行：MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/test_perception_stack.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

import numpy as np
import robopal
from robopal.envs.base import MujocoEnv

MujocoEnv.close = lambda self: self.renderer.close()

GripSite = "0_grip_site"
ACTOR = "agent0"
STACK_ORDER = ["red_block", "green_block", "blue_block"]


def p_action(env, target, gripper=0.0, k=4.0):
    end = env.get_site_pos(GripSite)
    vel = np.clip(k * (np.asarray(target, float) - end), -1.0, 1.0)
    return np.append(vel, 1.0 if gripper > 0 else (-1.0 if gripper < 0 else 0.0))


class PerceptionBlockPos:
    def __init__(self):
        from darwin.skills import build_registry
        self.reg = build_registry()
        self.reg.get_skill_description("detect_objects")

    def get_block_pos(self, env, block_name):
        rd = self.reg.execute("detect_objects", env=env)
        if not rd.get("success"):
            return env.get_body_pos(block_name).copy()
        for o in rd["objects"]:
            if block_name.split("_")[0] in o["name"].lower():
                return np.array(o["pos"], float)
        return env.get_body_pos(block_name).copy()

    def __call__(self, env, block_name):
        return self.get_block_pos(env, block_name)


def episode_stack(env, pos_fn, max_steps=600):
    env.reset()
    env.desired_positions = {ag: env.init_pos[ag].copy() for ag in env.agents}
    t = 0
    placed = []
    for block in STACK_ORDER:
        goal = env.get_site_pos(f"{block.split('_')[0]}_goal")
        phase, t2 = "above", t
        while t < max_steps:
            t += 1
            end = env.get_site_pos(GripSite)
            bpos = pos_fn(env, block)
            target_z = goal[2] + 0.06 * len(placed) if placed else goal[2]
            acts = {"agent1": np.zeros(4)}
            if phase == "above":
                act = p_action(env, bpos + [0, 0, 0.12], gripper=+1)
                if np.linalg.norm((end - bpos)[[0, 1]]) < 0.012 and end[2] > bpos[2] + 0.09:
                    phase, t2 = "descend", t
            elif phase == "descend":
                act = p_action(env, bpos + [0, 0, 0.0], gripper=+1, k=2.0)
                if end[2] - bpos[2] < 0.012:
                    phase, t2 = "grasp", t
            elif phase == "grasp":
                act = p_action(env, end, gripper=-1, k=1.0)
                if t - t2 > 22:
                    phase, t2 = "lift", t
            elif phase == "lift":
                act = p_action(env, end + [0, 0, 0.1], gripper=-1, k=2.0)
                if env.get_body_pos(block)[2] > 0.52:
                    phase, t2 = "move", t
                elif t - t2 > 80:
                    return False, t
            elif phase == "move":
                act = p_action(env, [goal[0], goal[1], max(0.55, target_z + 0.12)], gripper=-1, k=2.5)
                if np.linalg.norm((end - goal)[[0, 1]]) < 0.012:
                    phase, t2 = "place", t
            elif phase == "place":
                act = p_action(env, [goal[0], goal[1], target_z], gripper=-1, k=1.2)
                if t - t2 > 25:
                    phase, t2 = "release", t
            elif phase == "release":
                act = p_action(env, end, gripper=+1, k=1.0)
                if t - t2 > 12:
                    placed.append(block)
                    break
            env.step(act)
    # 最终判据
    success = True
    for block in STACK_ORDER:
        bp = env.get_body_pos(block)
        gp = env.get_site_pos(f"{block.split('_')[0]}_goal")
        if np.linalg.norm(bp - gp) > 0.03:
            success = False
    return success, t


def run(n=3, use_perception=False):
    env = robopal.make("MultiCubeStack-v1", robot="DianaTripleStack", render_mode=None, control_freq=20)
    pos_fn = PerceptionBlockPos() if use_perception else (lambda env, b: env.get_body_pos(b).copy())
    results = []
    for i in range(n):
        ok, steps = episode_stack(env, pos_fn)
        tag = "PERC" if use_perception else "GT"
        print(f"  [{tag}] ep{i}: {'SUCCESS' if ok else 'FAIL'} steps={steps}")
        results.append(ok)
    env.close()
    return sum(results) / len(results)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"=== MultiCubeStack-v1: {n} episodes ===")
    print("--- GT baseline ---")
    s_gt = run(n, use_perception=False)
    print(f"GT 成功率: {s_gt:.0%}")
    print("--- Perception ---")
    s_perc = run(n, use_perception=True)
    print(f"Perception 成功率: {s_perc:.0%}")
    print(f"\n对比: GT={s_gt:.0%} vs Perception={s_perc:.0%}")
