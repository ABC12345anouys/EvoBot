"""MultiCubeStack 的 GraspNet 抓取位姿测试。

对比：
- center: 用物体中心作为抓取点
- graspnet: 用 GraspNet 预测的抓取点
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
STACK_ORDER = ["red_block", "green_block", "blue_block"]


def p_action(env, target, gripper=0.0, k=4.0):
    end = env.get_site_pos(GripSite)
    vel = np.clip(k * (np.asarray(target, float) - end), -1.0, 1.0)
    return np.append(vel, 1.0 if gripper > 0 else (-1.0 if gripper < 0 else 0.0))


class GraspPipeline:
    def __init__(self):
        from darwin.skills.perception.grasp import grasp_from_env
        self.grasp_fn = grasp_from_env

    def get_grasp(self, env, block, use_graspnet=True):
        if not use_graspnet:
            return env.get_body_pos(block).copy()
        rg = self.grasp_fn(env, block)
        if rg.get("success"):
            return np.array(rg["position"], float)
        return env.get_body_pos(block).copy()


def episode_stack(env, grasp_fn, use_graspnet, max_steps=600):
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
            bpos = grasp_fn(env, block, use_graspnet)
            target_z = goal[2] + 0.06 * len(placed) if placed else goal[2]
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
    success = True
    for block in STACK_ORDER:
        bp = env.get_body_pos(block)
        gp = env.get_site_pos(f"{block.split('_')[0]}_goal")
        if np.linalg.norm(bp - gp) > 0.03:
            success = False
    return success, t


def run(n=3, use_graspnet=True):
    env = robopal.make("MultiCubeStack-v1", robot="DianaTripleStack", render_mode=None, control_freq=20)
    gp = GraspPipeline()
    results = []
    for i in range(n):
        ok, steps = episode_stack(env, gp.get_grasp, use_graspnet)
        tag = "GRASPNET" if use_graspnet else "CENTER"
        print(f"  [{tag}] ep{i}: {'SUCCESS' if ok else 'FAIL'} steps={steps}")
        results.append(ok)
    env.close()
    return sum(results) / len(results)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"=== MultiCubeStack GraspNet vs 物体中心: {n} episodes ===")
    s_center = run(n, use_graspnet=False)
    print(f"物体中心抓取: {s_center:.0%}")
    s_gn = run(n, use_graspnet=True)
    print(f"GraspNet 抓取: {s_gn:.0%}")
    print(f"\n对比: center={s_center:.0%} vs graspnet={s_gn:.0%}")
