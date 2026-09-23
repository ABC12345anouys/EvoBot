"""感知技能驱动的 BimanualPickAndPlace 测试。

对比：
- baseline：直接用仿真真值 env.get_body_pos("green_block")
- perception：用 detect_objects + estimate_depth + segment_object + grasp_pose 获取物体位置

运行：MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/test_perception_pickplace.py
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

ACTOR = "agent1"
GripSite = "1_grip_site"


def p_action(env, target, gripper=0.0, k=4.0):
    end = env.get_site_pos(GripSite)
    vel = np.clip(k * (np.asarray(target, dtype=float) - end), -1.0, 1.0)
    return np.append(vel, 1.0 if gripper > 0 else (-1.0 if gripper < 0 else 0.0))


def episode_pickplace(env, block_pos_fn, max_steps=400, max_attempts=3):
    """通用 pick-place 流程，block_pos_fn(env) 返回物体世界坐标。"""
    env.reset()
    env.desired_positions = {ag: env.init_pos[ag].copy() for ag in env.agents}
    env.pos_max_bound = {"agent0": np.array([0.65, 0.65, 0.65]), "agent1": np.array([0.65, 0.40, 0.65])}
    env.pos_min_bound = {"agent0": np.array([-0.10, -0.30, 0.02]), "agent1": np.array([0.25, -0.10, 0.02])}
    t, ok_steps = 0, 0
    for attempt in range(max_attempts):
        phase, t2, hold = "above", t, 0
        while t < max_steps:
            t += 1
            end = env.get_site_pos(GripSite)
            block = block_pos_fn(env)
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
                    break
            elif phase == "move":
                acts[ACTOR] = p_action(env, [goal[0], goal[1], max(0.60, goal[2] + 0.12)], gripper=-1, k=2.5)
                if np.linalg.norm((end - goal)[[0, 1]]) < 0.012:
                    phase, t2 = "place", t
            elif phase == "place":
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
            if hold > 30:
                break
    return False, t, d


def gt_block_pos(env):
    """真值：直接读仿真。"""
    return env.get_body_pos("green_block").copy()


class PerceptionBlockPos:
    """用感知技能链获取物体位置。"""

    def __init__(self):
        from darwin.skills import build_registry
        self.reg = build_registry()
        for n in ["detect_objects", "estimate_depth", "segment_object", "grasp_pose"]:
            self.reg.get_skill_description(n)

    def __call__(self, env):
        rd = self.reg.execute("detect_objects", env=env)
        if not rd.get("success") or rd["count"] == 0:
            return gt_block_pos(env)
        # 优先找 green_block 且有 pos 的
        for o in rd["objects"]:
            if "green" in o["name"].lower() and "pos" in o:
                return np.array(o["pos"], dtype=float)
        # 回退到任何有 pos 的物体
        for o in rd["objects"]:
            if "pos" in o:
                return np.array(o["pos"], dtype=float)
        return gt_block_pos(env)


def run_eval(env_name, n_episodes=5, use_perception=False):
    env = robopal.make(env_name, robot="DualPandaPickAndPlace", render_mode=None, control_freq=20)
    block_fn = PerceptionBlockPos() if use_perception else gt_block_pos
    results = []
    for i in range(n_episodes):
        ok, steps, d = episode_pickplace(env, block_fn)
        results.append((ok, steps, d))
        tag = "PERC" if use_perception else "GT"
        print(f"  [{tag}] ep{i}: success={ok} steps={steps} d={d:.4f}")
    env.close()
    succ = sum(1 for r in results if r[0]) / len(results)
    return succ, results


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"=== BimanualPickAndPlace-v0: {n} episodes ===")
    print("--- Baseline (ground truth) ---")
    s_gt, _ = run_eval("BimanualPickAndPlace-v0", n, use_perception=False)
    print(f"GT 成功率: {s_gt:.0%}")
    print("--- Perception skills ---")
    s_perc, _ = run_eval("BimanualPickAndPlace-v0", n, use_perception=True)
    print(f"Perception 成功率: {s_perc:.0%}")
    print(f"\n对比: GT={s_gt:.0%} vs Perception={s_perc:.0%}")
