"""测试 GraspNet 抓取位姿技能的端到端效果(用真值点云)。
对比:
- center: 直接用物体中心作为抓取点
- graspnet: 用 GraspNet 预测的抓取点
运行: MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/test_graspnet_pose.py
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
    vel = np.clip(k * (np.asarray(target, float) - end), -1.0, 1.0)
    return np.append(vel, 1.0 if gripper > 0 else (-1.0 if gripper < 0 else 0.0))


class GraspPipeline:
    def __init__(self):
        from darwin.skills.perception.grasp import grasp_from_env
        self.grasp_fn = grasp_from_env

    def get_grasp(self, env, use_graspnet=True):
        if not use_graspnet:
            return env.get_body_pos("green_block").copy()
        rg = self.grasp_fn(env, "green_block")
        if rg.get("success"):
            return np.array(rg["position"], float)
        return env.get_body_pos("green_block").copy()


def episode(env, grasp_fn, use_graspnet, max_steps=400, max_attempts=3):
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
            grasp_pt = grasp_fn(env, use_graspnet)
            goal = env.get_site_pos("goal_site")
            acts = {"agent0": np.zeros(4)}
            if phase == "above":
                acts[ACTOR] = p_action(env, grasp_pt + [0, 0, 0.12], gripper=+1)
                if np.linalg.norm((end - grasp_pt)[[0, 1]]) < 0.015 and end[2] > grasp_pt[2] + 0.08:
                    phase, t2 = "descend", t
            elif phase == "descend":
                acts[ACTOR] = p_action(env, grasp_pt + [0, 0, 0.0], gripper=+1, k=2.0)
                if end[2] - grasp_pt[2] < 0.015:
                    phase, t2 = "grasp", t
            elif phase == "grasp":
                acts[ACTOR] = p_action(env, end, gripper=-1, k=1.0)
                if t - t2 > 22:
                    phase, t2 = "lift", t
            elif phase == "lift":
                acts[ACTOR] = p_action(env, end + [0, 0, 0.1], gripper=-1, k=2.0)
                if env.get_body_pos("green_block")[2] > 0.52:
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


def run(n=5, use_graspnet=True):
    env = robopal.make("BimanualPickAndPlace-v0", robot="DualPandaPickAndPlace", render_mode=None, control_freq=20)
    gp = GraspPipeline()
    results = []
    for i in range(n):
        ok, steps, d = episode(env, gp.get_grasp, use_graspnet)
        tag = "GRASPNET" if use_graspnet else "CENTER"
        print(f"  [{tag}] ep{i}: {'OK' if ok else 'FAIL'} steps={steps} d={d:.4f}")
        results.append(ok)
    env.close()
    return sum(results) / len(results)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    print(f"=== GraspNet 抓取位姿 vs 物体中心: {n} episodes ===")
    s_center = run(n, use_graspnet=False)
    print(f"物体中心抓取: {s_center:.0%}")
    s_gn = run(n, use_graspnet=True)
    print(f"GraspNet 抓取: {s_gn:.0%}")
    print(f"\n对比: center={s_center:.0%} vs graspnet={s_gn:.0%}")
