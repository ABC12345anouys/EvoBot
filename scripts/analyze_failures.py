"""分析 BimanualPickAndPlace 失败模式，为自进化提供数据。

记录每个 episode 的：初始位置、GraspNet 抓取点、失败阶段、状态。
"""
import os
import sys
import json
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


def episode_debug(env, max_steps=400, max_attempts=3):
    """带详细调试信息的 episode。"""
    from darwin.skills.perception.grasp import grasp_from_env
    env.reset()
    env.desired_positions = {ag: env.init_pos[ag].copy() for ag in env.agents}
    env.pos_max_bound = {"agent0": np.array([0.65, 0.65, 0.65]), "agent1": np.array([0.65, 0.40, 0.65])}
    env.pos_min_bound = {"agent0": np.array([-0.10, -0.30, 0.02]), "agent1": np.array([0.25, -0.10, 0.02])}

    init_pos = env.get_body_pos("green_block").copy()
    grasp_pt = None
    fail_phase = None
    fail_state = None
    t, ok_steps = 0, 0

    for attempt in range(max_attempts):
        phase, t2, hold = "above", t, 0
        while t < max_steps:
            t += 1
            end = env.get_site_pos(GripSite)
            rg = grasp_from_env(env, "green_block")
            if rg.get("success"):
                grasp_pt = np.array(rg["position"], float)
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
                    fail_phase = "lift"
                    fail_state = {
                        "end_z": float(end[2]),
                        "block_z": float(env.get_body_pos("green_block")[2]),
                    }
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
                    return {"success": True, "steps": t, "init_pos": init_pos.tolist(),
                            "grasp_pt": grasp_pt.tolist() if grasp_pt is not None else None,
                            "goal": goal.tolist()}
            else:
                ok_steps = 0
            if hold > 30:
                fail_phase = "place_hold_timeout"
                fail_state = {"d": d, "end": end.tolist(), "goal": goal.tolist()}
                break
        if fail_phase and attempt < max_attempts - 1:
            fail_phase = None  # 重试
            continue
        if fail_phase:
            break

    return {"success": False, "steps": t, "init_pos": init_pos.tolist(),
            "grasp_pt": grasp_pt.tolist() if grasp_pt is not None else None,
            "fail_phase": fail_phase, "fail_state": fail_state,
            "goal": env.get_site_pos("goal_site").tolist()}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    env = robopal.make("BimanualPickAndPlace-v0", robot="DualPandaPickAndPlace", render_mode=None, control_freq=20)
    results = []
    for i in range(n):
        r = episode_debug(env)
        tag = "OK" if r["success"] else f"FAIL({r.get('fail_phase','?')})"
        print(f"ep{i}: {tag} steps={r['steps']}")
        if not r["success"]:
            print(f"  init={[round(x,3) for x in r['init_pos']]}")
            print(f"  grasp={[round(x,3) for x in r['grasp_pt']] if r['grasp_pt'] else None}")
            print(f"  state={r.get('fail_state')}")
        results.append(r)

    n_ok = sum(r["success"] for r in results)
    print(f"\n成功率: {n_ok}/{len(results)} = {n_ok/len(results):.0%}")
    # 统计失败阶段
    fail_phases = {}
    for r in results:
        if not r["success"]:
            fp = r.get("fail_phase", "unknown")
            fail_phases[fp] = fail_phases.get(fp, 0) + 1
    print(f"失败阶段分布: {fail_phases}")
    env.close()

    # 保存到 json 供自进化使用
    out = os.path.join(os.path.dirname(__file__), "..", "data", "fail_analysis.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"详细结果保存到: {out}")
