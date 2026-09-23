"""BimanualReach-v0 纯控制器验证：双臂各自速度跟踪目标 site。

- CARTIK reference='base'：期望位置在各自臂的基座系（agent0 基座 world(1.2,0,0.3) 且旋转180°）。
- 环境默认 pos 边界（base z>=0.2, x<=0.65）与其自身目标采样域矛盾（目标 world z 0.46~0.66,
  agent1 目标 world x 低至 0.15），脚本级放宽边界使其可达成（不改 robopal 源码）。
- 诚实判据：两臂 grip site 同时距各自目标 site < 2cm（env 的 is_success 被官方注释掉，
  按 compute_rewards 的 dist<=0.02 口径），不做出生点过滤。
运行：MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 python scripts/check_bimanual_reach.py
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

OVERRIDE_MAX = {"agent0": np.array([0.80, 0.30, 0.70]), "agent1": np.array([0.80, 0.30, 0.70])}
OVERRIDE_MIN = {"agent0": np.array([0.10, -0.30, 0.10]), "agent1": np.array([0.10, -0.30, 0.10])}


def base_frame(env, agent):
    bid = mujoco.mj_name2id(env.mj_model, mujoco.mjtObj.mjOBJ_BODY, env.robot.base_link_name[agent])
    return env.mj_data.xpos[bid].copy(), env.mj_data.xmat[bid].reshape(3, 3).copy()


def make_to_base(env, agent):
    bp, bm = base_frame(env, agent)
    return lambda p_world: bm.T @ (np.asarray(p_world) - bp)


def reach_episode(env, max_steps=300, verbose=False):
    env.reset()
    # 修复 robopal reset bug：desired_positions 残留上一局值
    env.desired_positions = {ag: env.init_pos[ag].copy() for ag in env.agents}
    env.pos_max_bound = {ag: OVERRIDE_MAX[ag].copy() for ag in env.agents}
    env.pos_min_bound = {ag: OVERRIDE_MIN[ag].copy() for ag in env.agents}
    goals0 = {ag: env.get_site_pos("goal_site%s" % ag[-1]).copy() for ag in env.agents}
    to_base = {ag: make_to_base(env, ag) for ag in env.agents}
    if verbose:
        for ag in env.agents:
            print("    %s goal(world)=%s goal(base)=%s" % (
                ag, goals0[ag].round(3), to_base[ag](goals0[ag]).round(3)), flush=True)
    t, ok_steps, hold = 0, 0, 0
    dists = {ag: 9.9 for ag in env.agents}
    while t < max_steps:
        t += 1
        acts = {}
        for ag in env.agents:
            grip_b = to_base[ag](env.get_site_pos("%s_grip_site" % ag[-1]))
            err = to_base[ag](goals0[ag]) - grip_b
            acts[ag] = np.append(np.clip(3.0 * err, -1.0, 1.0), 0.0)
        env.step(acts)
        dists = {ag: float(np.linalg.norm(env.get_site_pos("goal_site%s" % ag[-1])
                                          - env.get_site_pos("%s_grip_site" % ag[-1])))
                 for ag in env.agents}
        if all(d < 0.02 for d in dists.values()):
            ok_steps += 1
            hold = max(hold, ok_steps)
            if ok_steps >= 10:  # 连续 0.5s 保持即认定达成
                return True, t, dists, hold
        else:
            ok_steps = 0
    return False, t, dists, hold


def main():
    env = robopal.make("BimanualReach-v0", robot="DualDianaReach",
                       render_mode=None, control_freq=20)
    results = []
    for ep in range(3):
        ok, t, dists, hold = reach_episode(env, verbose=True)
        results.append(ok)
        print("[BimanualReach] ep%d: %s steps=%d hold=%d dists=%s" % (
            ep + 1, "SUCCESS" if ok else "FAIL", t, hold,
            {k: round(v, 4) for k, v in dists.items()}), flush=True)
    env.close()
    print("成功率: %d/%d" % (sum(results), len(results)), flush=True)


if __name__ == "__main__":
    main()
