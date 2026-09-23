"""验证 RobotBackend 抽象层:motion 原语不退化。

1. InsertEnv 构造时注入 backend(MujocoRobopalBackend + diana_med profile)
2. collision_check home:collision_free == True
3. path_plan 近距(z+2cm):err < 1e-4(原 9e-06)
4. path_plan 远距(14+ 步轨迹):err < 5mm(原 2.4mm)

运行:
    CUDA_VISIBLE_DEVICES=0 python scripts/test_motion_backend.py
"""
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from darwin.skills import build_registry
from darwin.agents.runner import _get_env


def main():
    env = _get_env("InsertEnv", "DianaTripleStack")
    env.reset()
    print("backend:", type(env.backend).__name__, "| profile:", env.backend.profile.manipulator)
    assert env.backend is not None, "backend 未注入"

    registry = build_registry()
    # 前置描述门控
    registry.get_skill_description("collision_check")
    registry.get_skill_description("path_plan")

    # ---- 1. collision_check home(无碰撞)----
    r = registry.execute("collision_check", env=env, actor="agent0")
    print(f"[collision_check home] success={r.get('success')} collision_free={r.get('collision_free')} reason={r.get('reason')}")
    assert r["success"], f"home 应无碰撞, got {r}"

    # ---- 2. path_plan 近距(z+2cm)----
    eef_pos, eef_quat = env.backend.get_eef_pose("agent0")
    near_goal = list(eef_pos + np.array([0, 0, 0.02])) + list(eef_quat)
    r = registry.execute("path_plan", env=env, goal_pose=near_goal, actor="agent0")
    print(f"[path_plan near z+2cm] success={r.get('success')} reason={r.get('reason')} "
          f"final_err={r.get('final_err'):.2e} traj_len={r.get('trajectory_len')}")
    assert r["success"], f"近距 path_plan 失败: {r}"
    assert r["final_err"] < 1e-4, f"近距 err 过大: {r['final_err']}"

    # ---- 3. path_plan 远距(到 home 上方高位置,多步轨迹)----
    env.reset()
    eef_pos, eef_quat = env.backend.get_eef_pose("agent0")
    far_goal = list(eef_pos + np.array([0.15, 0.10, 0.20])) + list(eef_quat)
    r = registry.execute("path_plan", env=env, goal_pose=far_goal, actor="agent0",
                         planning_time=2.0, skip=2, timeout=300)
    print(f"[path_plan far 0.15/0.10/0.20] success={r.get('success')} reason={r.get('reason')} "
          f"final_err={r.get('final_err')} executed={r.get('executed')} traj_len={r.get('trajectory_len')}")
    if not r["success"]:
        print(f"  [远距失败详情] {r}")
    assert r["success"], f"远距 path_plan 失败: {r}"
    assert r["final_err"] < 5e-3, f"远距 err 过大: {r['final_err']}"

    env.close()
    print("\n全部通过:RobotBackend 抽象层 motion 原语行为不退化")


if __name__ == "__main__":
    main()
