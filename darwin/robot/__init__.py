"""RobotBackend 抽象层:机型/接口无关的运动后端。

导出:
- RobotBackend / RobotProfile / ArmConfig / make_backend / register_backend
- MujocoRobopalBackend(mujoco+robopal+mplib 默认实现)

用法:
    from darwin.robot import make_backend
    env.backend = make_backend("diana_med", env)  # 在 env 构造时注入
    # Skill 内:backend = env.backend; backend.plan_path(goal_pose, actor=...)
"""
from .backend import (
    RobotBackend, RobotProfile, ArmConfig,
    make_backend, register_backend, PROFILES_DIR,
)
from .mujoco_robopal import MujocoRobopalBackend  # 注册 mujoco_robopal backend_type

__all__ = [
    "RobotBackend", "RobotProfile", "ArmConfig",
    "make_backend", "register_backend", "MujocoRobopalBackend", "PROFILES_DIR",
]
