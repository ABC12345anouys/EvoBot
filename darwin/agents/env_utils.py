"""env_utils：环境创建/缓存/边界工具（runner.py 与 runner_dynamic.py 共用）。

抽取自原 runner_dynamic.py / runner.py 的重复代码，消除两处维护。
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import numpy as np

_ENV_CACHE: Dict[str, Any] = {}


def get_env(env_id: str, robot: str, controller: str = None):
    """环境缓存（同一 env 复用，避免反复加载 MJCF）。"""
    # LIBERO 后端：env_id="libero:<suite>:<idx>"，走 robosuite 适配器
    from ..envs.libero_adapter import parse_env_id
    libero_spec = parse_env_id(env_id)
    if libero_spec is not None:
        from ..envs.libero_adapter import LiberoEnvAdapter
        key = f"libero|{libero_spec[0]}|{libero_spec[1]}"
        if key not in _ENV_CACHE:
            env = LiberoEnvAdapter(libero_spec[0], libero_spec[1])
            env._darwin_kind = "libero"
            _ENV_CACHE[key] = env
        return _ENV_CACHE[key]

    import robopal
    from robopal.envs.base import MujocoEnv
    MujocoEnv.close = lambda self: self.renderer.close()
    key = f"{env_id}|{robot}|{controller}"
    if key not in _ENV_CACHE:
        if env_id == "InsertEnv":
            from ..envs import InsertEnv
            from robopal.robots.diana_med import DianaTripleStack
            robot_cls = {"DianaTripleStack": DianaTripleStack}.get(robot, DianaTripleStack)
            env = InsertEnv(robot=robot_cls, render_mode=None, control_freq=20)
            env._darwin_kind = "insert"
        else:
            env = robopal.make(env_id, robot=robot, render_mode=None, control_freq=20)
            env._darwin_kind = "robopal"
        _PROFILE_MAP = {"DianaTripleStack": "diana_med"}
        profile_name = _PROFILE_MAP.get(robot)
        if profile_name is not None:
            from ..robot import make_backend
            env.backend = make_backend(profile_name=profile_name, env=env)
        _ENV_CACHE[key] = env
    return _ENV_CACHE[key]


class StepRecorder:
    def __new__(cls, env, path, bimanual: bool, fps: int = 20,
                width: int = 640, height: int = 480):
        from ..utils.recorder import StepRecorder as _StepRecorder
        return _StepRecorder(env, path, bimanual=bimanual, fps=fps,
                             width=width, height=height)


def apply_bounds(env, bounds: Optional[Dict[str, Dict[str, list]]]) -> None:
    if not bounds:
        return
    if not isinstance(getattr(env, "pos_max_bound", None), dict):
        env.pos_max_bound = {}
    if not isinstance(getattr(env, "pos_min_bound", None), dict):
        env.pos_min_bound = {}
    for ag, b in bounds.items():
        env.pos_max_bound[ag] = np.array(b["max"], float)
        env.pos_min_bound[ag] = np.array(b["min"], float)


# 向后兼容别名
_get_env = get_env
_apply_bounds = apply_bounds
