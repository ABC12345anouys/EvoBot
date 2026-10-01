"""env_utils：环境创建、缓存与边界工具（执行器共用）。"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

_ENV_CACHE: Dict[str, Any] = {}


def get_env(env_id: str, robot: str, controller: str = None):
    """按 env_id 创建并缓存环境（同一个环境复用，避免反复加载模型文件）。

    当前支持 LIBERO 后端，env_id 形如 ``libero:<suite>:<idx>``。
    """
    from ..envs.libero_adapter import LiberoEnvAdapter, parse_env_id
    spec = parse_env_id(env_id)
    if spec is None:
        raise ValueError(
            f"无法识别的 env_id: {env_id!r}；当前支持 'libero:<suite>:<idx>'")
    key = f"libero|{spec[0]}|{spec[1]}"
    if key not in _ENV_CACHE:
        env = LiberoEnvAdapter(spec[0], spec[1])
        env._darwin_kind = "libero"
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
