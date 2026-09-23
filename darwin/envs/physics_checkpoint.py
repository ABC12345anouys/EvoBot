"""物理状态快照/恢复的跨 env 统一入口（失败回退 N 步续跑用）。

- env 自带 save_state/restore_state（如 LiberoEnvAdapter，走 robosuite
  MjSimState 扁平向量，含 udd_state）→ 优先委托；
- 否则用 env.mj_model/mj_data 直拷 time/qpos/qvel/act 恢复 + mj_forward
  （robopal 路径；CARTIK 每步按当前位姿重算、无跨步积分，恢复后正常）。

token 是不透明对象（np.ndarray 或 dict），调用方只负责存取，不解释内容。
"""
from __future__ import annotations

from typing import Any

import numpy as np


def save_physics(env) -> Any:
    fn = getattr(env, "save_state", None)
    if callable(fn):
        return fn()
    d = env.mj_data
    na = getattr(d, "na", getattr(d, "nA", 0))
    act = np.asarray(d.act)
    return {
        "time": float(d.time),
        "qpos": np.array(d.qpos, dtype=np.float64).copy(),
        "qvel": np.array(d.qvel, dtype=np.float64).copy(),
        "act": np.array(act, dtype=np.float64).copy() if na else None,
    }


def restore_physics(env, token: Any) -> None:
    fn = getattr(env, "restore_state", None)
    if callable(fn):
        fn(token)
        return
    import mujoco
    d, m = env.mj_data, env.mj_model
    if isinstance(token, dict):
        d.time = float(token["time"])
        np.copyto(d.qpos, np.asarray(token["qpos"], dtype=np.float64)[:m.nq])
        np.copyto(d.qvel, np.asarray(token["qvel"], dtype=np.float64)[:m.nv])
        na = getattr(d, "na", getattr(d, "nA", 0))
        if na and token.get("act") is not None:
            np.copyto(d.act, np.asarray(token["act"], dtype=np.float64)[:na])
    else:
        # 扁平 MjSimState 兜底（与 robosuite 同布局时可用；通常 env 已自带
        # restore_state，走不到这里）
        flat = np.asarray(token, dtype=np.float64)
        np.copyto(d.qpos, flat[:m.nq])
        np.copyto(d.qvel, flat[m.nq:m.nq + m.nv])
    mujoco.mj_forward(m, d)
