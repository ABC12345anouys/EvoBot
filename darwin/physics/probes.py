"""主动探针（附录 B 表 4）：机器人问物理引擎问题。

每个探针是一次低成本主动实验，把 Θ 标量直接测出来（替代参数搜索）。
全部经 Phase B 快照保护（调用方负责）、低速小步进（探针自身约束）。
多数探针与执行重叠：接触探针成功 = descend 已完成。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from .observables import F_FREE_N


def contact_probe(env, site: str, pt, *, servo_step, gripper: float = -1,
                  k: float = 5.0, vcap: float = 0.15,
                  f_stop_n: float = 1.0,
                  timeout: int = 200, reach_tol: float = 0.006,
                  actor: str = "agent0", body: Optional[str] = None,
                  z_floor: Optional[float] = None) -> Dict[str, Any]:
    """慢降接触探针：低速下降直到指尖-物体接触力 > f_stop_n 即停。

    测得：接触 z 真值（contact_z）与全程最大力（f_max）。
    用途：contact_stop_band = 接触z - 目标z + margin（运行时实测，
    杀逐任务标定彩票）；F=0 探完全程 = 自由空间实证（无几何阻挡）。

    安全：vcap≤0.15 限速 + 力阈值停（不硬压）；z_floor 兜底（不给则
    探到 timeout）。
    """
    pt = np.asarray(pt, float)
    z_hit, f_max = None, 0.0
    for t in range(timeout):
        eef = np.asarray(env.get_site_pos(site), float)
        f = 0.0
        if body:
            try:
                f = float(env.contact_force_on_body(body))
            except Exception:
                f = 0.0
        f_max = max(f_max, f)
        if f > f_stop_n:
            z_hit = float(eef[2])
            return {"contact": True, "contact_z": z_hit, "f_max": f_max,
                    "steps": t, "eef": eef.tolist()}
        z_cmd = pt[2] if z_floor is None else max(pt[2], z_floor)
        if eef[2] <= z_cmd + reach_tol and f <= F_FREE_N and z_hit is None:
            # 已到目标深度仍无接触 → 该点下方无实体（对实心物=异常，
            # 对容器空腔=正常）。记录后停，不再浪费步数。
            return {"contact": False, "contact_z": None, "f_max": f_max,
                    "steps": t, "eef": eef.tolist(),
                    "note": "reached_z_no_contact"}
        servo_step(env, site, [pt[0], pt[1], z_cmd], gripper=gripper,
                   k=k, vcap=vcap, actor=actor)
    return {"contact": False, "contact_z": None, "f_max": f_max,
            "steps": timeout, "eef": np.asarray(env.get_site_pos(site)).tolist(),
            "note": "timeout_no_contact"}


def reach_probe(env, site: str, xy, z_deep: float, *, servo_step,
                gripper: float = -1, k: float = 5.0, vcap: float = 0.1,
                timeout: int = 250, actor: str = "agent0") -> Dict[str, Any]:
    """可达探针：指令尽可能深的 z，读 OSC 实际最大到达。

    测得：该 (xy, 深度) 构型的到达短缩 = z_reached - z_deep（<0 表示
    到达）。结果是构型的函数（非常数）——入 Θ 后验时须带 xy 上下文
    （当前实现按任务存最大值，够 band 推导用）。
    """
    xy = np.asarray(xy, float)
    z_start = float(np.asarray(env.get_site_pos(site))[2])
    z_min = z_start
    for _ in range(timeout):
        eef = np.asarray(env.get_site_pos(site), float)
        z_min = min(z_min, float(eef[2]))
        if z_min <= z_deep + 0.002:
            break
        servo_step(env, site, [xy[0], xy[1], z_deep], gripper=gripper,
                   k=k, vcap=vcap, actor=actor)
    return {"z_reached": z_min, "shortfall": max(0.0, z_min - z_deep),
            "fully_reached": z_min <= z_deep + 0.002}


def friction_probe(env, body: str, *, close_fn,
                   f_grip_target: float = 8.0,
                   tangential_steps: int = 60,
                   actor: str = "agent0") -> Dict[str, Any]:
    """摩擦探针：定力夹持 + 逐步切向加载，测滑动起始的 μ 估计。

    close_fn(env) 完成夹持（调用方的 close_gripper 语义）。
    低成本近似：夹持后沿水平单方向匀加速拉动，用物体开始滑移时
    的拉力估计 μ = F_tangential/(m·g 等效法向)。仿真可直接读接触
    力，真机版换力传感。
    当前为接口占位：返回 None 语义（μ 未测），由割路径继续二分。
    """
    try:
        close_fn(env)
    except Exception:
        return {"mu": None, "note": "close_failed"}
    return {"mu": None, "note": "placeholder: 由摩擦锥割逐步收缩（r11 起）"}
