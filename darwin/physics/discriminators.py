"""判别子（附录 B 表 3）：遥测签名 → 机制二分。

每个判别子是纯函数：Evidence + 少量上下文 → Mechanism。
机制枚举即"假设空间的原子"——重试采样器（P0-5）与割算子（cuts.py）
都以机制为接口单位，新增失败模式 = 新增一个 Mechanism + 一个判别
分支 + 一行割，不写新 if-else 业务代码。
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

import numpy as np

from .observables import Evidence, F_FREE_N, RATE_EPS


class Mechanism(Enum):
    # descend/above 族
    REACH_LIMIT = "reach_limit"        # 自由空间停滞：OSC 可达极限，可接受
    CONTACT_BLOCKED = "contact_blocked"  # 几何接触阻挡：真问题，修几何
    BUDGET_SHORT = "budget_short"      # 慢型进展：预算不足，加时有效
    IK_UNREACHABLE = "ik_unreachable"  # 目标点不在工作空间/被障碍楔止
    # 抓取/夹持族
    NO_GRIP_AIR = "no_grip_air"        # 夹空：构型不在力封闭集（修候选点）
    FRICTION_SLIP = "friction_slip"    # 摩擦锥违背：该滑（加力/降速/减加速度）
    GEOMETRY_SQUEEZE = "geometry_squeeze"  # 被几何挤飞：修放置点/路径
    # place 族
    SUPPORT_OVERFLOW = "support_overflow"  # 质心投影越支撑多边形
    # 其他
    CLEARANCE_VIOLATION = "clearance_violation"
    UNKNOWN = "unknown"


def classify_stall(ev: Evidence, xy_tol: float = 0.015,
                   z_goal: Optional[float] = None,
                   reach_limit_band: float = 0.035) -> Mechanism:
    """z 停滞的三分（表 3 第 1/2 行）——同为 "stall"，对策相反，
    不判别就会像 r9 那样把 5 次重试全烧错（spatial:4 vs spatial:2 实证）。

    输入停滞已由调用方双窗口确认；本函数负责机制分类。
    """
    if z_goal is None:
        z_goal = ev.goal_z
    above_goal = (ev.z_trace[-1] - z_goal) if (ev.z_trace and z_goal is not None) else None

    # 第 1 刀：接触力。F≈0 → 指尖在自由空间，停滞不可能是几何阻挡。
    if ev.f_at_end < F_FREE_N:
        # 第 2 刀：进展速率。匀速慢 = 预算不足；真 0 进展 = 可达极限/楔止。
        if ev.z_rate_median() > RATE_EPS:
            return Mechanism.BUDGET_SHORT
        if above_goal is not None and 0.0 <= above_goal <= reach_limit_band:
            return Mechanism.REACH_LIMIT
        return Mechanism.IK_UNREACHABLE
    # F>0：真接触。z 停在被挡高度 → 几何阻挡（沿口/障碍）。
    if above_goal is not None and above_goal > reach_limit_band:
        return Mechanism.IK_UNREACHABLE
    return Mechanism.CONTACT_BLOCKED


def classify_no_grip(ev: Evidence) -> Mechanism:
    """lift_no_grip / 夹空的第一刀（表 3 第 3 行）。"""
    if ev.f_at_end < F_FREE_N and ev.f_trace and max(ev.f_trace) < F_FREE_N:
        return Mechanism.NO_GRIP_AIR
    return Mechanism.FRICTION_SLIP


# 弹射判距：滑移位移超过该值判"挤压飞"（被几何弹出，方向确定、单次
# 位移大）；小位移才是摩擦滑（沿接触面缓慢滑出）。goal:2 瓶类沿爪下滑
# 位移 <1cm vs 被沿口弹出 >5cm，量级差足以二分。
EJECT_DIST_M = 0.05


def classify_slip(ev: Evidence,
                  slip_direction: Optional[tuple] = None) -> Mechanism:
    """滑移的机理二分（表 3 第 4 行）——spatial:9/goal:2 place slipped
    切这刀后，挤压飞走几何修正（换放置点/路径），摩擦滑走物理修正
    （降速/压稳），不再一刀切 friction_slip 然后 place_vcap 盲试。

    slip_direction：物体滑移位移向量 (dx, dy)（世界系水平分量，未归一
    化——模长即滑移距离）；None 时退化为摩擦滑（保守：物理修正无害，
    几何修正在证据不足时可能误导）。
    """
    if slip_direction is not None:
        d = float(np.sqrt(float(slip_direction[0]) ** 2
                          + float(slip_direction[1]) ** 2))
        if d >= EJECT_DIST_M:
            return Mechanism.GEOMETRY_SQUEEZE
        return Mechanism.FRICTION_SLIP
    return Mechanism.FRICTION_SLIP


def classify_place_timeout(ev: Evidence,
                           obj_z_follows: Optional[bool] = None) -> Mechanism:
    """place_timeout 的机制分类（表 3 第 3/6 行组合）。"""
    if obj_z_follows is False:
        return Mechanism.NO_GRIP_AIR
    if ev.clearance_min < 0.005:
        return Mechanism.CLEARANCE_VIOLATION
    return Mechanism.GEOMETRY_SQUEEZE


def discriminate(reason: str, ev: Evidence, **ctx) -> Mechanism:
    """reason（skill 失败因由）+ 证据 → 机制的统一分发。

    skill 层失败因由保留（日志/反思兼容），机制分类在其上加一层——
    这就是 §16.2 "判别子比约束稀缺" 的落点：同一 reason（如
    ik_descend_stalled）按证据切到不同机制。
    """
    r = reason or ""
    if "xy_drift" in r:
        return Mechanism.IK_UNREACHABLE
    if "stalled" in r or "stall" in r:
        return classify_stall(ev, **{k: v for k, v in ctx.items()
                                     if k in ("xy_tol", "z_goal", "reach_limit_band")})
    if "no_grip" in r or "grip_fail" in r:
        return classify_no_grip(ev)
    if "slipped" in r or "slip" in r:
        return classify_slip(ev, slip_direction=ctx.get("slip_direction"))
    if "timeout" in r:
        return classify_place_timeout(ev, obj_z_follows=ctx.get("obj_z_follows"))
    if "collision" in r:
        return Mechanism.CLEARANCE_VIOLATION
    return Mechanism.UNKNOWN
