"""割算子（附录 B 表 2）：机制 + 证据 → Θ 不等式或几何指令。

每个割是纯函数：(mechanism, evidence, context) → Cut 列表。
Cut 两类：
  theta_cut  —— 写进 Θ 后验的不等式（学习发生在这里）
  directive  —— 几何/路径指令（无 Θ，直接改候选点/路径）

reflection.py 的 if-elif 调参方向将逐条迁移到这张表；新增失败
模式 = 新一行，不写新业务代码。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .discriminators import Mechanism
from .observables import Evidence
from .posterior import body_mass


@dataclass
class Cut:
    kind: str                 # "theta_cut" | "directive"
    name: str = ""            # theta 名（kind=theta_cut）
    lo: Optional[float] = None
    hi: Optional[float] = None
    conf: float = 1.0
    directive: str = ""       # kind=directive 时的指令文本
    basis: str = ""           # 实证出处（表行五列标准的"实证来源"）


G = 9.81


def _budget_cut(ev: Evidence, **_) -> Optional[Cut]:
    """BUDGET_SHORT：timeout ≥ 距离/实测速率 × margin（spatial:4 实证：
    需 ~280 步 @0.3mm/步，90 步硬预算必 stall）。"""
    if ev.goal_z is None or not ev.z_trace:
        return None
    rate = ev.z_rate_median()
    if rate <= 0:
        return None
    dist = ev.z_trace[0] - ev.goal_z
    need = int(dist / rate * 1.3) + 1  # 裕度 1.3（spatial:4 实证）
    return Cut(kind="theta_cut", name="timeout_need", lo=float(need),
               conf=0.9, basis="spatial:4 慢型 stall 实证")


def _friction_cut(ev: Evidence, env=None, body: Optional[str] = None,
                  f_grip: Optional[float] = None, **_) -> Optional[Cut]:
    """FRICTION_SLIP：μ < m·g/F_grip（spatial:9/goal:2 place slipped
    的直接读法）。F_grip 缺省时用证据里的最大接触力。"""
    body = body or ev.body
    f = f_grip or (max(ev.f_trace) if ev.f_trace else 0.0)
    if not body or f <= 0:
        return None
    m = body_mass(env, body) if env is not None else None
    if not m or m <= 0:
        return None
    mu_hi = m * G / f
    return Cut(kind="theta_cut", name="mu", hi=mu_hi, conf=0.85,
               basis=f"摩擦锥: μ·m 支撑需 F≥m·g, 实测 F={f:.2f}N m={m:.3f}kg")


def _reach_limit_cut(ev: Evidence, z_goal: Optional[float] = None,
                     **_) -> Optional[Cut]:
    """REACH_LIMIT：实测短缩写库——band 推导的输入（探针的免费副产品）。"""
    if not ev.z_trace:
        return None
    zg = ev.goal_z if z_goal is None else z_goal
    if zg is None:
        return None
    short = ev.z_trace[-1] - zg
    if short < 0:
        return None
    return Cut(kind="theta_cut", name="reach_shortfall", lo=0.0,
               hi=short * 1.5, conf=0.9,
               basis="F=0 自由空间停滞=OSC 可达极限（ik_servo reach_limit）")


_CUT_TABLE = {
    Mechanism.BUDGET_SHORT: [_budget_cut],
    Mechanism.FRICTION_SLIP: [_friction_cut],
    Mechanism.REACH_LIMIT: [_reach_limit_cut],
    # CONTACT_BLOCKED / NO_GRIP_AIR / GEOMETRY_SQUEEZE / IK_UNREACHABLE：
    # 几何修正（无 Θ），由 directive 表达，消费方是候选生成/路径层。
    Mechanism.CONTACT_BLOCKED: [],
    Mechanism.NO_GRIP_AIR: [],
    Mechanism.GEOMETRY_SQUEEZE: [],
    Mechanism.IK_UNREACHABLE: [],
    Mechanism.SUPPORT_OVERFLOW: [],
    Mechanism.CLEARANCE_VIOLATION: [],
    Mechanism.UNKNOWN: [],
}

_DIRECTIVES = {
    Mechanism.CONTACT_BLOCKED:
        "抓取点移出障碍/沿口 AABB⊕margin，或换 straddle 方向",
    Mechanism.NO_GRIP_AIR:
        "候选点不在力封闭集：换候选点/调 straddle 偏置（几何推导，无 Θ）",
    Mechanism.GEOMETRY_SQUEEZE:
        "放置点修正：质心投影入支撑多边形 / 释放时机提前",
    Mechanism.IK_UNREACHABLE:
        "目标 ∉ 工作空间投影：换路径点（approach 绕行）",
    Mechanism.SUPPORT_OVERFLOW:
        "放置点向支撑多边形内收",
    Mechanism.CLEARANCE_VIOLATION:
        "raise clearance margin / 绕障路径",
}


def cut_for(mechanism: Mechanism, ev: Evidence, *,
            env=None, body: Optional[str] = None,
            f_grip: Optional[float] = None,
            z_goal: Optional[float] = None) -> List[Cut]:
    """机制 → 割列表（theta_cut 和 directive 的并）。"""
    out: List[Cut] = []
    for fn in _CUT_TABLE.get(mechanism, []):
        try:
            c = fn(ev, env=env, body=body, f_grip=f_grip, z_goal=z_goal)
        except Exception:
            c = None
        if c is not None:
            out.append(c)
    d = _DIRECTIVES.get(mechanism)
    if d:
        out.append(Cut(kind="directive", directive=d, basis=str(mechanism.value)))
    return out
