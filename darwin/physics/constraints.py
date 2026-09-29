"""L1 约束判据（附录 B 表 1）：机器人无关的物理可行检查。

判据全部用物体中心坐标系写（附录 D 规则 1），输入是几何特征与
Θ 后验，输出是 (满足?, 违背部, 裕度)。运行时监视器（P0-3）每 N 步
对 active constraints 求值——本模块是监视器的判据库。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

# 名义摩擦系数下限（保守值）。精确值应在 Θ 后验 mu 域（辨识主体），
# 此处只供"力封闭裕度"的 advisory 判读（P0-3 完整版从后验读）。
MU_NOMINAL = 0.5


def friction_cone(f_t: float, f_n: float, mu: float) -> Tuple[bool, float]:
    """库仑摩擦锥 ‖f_t‖ ≤ μ·f_n。返回 (满足, 裕度=μ·f_n−‖f_t‖)。

    μ 取 Θ 后验下界（保守：宁可高估所需摩擦）。裕度<0 即违背。"""
    if f_n <= 0:
        return False, -abs(f_t)
    return (f_t <= mu * f_n), (mu * f_n - f_t)


def support_polygon(body_xy, poly: np.ndarray) -> Tuple[bool, float]:
    """质心（水平）投影 ∈ 支撑多边形（凸，顶点序无关）。

    poly: (k,2) 凸多边形顶点。返回 (在内, 到边最小距离)。
    简化实现：点在凸多边形内 = 对所有边同侧（按面积符号）。
    """
    p = np.asarray(body_xy, float)[:2]
    v = np.asarray(poly, float)
    if len(v) < 3:
        return True, 0.0
    signs = []
    dists = []
    k = len(v)
    for i in range(k):
        a, b = v[i], v[(i + 1) % k]
        e = b - a
        n = np.array([-e[1], e[0]])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = float(np.dot(p - a, n))
        signs.append(np.sign(d))
        dists.append(abs(d))
    inside = (len(set(int(s) for s in signs if s != 0)) <= 1)
    return inside, (min(dists) if dists else 0.0)


def finger_span_ok(obj_half_width: float, finger_span: float,
                   grip_margin: float = 0.004) -> Tuple[bool, float]:
    """夹爪开口 ≥ 物体宽 + margin（表 1 几何尺寸行）。

    finger_span：两指内侧间距；obj_half_width：物体沿闭合方向的半宽。
    """
    need = 2.0 * obj_half_width + grip_margin
    return (finger_span >= need), (finger_span - need)


def clearance_ok(clearance: float, d_min: float) -> Tuple[bool, float]:
    """signed distance ≥ d_min（碰撞约束的标量版，表 1 运动学/碰撞行）。"""
    return (clearance >= d_min), (clearance - d_min)


def force_closure_lite(f_contact: float, m: float,
                       mu: float, safety: float = 1.5) -> Tuple[bool, float]:
    """力封闭 lite：接触力足以支撑重力（2μ·f ≥ s·m·g，双指各半）。

    完整 grasp wrench space 检验留给 P1（需要接触点几何）；lite 版
    覆盖 LIBERO 抓放的主流情形（竖直搬运）。返回 (满足, 裕度 N)。
    """
    need = safety * m * 9.81
    return (2.0 * mu * f_contact >= need), (2.0 * mu * f_contact - need)


# ---- 扩展判据（r12 失败簇的判据化；每条对准一个主导机制的死因）----

def reach_feasible(z_goal: float, z_reach_limit: float,
                   margin: float = 0.01) -> Tuple[bool, float]:
    """可达判据：目标深度在可达集内（L1 几何 × Θ reach_shortfall）。

    z_reach_limit = OSC 深度极限 − 实测短缩（后验辨识，band 推导的副产品）。
    违背 = ik_unreachable 的病理解：换更高 hover / 更陡接近角 / 换抓取点，
    而不是重试同一参数（goal:3/4/5 共 118 次 ik_unreachable 空转的教训——
    参数不是可达性的解）。返回 (满足, 深度裕度 m)。
    """
    return (z_goal >= z_reach_limit - margin), (z_goal - z_reach_limit + margin)


def grip_force_need_met(f_hold: float, mu: float, m: float,
                        safety: float = 1.5) -> Tuple[bool, float]:
    """夹持力需求判据（derives.grip_force_need 的判据形态）：实测夹持力 ≥
    需求。close_gripper 后立即求值——把 lift_no_grip 从"抬完才知道"提前到
    "闭合就知道"。

    违背且夹爪行程已饱和（f_hold 是最大可设法向力）→ 调参 exhausted，
    必须升级换候选/换抓取点（goal:5 24 次 lift_no_grip：stop_above/k
    给不了更大法向力，参数方向存在但已到顶）。返回 (满足, 力裕度 N)。
    """
    need = safety * m * 9.81 / max(mu, 1e-6)
    return (f_hold >= need), (f_hold - need)


def insertion_clearance_ok(clearance: float,
                           tol: float) -> Tuple[bool, float]:
    """peg-in-hole 间隙判据：装配间隙 vs 感知/定位容差。

    间隙 < tol（定位噪声量级）时直插必失败（goal:2 瓶架 marginals）——
    判据触发即路由到 transfer_pose 螺旋搜索/倒角捕获（技能已在，判据只
    做路由，不学新东西）。返回 (满足, 间隙−tol)。
    """
    return (clearance >= tol), (clearance - tol)


def approach_corridor_ok(p0, p1, clearance_fn, r_finger: float,
                         margin: float = 0.004,
                         samples: int = 9) -> Tuple[bool, float]:
    """接近走廊判据：悬停点→抓取点线段上，指包络圆柱的 clearance 全程
    ≥ margin（L1 几何关系；clearance_fn 是 L2 投影层注入的符号距离）。

    descend xy_drift 的判据化（goal:3/4：下降途中 TCP 被障碍楔偏，xy 漂
    >15mm——端点可达 ≠ 路径可达）。违背 → 规划时抬高 hover / 侧向偏移
    接近点 / 改接近角，在到达之前就绕开。返回 (满足, 全程最小裕度 m)。

    samples 为奇数：保证线段中点被采到（障碍在悬停-抓取正中时偶数采样
    会从中点两侧漏过——本函数自测实证）。
    """
    a = np.asarray(p0, float)
    b = np.asarray(p1, float)
    need = r_finger + margin
    worst = float("inf")
    for s in np.linspace(0.0, 1.0, max(samples, 3)):
        p = a + (b - a) * float(s)
        try:
            c = float(clearance_fn(p))
        except Exception:
            continue
        worst = min(worst, c - need)
    if worst == float("inf"):
        return True, 0.0
    return (worst >= 0.0), worst


# ---- 监视器快照（P0-3 的 active constraint 集合）----

def active_set(step: str, **kwargs) -> Dict[str, Dict[str, Any]]:
    """按 step 名给出运行时监视的判据集与当前参数（kwargs 注入观测）。"""
    if step in ("descend", "ik_servo"):
        return {"friction_cone": {}, "clearance_ok": kwargs.get("d_min", 0.0),
                "reach_feasible": {"z_goal": kwargs.get("z_goal"),
                                   "z_reach_limit": kwargs.get("z_reach_limit")},
                "approach_corridor_ok": {"p0": kwargs.get("p0"),
                                         "p1": kwargs.get("p1"),
                                         "clearance_fn": kwargs.get("clearance_fn"),
                                         "r_finger": kwargs.get("r_finger", 0.008)}}
    if step in ("carry", "lift"):
        return {"force_closure_lite": {"m": kwargs.get("m", 0.0),
                                       "mu": kwargs.get("mu", 0.5)}}
    if step == "close_gripper":
        return {"grip_force_need_met": {"f_hold": kwargs.get("f_hold", 0.0),
                                        "mu": kwargs.get("mu", MU_NOMINAL),
                                        "m": kwargs.get("m", 0.0)}}
    if step == "place":
        return {"support_polygon": {"poly": kwargs.get("poly")},
                "clearance_ok": kwargs.get("d_min", 0.0),
                "insertion_clearance_ok": {"clearance": kwargs.get("clearance"),
                                           "tol": kwargs.get("tol", 0.003)}}
    return {}
