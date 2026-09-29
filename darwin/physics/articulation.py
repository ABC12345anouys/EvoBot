"""关节约束类（L1，机器人无关）：一种参数化声明覆盖全部关节情况。

柜门/抽屉/翻盖/旋钮看似多种任务，物理上全是同一个结构：

    关节约束 = { 流形 M, 配置 q, 限位 [q_lo, q_hi], 作用点 h(q),
                 有效力旋量方向 }

差别只在四个参数的值（"很多种情况"是参数实例不是类实例——L1 设计
准则：任务差异全部声明化，代码零分叉）。

参数来源：关节轴/限位/类型在 URDF/MJCF 是显式字段（jnt_type/jnt_axis/
jnt_range/jnt_pos），extract_joint_decl 从模型白拿——零感知、零学习。
感知兜底只有一种场景：真实世界未知家具（轴不可见时从把手运动猜测），
那是 Θ 后验该辨识的量，复用同样的半空间割机制。

判据三条（on_manifold / within_limits / aligned_wrench），其余全部
复用现有判据换输入：把手接近→approach_corridor_ok；拉脱手→friction_cone
（法向换流形法向）；抽屉内放物→insertion_clearance_ok；运动卡死→
clearance_ok 沿 h(q) 轨迹。

失效边界（诚实标注）：已知轴+限位+单链刚体。多链折叠门/柔性门封/
变形物不在覆盖承诺内。

机制映射（落进现有 Mechanism 框架，零新枚举）：
拉脱手=friction_slip；运动卡死=contact_blocked；反向/顶死点=
within_limits 违背（directive：换方向/声明 q 目标）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

# MuJoCo jnt_type 枚举（mjmodel.h）：0=free 1=ball 2=slide 3=hinge
_JNT_REVOLUTE = 3
_JNT_PRISMATIC = 2


@dataclass
class JointDecl:
    """关节声明（L1 参数表的一行）：机器人无关，跨臂共享。"""
    jtype: str                       # "revolute" | "prismatic"
    axis: np.ndarray                 # 关节轴单位向量（世界系）
    point: np.ndarray                # 轴上一点（世界系）
    q_range: Tuple[float, float]     # 限位 [lo, hi]（rad 或 m）
    handle_body: str                 # 作用物体（门/抽屉/盖 body 名）
    qpos_adr: Optional[int] = None   # mj_data.qpos 地址（读当前 q 用）

    def handle_point(self, q: float) -> np.ndarray:
        """作用点运动学 h(q)：q=0 取当前 world 位形（声明时刻）。"""
        p0 = self._h0
        if self.jtype == "revolute":
            # Rodrigues 旋转：h(q) = point + R(axis, q)·(h0 − point)
            v = p0 - self.point
            k = self.axis
            return self.point + (v * np.cos(q)
                                 + np.cross(k, v) * np.sin(q)
                                 + k * np.dot(k, v) * (1 - np.cos(q)))
        return p0 + self.axis * q       # prismatic：沿轴平移

    _h0: np.ndarray = None             # 声明时刻的作用点（世界系），运行时填


def extract_joint_decl(env, body: str) -> Optional[JointDecl]:
    """从 mj_model 读关节声明。零感知零学习——模型字段白拿。

    body：作用物体（柜门/抽屉/盖的 body 名）。该 body 或其父链上找
    第一个有限位关节（slide/hinge）；free/ball 不属于关节约束类
    （free=刚体抓放本体，ball 在 LIBERO 无实例）。
    读不到返回 None（调用方按"非关节体"走刚体路径）。
    """
    try:
        m = env.mj_model
        bid = int(m.body_name2id(body))
        njnt = int(m.body_jntnum[bid])
        if njnt == 0:
            return None
        jadr = int(m.body_jntadr[bid])
        jt = int(m.jnt_type[jadr])
        if jt not in (_JNT_REVOLUTE, _JNT_PRISMATIC):
            return None
        # 轴/轴点：模型局部系 → 世界系（用 body 当前位形变换）
        xpos = np.asarray(m.body_xpos[bid], float)
        xmat = np.asarray(m.body_xmat[bid], float).reshape(3, 3)
        axis = xmat @ np.asarray(m.jnt_axis[jadr], float)
        n = np.linalg.norm(axis)
        axis = axis / n if n > 1e-9 else np.array([0., 0., 1.])
        point = xpos + xmat @ np.asarray(m.jnt_pos[jadr], float)
        q_range = (float(m.jnt_range[jadr][0]), float(m.jnt_range[jadr][1]))
        decl = JointDecl(
            jtype="revolute" if jt == _JNT_REVOLUTE else "prismatic",
            axis=axis, point=point, q_range=q_range, handle_body=body,
            qpos_adr=int(m.jnt_qposadr[jadr]))
        decl._h0 = np.asarray(env.get_body_pos(body), float)
        return decl
    except Exception:
        return None


def read_q(env, decl: JointDecl) -> Optional[float]:
    """当前关节配置（mj_data.qpos[qpos_adr]）。"""
    try:
        return float(env.mj_data.qpos[decl.qpos_adr])
    except Exception:
        return None


# ---- 三条判据（输出统一 (满足?, 裕度)，与 constraints.py 同型）----

def on_manifold_track(p_now: np.ndarray, q_now: float,
                      decl: JointDecl) -> Tuple[bool, float]:
    """流形成员判据：实测作用点 ∈ h(q) 的像（运行时监视版）。

    q_now 从编码器/仿真读（物体应随关节到的位形），p_now 从感知读
    （物体实际在哪）；两者相符 = 在流形上跟着关节走。违背 = 拉歪了
    （柜门被拉成门-框相对位移、抽屉斜卡）——关节任务最常见的静默失败：
    q 不动但物体在别处分力。裕度 = ‖p_now − h(q_now)‖（m）。
    """
    h = decl.handle_point(q_now)
    d = float(np.linalg.norm(np.asarray(p_now, float) - h))
    return (d <= MANIFOLD_TOL_M), d


MANIFOLD_TOL_M = 0.015   # 拉歪阈值：门缝量级（>15mm 相对位移=脱离流形）


def within_limits(q: float, q_range: Tuple[float, float],
                  margin: float = 0.02) -> Tuple[bool, float]:
    """限位判据：q ∈ [lo+margin, hi−margin]。

    违背 = 反向拉（q 朝 lo 走是关门）或顶死点。裕度 = 到最近限位的
    距离（同侧为正、越界为负）。
    """
    lo, hi = q_range
    if q < lo - margin or q > hi + margin:
        return False, min(q - lo, hi - q)
    return True, min(q - lo, hi - q)


def aligned_wrench(force: np.ndarray, decl: JointDecl,
                   apply_point: Optional[np.ndarray] = None,
                   ratio_min: float = 2.0) -> Tuple[bool, float]:
    """力旋量对齐判据：有效分量（绕轴力矩/沿轴力）vs 浪费分量。

    开柜门要绕轴力矩，垂直于轴的蛮力只会脱手或拉坏；旋钮=纯力矩，
    抽屉=纯力，柜门=力×力臂的力矩。force 作用在 apply_point（默认
    作用点=h(q) 当前位置）；revolute 的力矩参考点取作用点在轴上的
    投影（τ·axis 是不变量，与把手高度无关）。

    返回 (对齐?, 有效/浪费比)：revolute 看 τ·axis / ‖τ_perp‖，
    prismatic 看 F·axis / ‖F_perp‖。比值 ≥ ratio_min 才算有效驱动
    （低于此值大部分力在拧/撬，浪费且伤机构）。
    """
    F = np.asarray(force, float)
    if decl.jtype == "prismatic":
        f_ax = float(np.dot(F, decl.axis))
        f_perp = F - decl.axis * f_ax
        w = float(np.linalg.norm(f_perp))
    else:
        # 参考点取作用点在轴上的投影（pa）：r ⊥ 轴，τ·axis 是该选择下
        # 的不变量（轴矩与轴上参考点选取无关——柜门把手在任何高度拉，
        # 开门力矩都只由水平力臂贡献）。τ_perp = 弯曲/撬的浪费分量
        # （来自沿轴的推力等）。
        p = (np.asarray(apply_point, float) if apply_point is not None
             else (decl._h0 if decl._h0 is not None else decl.point))
        pa = decl.point + decl.axis * float(np.dot(p - decl.point, decl.axis))
        tau = np.cross(p - pa, F)
        t_ax = float(np.dot(tau, decl.axis))
        t_perp = tau - decl.axis * t_ax
        f_ax, w = t_ax, float(np.linalg.norm(t_perp))
    if w < 1e-9:
        # 无浪费分量的退化情形要再看有效分量：纯切向拉（τ 全在轴上）=
        # 完美对齐；径向对轴推（τ≡0）= 零驱动，不是对齐
        return (abs(f_ax) > 1e-9), (float("inf") if abs(f_ax) > 1e-9 else 0.0)
    return (abs(f_ax) / w >= ratio_min), (abs(f_ax) / w)


# ---- 机制映射（落进现有 Mechanism 框架，零新枚举）----

def mechanism_of_violation(violation: str) -> str:
    """关节判据违背 → 现有机制名（判别子/割/重试框架直接复用）。

    拉脱手=friction_slip（流形法向的摩擦锥违背）；
    运动卡死=contact_blocked（沿 h(q) 轨迹 clearance 不足）；
    拉歪/限位=geometry_squeeze（修方向/声明 q 目标，directive 表达）。
    """
    return {
        "off_manifold": "geometry_squeeze",
        "limit": "geometry_squeeze",
        "jammed": "contact_blocked",
        "slipped_off": "friction_slip",
    }.get(violation, "unknown")


def active_set(step: str, **kwargs) -> Dict[str, Dict[str, Any]]:
    """关节任务的监视器判据集（与 constraints.active_set 同型并行）。

    articulate/关节推拉步：流形跟踪 + 限位 + 力对齐三判据常驻。
    """
    if step in ("articulate", "pull_handle", "open_door", "open_drawer"):
        return {"on_manifold_track": {"decl": kwargs.get("decl")},
                "within_limits": {"q": kwargs.get("q"),
                                  "q_range": kwargs.get("q_range")},
                "aligned_wrench": {"force": kwargs.get("force"),
                                   "decl": kwargs.get("decl")}}
    return {}
