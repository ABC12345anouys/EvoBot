"""参数推导（附录 D 规则 1/2 的代码化）：参数 = f(场景特征)，不是裸值。

每条推导是物体中心坐标系的函数（禁世界坐标）：输入是 bounds /
gripper 几何 / Θ 后验，输出是 skill 参数。straddle 0.7 规则、
z_top 角点变换等**已验证推导**从 policies/rules.py 收敛到这里，
成为单一来源。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

# ---- 步数预算（§13"预算参数也是参数"的推导式）----

def descend_timeout(dist_z: float, vcap: float,
                    rate_measured: Optional[float] = None,
                    margin: float = 1.3) -> int:
    """timeout = f(下降距离, 实测速率)。

    rate_measured 有值时用实测（探针/前次 attempt 的证据），否则用
    vcap 保守上界速率。spatial:4 实证：90 步硬预算 × 倍率 全灭，
    距离/速率 推导一步到位。
    """
    if rate_measured and rate_measured > 0:
        return int(max(60.0, dist_z / rate_measured * margin)) + 1
    # 无实测：按 vcap 的 1/20 作保守速率（OSC 深 z 实测只有指令速率的零头）
    rate = max(vcap * 0.05, 1e-4)
    return int(max(120.0, dist_z / rate * margin)) + 1


def carry_timeout(dist_xy: float, carry_vcap: float,
                  timeout_scale: float = 1.0) -> int:
    """carry 预算 = f(水平距离, 巡航限速)。现有 methods.py:65 公式的
    收敛版（去 magic number，保行为）。dist/rate × margin + 基础步。"""
    base = 160.0 + 120.0 * dist_xy / max(carry_vcap, 0.01)
    return int(min(1500.0, max(160.0, base)) * timeout_scale)


# ---- 转运高度（carry 时夹持物底部必须全程越过容器口沿，否则刮沿撞脱爪：
# milk/basket 实测 milk 底 0.129 < 沿 0.137 脱爪；上限受 OSC 可达与
# 炉前壁等遮蔽约束，走 phys 通道 carry_z_cap_m，不入此推导）----

# 口沿上方净余量：防数值抖动与沿厚误差
CARRY_RIM_CLEARANCE = 0.015
# TCP 悬深：夹持点 TCP 在物体顶部下方的典型深度（手指包络）
CARRY_TCP_DROP = 0.06


def carry_lift_need(rim_z: float, body_half_h: float) -> float:
    """转运高度下限 = 口沿 + 净余量 + 物体半高 + TCP 悬深。

    rim_z 是容器口沿顶面高度；body_half_h 是被夹物体的半高。
    """
    return rim_z + CARRY_RIM_CLEARANCE + body_half_h + CARRY_TCP_DROP


# ---- 接触软停带（杀标定彩票：从逐任务 YAML 到运行时推导）----

def contact_stop_band(measured_shortfall: Optional[float] = None,
                      default: float = 0.05,
                      container: bool = False,
                      margin: float = 0.008) -> float:
    """band = f(实测短缩)。无实测时回落现行默认值（容器 0.015 /
    实心 0.05，均已实证）；有实测（接触探针/前次证据）时
    band = shortfall + margin。测量替代标定。"""
    if measured_shortfall is not None and measured_shortfall >= 0:
        return float(measured_shortfall) + margin
    return 0.015 if container else default


# ---- 抓取几何（已验证规则的收敛来源，原 policies/rules.py）----

def straddle_offset(half_y: float, ratio: float = 0.70) -> float:
    """容器 straddle 偏置：center ± ratio·half_y，纯 Y 方向。

    实证（探针，akita_black_bowl）：0.7·half_y=39.4mm 时内指径向 0.6mm
    落腔、外指远超沿口，straddle 决定性；X 向双指蹭沿最差。
    纯 Y 的语义是"沿夹爪两指分离方向偏置"——L2 换臂时分离方向由
    gripper geometry 投影（P1-1）。
    """
    return float(ratio) * float(half_y)


def straddle_offset_search(cloud, center, direction, z_lo, z_hi,
                           half_spread: float, finger_r: float,
                           n: int = 64, own=None) -> Optional[float]:
    """容器 straddle 偏置 = 双指下降走廊净空的 argmax（几何推导）。

    对偏置 off 网格，夹爪中心位于 center + off·dir，两指 shaft 位于
    center + (off ∓ half_spread)·dir；净空 = (z_lo, z_hi) 高度带内
    表面点到 shaft 轴的最小距离 − finger_r。取双指净空的最小值作为该
    off 的分数，argmax 者为偏置；与最大值差 <1mm 的最优平台取中位，
    避免网格/采样噪声把偏置推到平台边缘。

    straddle 覆盖约束（own 非空时）：偏置必须让目标自身口沿（own 点云
    z 带内投影）至少有一点落在两指 shaft 之间——否则双指都在容器外，
    不是 straddle 而是"离目标越远净空越大"的退化解，argmax 恒落在
    网格边界、夹爪整体偏空（spatial:6 实证：网格顶 0.203，双指越过
    碗沿 9cm，抓取点甩到盘子上方）。own=None 时退回纯净空语义
    （旧行为，点云不可得路径不启用）。

    与 straddle_offset(ratio) 的关系：ratio 标定点在网格可达范围内时
    自动参与竞争——净空更优的几何点胜出，旧标定从"规则"降级为"下界"，
    沿口壁厚/腔径随物体变化时不再需要重新标定 ratio。

    点云为空返回 None（调用方回落 ratio 路径）。
    """
    pts = np.asarray(cloud, float).reshape(-1, 3)
    band = pts[(pts[:, 2] > float(z_lo)) & (pts[:, 2] < float(z_hi))]
    if len(band) == 0:
        return None
    c = np.asarray(center, float).reshape(2)
    u = np.asarray(direction, float).reshape(2)
    u = u / max(np.linalg.norm(u), 1e-9)
    rel = band[:, :2] - c
    along = rel @ u                    # 表面点在 dir 上的投影
    perp = rel - np.outer(along, u)    # 垂向分量
    perp_n = np.linalg.norm(perp, axis=1)
    r_max = float(np.hypot(rel[:, 0], rel[:, 1]).max())
    lo = max(0.0, float(half_spread) - r_max)
    hi = float(half_spread) + r_max
    offs = np.linspace(lo, hi, int(n))
    # 目标自身口沿投影：straddle 覆盖判据（见 docstring）
    own_along = None
    if own is not None:
        op = np.asarray(own, float).reshape(-1, 3)
        ob = op[(op[:, 2] > float(z_lo)) & (op[:, 2] < float(z_hi))]
        if len(ob):
            own_along = (ob[:, :2] - c) @ u
    scores = []
    for off in offs:
        if own_along is not None and not (
                ((own_along > off - float(half_spread))
                 & (own_along < off + float(half_spread))).any()):
            scores.append(-np.inf)     # 口沿不在两指间：非 straddle
            continue
        in_w = (off - float(half_spread)) - along    # 内指（靠腔侧）shaft
        out_w = (off + float(half_spread)) - along   # 外指 shaft
        d_in = float(np.sqrt(in_w ** 2 + perp_n ** 2).min())
        d_out = float(np.sqrt(out_w ** 2 + perp_n ** 2).min())
        scores.append(min(d_in, d_out) - float(finger_r))
    scores = np.asarray(scores)
    finite = scores[np.isfinite(scores)]
    if not len(finite):
        return None                    # 全网格都不构成 straddle → 回落 ratio
    best = finite.max()
    plateau = offs[scores >= best - 1e-3]
    return float(np.median(plateau))


def straddle_direction(center, clearance_fn, candidates=2) -> int:
    """±Y 翻向：按两侧障碍净空选（<2cm 为楔止风险，spatial:4 实证）。

    clearance_fn(p) → 该点的水平净空（m）。返回符号 +1/-1。
    """
    cx, cy = float(center[0]), float(center[1])
    offs = [(+1.0, float(clearance_fn([cx, cy + 0.05]))),
            (-1.0, float(clearance_fn([cx, cy - 0.05])))]
    best_s, best_c = max(offs, key=lambda t: t[1])
    if best_c < 0.02:
        # 两侧都不足：退 12 方向搜最大净空
        import math
        for k in range(12):
            a = 2 * math.pi * k / 12.0
            p = [cx + 0.05 * math.cos(a), cy + 0.05 * math.sin(a)]
            c = float(clearance_fn(p))
            if c > best_c:
                best_s, best_c = None, c  # 非标准方向由调用方处理
        return 0  # 信号：需全方向搜索
    return int(best_s)


def grasp_z_flat(thickness: float, z_top: float,
                 thin_thresh: float = 0.015) -> float:
    """扁平物包边（goal:5 实证）：thickness<阈值 → 指尖包沿（z_top-2mm），
    否则中心置顶（z_top-min(0.012, z_top·0.5)）。"""
    if thickness < thin_thresh:
        return float(z_top) - 0.002
    return float(z_top) - min(0.012, float(z_top) * 0.5)


# ---- 推动（push）几何：平板类物体抓不可取时的刚体操作策略 ----
# goal:5 实证探针（probe_goal5_plate/2/3）：plate 厚 19.1mm，任何 TCP z
# 下上夹爪闭合都夹空（指尖-TCP 偏置 11.8mm ≈ 盘厚 2/3，力封闭不可建）；
# demo 真值（probe_goal5_demo）：155 步动作夹爪全程张开——任务本体是推。
PUSH_FLAT_MAX_THICKNESS_M = 0.02   # 低于此厚度：抓不可取，路由 push
PUSH_FINGER_DROP_M = 0.012         # 指尖在 TCP 下方偏置（libero panda 实测）


def pushable_by_thickness(thickness: float) -> bool:
    """厚度低于力封闭下限 → 抓取不可行（上夹爪指尖偏置相对盘厚过大）。"""
    return float(thickness) < PUSH_FLAT_MAX_THICKNESS_M


def push_z(z_mid: float, finger_drop: float = PUSH_FINGER_DROP_M) -> float:
    """推动时 TCP 高度：指尖（TCP 下方 finger_drop）对齐物体厚中部，
    指根平面推物体侧面，防滑到顶面/底面下。"""
    return float(z_mid) + float(finger_drop)


def push_start_xy(body_xy, target_xy, half_max: float,
                  margin: float = 0.035) -> np.ndarray:
    """推动起点：物体背面（背离目标一侧）沿半长 + 余量，直线路径不横穿物体。"""
    body_xy = np.asarray(body_xy, float)[:2]
    target_xy = np.asarray(target_xy, float)[:2]
    d = body_xy - target_xy
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        d = np.array([1.0, 0.0])
        n = 1.0
    return body_xy + (d / n) * (float(half_max) + float(margin))


# ---- 摩擦锥应用（Θ 后验 → 执行参数）----

def place_vcap_from_mu(mu: float, m: float, f_grip: float,
                       default: float = 0.05) -> float:
    """放置下落限速 = f(μ 后验, 质量, 夹持力)。

    摩擦可承担的切向减速度 a_t ≤ μ·(F_grip - m·g)/m；下落段在 v 内
    刹停需 a = v²/(2·d)。保守取 v = sqrt(2·d·a_t)，d=1cm 制动距离。
    μ 后验缺省（无辨识）时回落 default（现行 0.05 实证值）。
    """
    if mu <= 0 or m <= 0 or f_grip <= m * 9.81:
        return default
    a_t = mu * (f_grip - m * 9.81) / m
    v = float(np.sqrt(2.0 * 0.01 * max(a_t, 0.0)))
    return float(min(0.5, max(0.005, v)))


def grip_force_need(mu: float, m: float, safety: float = 1.5) -> float:
    """夹持力下限 = safety·m·g/μ（摩擦锥裕度版）。Θ μ 缺省时返回
    None 语义由调用方处理（保现行行为）。"""
    if mu <= 0:
        return 0.0
    return safety * m * 9.81 / mu
