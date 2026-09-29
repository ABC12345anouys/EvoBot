"""几何绕障 waypoint 规划（规划层补丁，LIBERO carry/above 撞墙的根因修复）。

背景（见 TODO 09-23 归因）：carry 是固定 L 形三段直线（ik_servo.py CarrySkill），
水平段与障碍物 AABB 相交时没有任何避让，磨 368 步等超时；失败后 resume 重跑
同一路径再撞。本模块提供纯几何的绕障 waypoint 生成，输入只有：

- 当前/目标 TCP（世界坐标）
- 障碍 AABB 列表（由 env.object_bounds + auto_obstacles 清单现场取）
- 巡航高度上限（carry_z_cap_m，OSC 可达极限约束）

策略（两级逃逸，与 Phase 2 计划的渐进重规划对应）：
1. 首选：水平段与膨胀障碍相交时，把巡航高度提升到"最高挡路障碍顶 + 余量"
   （飞越），受 z_cap 约束；
2. 飞不过（cap 限制）：在挡路障碍的膨胀盒外侧插入一个垂直于路径方向的
   侧向绕点（绕行），生成 5 段 waypoint。

纯函数、无 env 依赖，可离线单测（scripts 里直接跑本文件的 __main__）。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Aabb = Tuple[np.ndarray, np.ndarray]  # (min[3], max[3]) 世界坐标轴对齐盒


def bounds_to_aabb(bounds: Dict[str, float], margin: float = 0.0) -> Aabb:
    """object_bounds dict → 世界 AABB（膨胀 margin）。"""
    cx, cy = float(bounds["center"][0]), float(bounds["center"][1])
    hx, hy = float(bounds["half_x"]), float(bounds["half_y"])
    z0, z1 = float(bounds["z_bottom"]), float(bounds["z_top"])
    lo = np.array([cx - hx - margin, cy - hy - margin, z0 - margin])
    hi = np.array([cx + hx + margin, cy + hy + margin, z1 + margin])
    return lo, hi


def segment_hits_aabb(p0: np.ndarray, p1: np.ndarray, aabb: Aabb) -> bool:
    """线段-AABB 相交（slab 法）。端点在盒内也算相交。"""
    lo, hi = aabb
    d = p1 - p0
    t0, t1 = 0.0, 1.0
    for axis in range(3):
        if abs(d[axis]) < 1e-12:
            if p0[axis] < lo[axis] or p0[axis] > hi[axis]:
                return False
            continue
        a = (lo[axis] - p0[axis]) / d[axis]
        b = (hi[axis] - p0[axis]) / d[axis]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1:
            return False
    return True


def blocking_obstacles(p0: np.ndarray, p1: np.ndarray,
                       obstacles: Sequence[Aabb]) -> List[int]:
    """返回水平投影线段（取两端 z 的低位）碰到的障碍下标。

    水平段只看 xy 相交 + z 区间重叠：carry 水平段在巡航高度，障碍只要
    顶面高于该段 z 就可能挡路。
    """
    hits = []
    for i, (lo, hi) in enumerate(obstacles):
        # xy 投影相交（2D slab）
        d = p1 - p0
        ok = True
        for axis in (0, 1):
            if abs(d[axis]) < 1e-12:
                if p0[axis] < lo[axis] or p0[axis] > hi[axis]:
                    ok = False
                    break
                continue
            a = (lo[axis] - p0[axis]) / d[axis]
            b = (hi[axis] - p0[axis]) / d[axis]
            if min(a, b) > 1.0 or max(a, b) < 0.0:
                ok = False
                break
        # z 区间：障碍顶高于线段两端较低者（会被撞）
        if ok and hi[2] > min(p0[2], p1[2]):
            hits.append(i)
    return hits


def plan_carry_waypoints(cur: Sequence[float], tgt: Sequence[float],
                         obstacles: Sequence[Aabb], *,
                         cruise: float, hover: float,
                         z_cap: float,
                         lateral_step: float = 0.16,
                         margin: float = 0.03) -> List[np.ndarray]:
    """生成绕障 carry waypoint（rise → 水平（可能飞越/绕行）→ descend）。

    - 无障碍相交：保持旧三段 L 形（行为不变）；
    - 相交：优先飞越（巡航提到挡路障碍最高顶 + margin，受 z_cap 约束）；
      cap 内飞不过 → 侧向绕点（取障碍膨胀盒在垂直路径方向上较近的一侧，
      绕出 lateral_step）；
    - 两个方向都绕不出（目标被包围）：退回飞越路径（尽力而为）。
    """
    cur = np.asarray(cur, float).reshape(3)
    tgt = np.asarray(tgt, float).reshape(3)
    cruise = float(min(cruise, z_cap))
    wps: List[np.ndarray] = [np.array([cur[0], cur[1], cruise])]

    hits = blocking_obstacles(wps[0],
                              np.array([tgt[0], tgt[1], cruise]),
                              obstacles)
    if not hits:
        wps.append(np.array([tgt[0], tgt[1], cruise]))
    else:
        top_max = max(obstacles[i][1][2] for i in hits)
        fly_z = min(top_max + margin + 0.02, z_cap)
        fly_hits = blocking_obstacles(
            np.array([cur[0], cur[1], fly_z]),
            np.array([tgt[0], tgt[1], fly_z]), obstacles) \
            if fly_z > cruise else hits
        if fly_z > cruise and not fly_hits:
            wps.append(np.array([cur[0], cur[1], fly_z]))
            wps.append(np.array([tgt[0], tgt[1], fly_z]))
        else:
            # 侧向绕行：在垂直于路径的方向上选较近一侧的绕点
            path = np.array([tgt[0] - cur[0], tgt[1] - cur[1]])
            plen = float(np.hypot(*path))
            if plen < 1e-6:
                wps.append(np.array([tgt[0], tgt[1], cruise]))
            else:
                perp = np.array([-path[1], path[0]]) / plen
                best = None
                for side in (1.0, -1.0):
                    mid = np.array([(cur[0] + tgt[0]) / 2,
                                    (cur[1] + tgt[1]) / 2])
                    detour = mid + side * perp * lateral_step
                    det_hits = blocking_obstacles(
                        np.array([cur[0], cur[1], cruise]),
                        np.array([detour[0], detour[1], cruise]), obstacles)
                    det_hits += blocking_obstacles(
                        np.array([detour[0], detour[1], cruise]),
                        np.array([tgt[0], tgt[1], cruise]), obstacles)
                    if best is None or len(det_hits) < best[0]:
                        best = (len(det_hits), detour)
                detour = best[1]
                wps.append(np.array([detour[0], detour[1], cruise]))
    wps.append(np.array([tgt[0], tgt[1], tgt[2] + hover]))
    # 剔除与当前点重合的 waypoint
    out = [w for w in wps if np.linalg.norm(w - cur) > 0.02]
    return out or [np.array([tgt[0], tgt[1], tgt[2] + hover])]


if __name__ == "__main__":
    # 离线单测：纯几何，不需要 sim
    ob_cabinet = (np.array([-0.05, -0.30, 0.85]), np.array([0.12, -0.12, 1.05]))
    ob_plate = (np.array([0.0, 0.14, 0.89]), np.array([0.11, 0.24, 0.92]))

    # 1. 线段-AABB
    assert segment_hits_aabb(np.array([-0.2, -0.21, 0.97]),
                             np.array([0.05, -0.21, 0.97]), ob_cabinet)
    assert not segment_hits_aabb(np.array([-0.2, 0.2, 0.97]),
                                 np.array([0.05, 0.2, 0.97]), ob_cabinet)

    # 2. 无障碍 → L 形（cur 已在巡航高度时 rise 段被剔除 = 2 段）
    wps = plan_carry_waypoints([-0.2, 0.2, 1.0], [0.05, 0.2, 0.94],
                               [ob_cabinet], cruise=1.0, hover=0.08,
                               z_cap=1.18)
    assert len(wps) == 2, wps

    # 3. 被柜挡 → 飞越（柜顶 1.05 + margin < cap 1.18）
    wps = plan_carry_waypoints([-0.2, -0.21, 0.97], [0.05, -0.21, 0.94],
                               [ob_cabinet], cruise=1.0, hover=0.08,
                               z_cap=1.18)
    assert any(abs(w[2] - 1.10) < 1e-6 for w in wps), wps  # 1.05+0.03+0.02

    # 4. cap 内飞不过 → 侧向绕点存在
    ob_tall = (np.array([-0.05, -0.30, 0.85]), np.array([0.12, -0.12, 1.17]))
    wps = plan_carry_waypoints([-0.2, -0.21, 0.97], [0.05, -0.21, 0.94],
                               [ob_tall], cruise=1.0, hover=0.08,
                               z_cap=1.18)
    assert len(wps) >= 4, wps
    # 每个水平段（用段自身高度）不再碰障碍
    pts = [np.array([-0.2, -0.21, 0.97])] + list(wps)
    for a, b in zip(pts, pts[1:]):
        if abs(a[2] - b[2]) < 1e-6:  # 只查水平段
            assert not blocking_obstacles(a, b, [ob_tall]), (wps, a, b)

    # 5. 障碍低于巡航段 → 不挡（rise 剔除后 = 2 段）
    wps = plan_carry_waypoints([-0.2, 0.2, 1.0], [0.05, 0.2, 0.94],
                               [ob_plate], cruise=1.0, hover=0.08,
                               z_cap=1.18)
    assert len(wps) == 2, wps
    print("avoidance 单测全部通过 ✔")
