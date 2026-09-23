"""通用 IK 末端伺服原语（跨 env：LIBERO OSC / robopal CARTIK 共享）。

设计目标（用户架构）：IK/伺服用**一个 skill**，操作标准化，精细化差异
（hover、k、stall_window、reach_tol、release_offset 等）走 per-env YAML
（`skills/ik_servo.<env>.yaml`），由 runner 起任务时 deep_merge 进 cfg，
方法链通过 `$cfg.<key>` 取值。反思学习把修正参数写回 YAML，不写死代码。

底层不再各自直连控制器，统一调 `primitives.servo_step`（语义夹爪
+1=闭合 / -1=张开 / 0=保持，内部按 `env._darwin_kind` 翻成各 env 物理值）；
碰撞避障复用 `collision.follow_waypoints` + `get_monitor`（libero 无
obstacle_bodies 时返回 None，自动降级直线 P 控制）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..base import Skill, SkillSpec, SkillKind, Confidence
from . import servo_step, _grip_site
from .collision import follow_waypoints, get_monitor, plan_corridor

# 伺服/卡死默认值（可被 per-env YAML 经 $cfg 覆盖）
_DEFAULT_K = 5.0
_DEFAULT_STALL_WINDOW = 30
_DEFAULT_STALL_PROGRESS = 0.005


def _eef(env, site: str) -> np.ndarray:
    return np.asarray(env.get_site_pos(site), dtype=float)


def _follow_waypoints(env, site: str, wps: List[np.ndarray], *,
                      gripper: float, tol: float = 0.015, timeout: int = 120,
                      k: float = _DEFAULT_K, vcap: float = 1.0,
                      stall_window: int = _DEFAULT_STALL_WINDOW,
                      stall_progress: float = _DEFAULT_STALL_PROGRESS,
                      actor: str = "agent0") -> Dict[str, Any]:
    """顺序跟随经过点（逐段 3D 距离收敛），带 stall fail-fast。

    跨 env 通用版：每步调 servo_step（语义 gripper），不直连任何控制器。
    vcap 走 per-env YAML（libero OSC=1.0 全量程；robopal CARTIK=0.05 限速）。
    """
    total = 0
    for wp in wps:
        wp = np.asarray(wp, float)
        d_at_window = None
        reached = False
        for i in range(timeout):
            end = _eef(env, site)
            d = float(np.linalg.norm(end - wp))
            if d < tol:
                reached = True
                break
            if i % stall_window == 0:
                # 窗口起点记距离；窗口结束时进展不足才判卡死（长走廊不误报）
                if d_at_window is not None and i >= stall_window \
                        and d_at_window - d < stall_progress:
                    return {"success": False, "reason": "ik_stalled",
                            "steps": total + i, "end": end.tolist()}
                d_at_window = d
            servo_step(env, site, wp, gripper=gripper, k=k, vcap=vcap, actor=actor)
        if not reached:
            return {"success": False, "reason": "ik_reach_timeout",
                    "steps": total + timeout, "end": _eef(env, site).tolist()}
        total += i + 1
    return {"success": True, "steps": total, "end": _eef(env, site).tolist()}


class IkServoSkill(Skill):
    """通用 IK 末端伺服（mode=above/descend/waypoint）。

    - above：移动到 point 正上方 hover 高度（先升高再平移的走廊，避障），
      张开夹爪。跨 env：robopal 走 plan_corridor 安全走廊，libero 无障碍走直线。
    - descend：竖直下降到 point（xy 锁死、张爪），带 stall + 接触软停判定。
    - waypoint：跟随显式经过点序列（供方法链灵活编排）。
    """

    spec = SkillSpec(
        name="ik_servo",
        description="IK end-effector servo (mode=above/descend/waypoint), cross-env via servo_step",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.VERIFIED,
        applies_when="IK 末端闭环伺服（悬停/下降/经过点）",
    )

    def execute(self, env, point: Optional[List[float]] = None,
                mode: str = "above", hover: float = 0.08,
                wps: Optional[List[List[float]]] = None,
                stop_above: float = 0.0, reach_tol: float = 0.006,
                actor: str = "agent0", grip_site: Optional[str] = None,
                timeout: int = 120, k: float = _DEFAULT_K, vcap: float = 1.0,
                stall_window: int = _DEFAULT_STALL_WINDOW,
                stall_progress: float = _DEFAULT_STALL_PROGRESS,
                contact_stop_band: float = 0.05,
                body: Optional[str] = None) -> Dict[str, Any]:
        site = _grip_site(grip_site, actor)
        mon = get_monitor(env, actor)
        clear_min, pair_min = 1.0, ""

        if mode == "waypoint":
            if not wps:
                return {"success": False, "reason": "ik_no_waypoints", "steps": 0}
            pts = [np.asarray(w, float) for w in wps]
            r = _follow_waypoints(env, site, pts, gripper=0.0, timeout=timeout,
                                  k=k, vcap=vcap, stall_window=stall_window,
                                  stall_progress=stall_progress, actor=actor)
            if mon is not None:
                clear_min, pair_min = mon.clearance()
            r.update({"min_clearance": clear_min, "coll_pair": pair_min})
            return r

        if mode == "above":
            pt = np.asarray(point, float)
            top = np.array([pt[0], pt[1], pt[2] + hover])
            cur = _eef(env, site)
            # 当前低于走廊高度时先竖直抬起，再水平平移，落到悬停点
            wp_list: List[np.ndarray] = []
            if cur[2] < top[2] - 0.01:
                wp_list.append(np.array([cur[0], cur[1], top[2]]))
            wp_list.append(np.array([pt[0], pt[1], top[2]]))
            corridor = plan_corridor(cur, wp_list[-1], mon) if mon is not None else wp_list
            return follow_waypoints(env, actor, site, corridor, mon, gripper=-1,
                                     k=k, seg_tol=0.02, timeout=timeout,
                                     vcap=vcap,
                                     final_check=lambda e: bool(
                                         np.linalg.norm((e - top)[[0, 1]]) < 0.015))

        if mode == "descend":
            pt = np.asarray(point, float)
            # 抓取点 z（GraspNet/env.grasp_point 已含指尖偏移）+ stop_above。
            # robopal DescendSkill 用 body z 跟踪被碰歪物体；libero 用 pt.z（旧
            # LiberoDescendSkill 同款），GraspNet 接管后两 env 都用 pt.z。
            z_goal = pt[2] + stop_above
            # xy 守卫：z 到位但 TCP xy 偏离 grasp_pt > xy_tol 时不判成功，
            # 继续伺服纠正 xy。OSC 在某些臂构型下下降时 xy 会漂移 2-4cm，
            # 若直接判成功→close_gripper 夹空→lift_no_grip。
            # xy_tol 与 above 模式 final_check 一致（1.5cm），够松不误报。
            xy_tol = max(0.015, reach_tol * 2.5)
            z_window_start = None
            stall_hits = 0  # 连续无进展窗口数；双窗口确认防 OSC 慢速被误判
            for t in range(timeout):
                end = _eef(env, site)
                xy_err = float(np.linalg.norm(end[:2] - pt[:2]))
                # z 到位且 xy 到位才算成功（防 xy 漂移致假成功→夹空）
                if abs(end[2] - z_goal) < reach_tol and xy_err < xy_tol:
                    return {"success": True, "steps": t, "end": end.tolist(),
                            "xy_err": xy_err,
                            "min_clearance": clear_min, "coll_pair": pair_min}
                if t % stall_window == 0:
                    if z_window_start is not None:
                        if z_window_start - end[2] < stall_progress:
                            stall_hits += 1
                            # 双窗口确认（连续 60 步无 z 进展）才软停/判卡死：
                            # OSC 在关节限位边缘下降慢，单窗口 30 步会误判，
                            # 提前 close 夹空（10_07 实测差 8mm 到位被判停）
                            if stall_hits >= 2:
                                # 接触软停：已进入目标上方 contact_stop_band 夹持带
                                # 且 xy 已收敛（OSC 对深 z 有到达极限，放宽容差
                                # 避免误判卡死）；xy 仍偏则不算到位
                                if 0.0 <= end[2] - z_goal <= contact_stop_band \
                                        and xy_err < xy_tol:
                                    return {"success": True, "steps": t,
                                            "end": end.tolist(), "contact_stop": True,
                                            "xy_err": xy_err,
                                            "min_clearance": clear_min,
                                            "coll_pair": pair_min}
                                # z 停且 xy 偏 → xy 不可达，用 grip_failed 类因由
                                # 让反思提高 k_descend（而非 timeout 类降低 k）
                                reason = ("ik_descend_xy_drift" if xy_err >= xy_tol
                                          else "ik_descend_stalled")
                                return {"success": False, "reason": reason,
                                        "steps": t, "end": end.tolist(),
                                        "xy_err": xy_err,
                                        "min_clearance": clear_min, "coll_pair": pair_min}
                        else:
                            stall_hits = 0
                    z_window_start = end[2]
                servo_step(env, site, [pt[0], pt[1], z_goal], gripper=-1, k=k,
                           vcap=vcap, actor=actor)
                if mon is not None:
                    d, pr = mon.clearance()
                    if d < clear_min:
                        clear_min, pair_min = d, pr
                    if mon.violated(d):
                        return {"success": False, "reason": "collision_risk",
                                "coll_pair": pr, "min_clearance": d, "steps": t}
            return {"success": False, "reason": "ik_descend_timeout",
                    "steps": timeout, "end": _eef(env, site).tolist(),
                    "min_clearance": clear_min, "coll_pair": pair_min}

        return {"success": False, "reason": f"ik_unknown_mode:{mode}", "steps": 0}


class CarrySkill(Skill):
    """携带物体移动到目标点上方（先升高、再平移、再降到悬停的安全走廊）。

    跨 env：libero 无障碍走三段 waypoint；robopal 经 plan_corridor 避柜子。
    夹爪保持闭合（语义 +1，holding）。
    """

    spec = SkillSpec(
        name="carry",
        description="Carry grasped object to above target via rise-translate-descend corridor",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.VERIFIED,
        applies_when="抓取后搬运到放置目标上方",
    )

    def execute(self, env, target: List[float], hover: float = 0.10,
                safe_z_margin: float = 0.06, actor: str = "agent0",
                grip_site: Optional[str] = None,
                timeout: int = 160, k: float = _DEFAULT_K, vcap: float = 1.0
                ) -> Dict[str, Any]:
        site = _grip_site(grip_site, actor)
        tgt = np.asarray(target, float)
        end0 = _eef(env, site)
        cruise = max(end0[2], tgt[2] + hover + safe_z_margin)
        # 巡航上限 carry_z_cap_m：须高于最高容器口沿+物体底余量（微波炉
        # 1.18），低于 OSC 可达极限（1.29 跑飞）。libero yaml 设 1.18，
        # 默认 1.0（robopal 场景矮，行为不变）。
        from ..physics_profile import phys_get
        cruise = min(float(cruise),
                     float(phys_get(env, "carry_z_cap_m", 1.0)))
        wps = [
            np.array([end0[0], end0[1], cruise]),        # 竖直抬到巡航高度
            np.array([tgt[0], tgt[1], cruise]),          # 水平平移到目标上方
            np.array([tgt[0], tgt[1], tgt[2] + hover]),  # 降到目标上方悬停
        ]
        mon = get_monitor(env, actor)
        return follow_waypoints(env, actor, site, wps, mon, gripper=+1, k=k,
                                seg_tol=0.02, timeout=timeout, vcap=vcap,
                                final_check=lambda e: bool(
                                    np.linalg.norm((e - wps[-1])[[0, 1]]) < 0.02))


__all__ = ["IkServoSkill", "CarrySkill"]
