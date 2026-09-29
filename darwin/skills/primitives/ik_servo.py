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
                reach_limit_band: float = 0.035,
                f_free_n: float = 0.5,
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
            # 绕障走廊（与 CarrySkill 同一套）：水平段与障碍膨胀 AABB 相交时
            # 飞越或侧向绕点。goal:3/4 实证 above 直线会卡柜顶磨停（stall）。
            wps = None
            try:
                from .avoidance import bounds_to_aabb, plan_carry_waypoints
                from ..physics_profile import phys_get
                obs_names = getattr(env, "obstacle_bodies", None)
                if obs_names and hasattr(env, "object_bounds"):
                    aabbs = []
                    for nm in obs_names:
                        try:
                            aabbs.append(bounds_to_aabb(
                                env.object_bounds(nm), margin=0.03))
                        except Exception:
                            continue
                    if aabbs:
                        wps = plan_carry_waypoints(
                            cur, top, aabbs, cruise=float(top[2]), hover=0.0,
                            z_cap=float(phys_get(env, "carry_z_cap_m", 1.0)),
                            lateral_step=0.16, margin=0.03)
            except Exception:
                wps = None
            if wps is None:
                # 旧走廊：先竖直抬到 hover 高度，再水平平移
                wps = []
                if cur[2] < top[2] - 0.01:
                    wps.append(np.array([cur[0], cur[1], top[2]]))
                wps.append(np.array([pt[0], pt[1], top[2]]))
                wps = plan_corridor(cur, wps[-1], mon) if mon is not None else wps
            return follow_waypoints(env, actor, site, wps, mon, gripper=-1,
                                     k=k, seg_tol=0.02, timeout=timeout,
                                     vcap=vcap,
                                     final_check=lambda e: bool(
                                         np.linalg.norm((e - top)[[0, 1]]) < 0.015))

        if mode == "descend":
            from ...physics.observables import Collector
            from ...physics.discriminators import classify_stall
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
            col = Collector(env, site, step="ik_servo.descend", body=body)
            z_window_start = None
            stall_hits = 0  # 连续无进展窗口数；双窗口确认防 OSC 慢速被误判

            def _done(ok, reason, t, end, xy_err, mech=None, **extra):
                ev = col.finish(ok, reason, clear_min, pair_min, goal_z=z_goal)
                d = {"success": ok, "steps": t, "end": end.tolist(),
                     "xy_err": xy_err, "min_clearance": clear_min,
                     "coll_pair": pair_min,
                     "evidence": ev.to_dict()}
                if mech is not None:
                    d["mechanism"] = mech.value
                d.update(extra)
                return d

            for t in range(timeout):
                end = _eef(env, site)
                col.sample(t, end)
                xy_err = float(np.linalg.norm(end[:2] - pt[:2]))
                # z 到位且 xy 到位才算成功（防 xy 漂移致假成功→夹空）
                if abs(end[2] - z_goal) < reach_tol and xy_err < xy_tol:
                    return _done(True, "", t, end, xy_err, z_hit=True)
                if t % stall_window == 0:
                    if z_window_start is not None:
                        if z_window_start - end[2] < stall_progress:
                            stall_hits += 1
                            # 双窗口确认（连续 60 步无 z 进展）才软停/判卡死：
                            # OSC 在关节限位边缘下降慢，单窗口 30 步会误判，
                            # 提前 close 夹空（10_07 实测差 8mm 到位被判停）
                            if stall_hits >= 2:
                                col.sample_force(t)
                                mech = classify_stall(
                                    col.ev, xy_tol=xy_tol, z_goal=z_goal,
                                    reach_limit_band=reach_limit_band)
                                # 接触软停：已进入目标上方 contact_stop_band 夹持带
                                # 且 xy 已收敛（OSC 对深 z 有到达极限，放宽容差
                                # 避免误判卡死）；xy 仍偏则不算到位
                                if 0.0 <= end[2] - z_goal <= contact_stop_band \
                                        and xy_err < xy_tol:
                                    return _done(True, "", t, end, xy_err,
                                                 mech, contact_stop=True)
                                # F=0 到达极限接受：停滞时若指尖-物体接触力≈0，
                                # 说明停点在自由空间，停滞原因是 OSC 可达极限而
                                # 非几何阻挡（spatial:4 实证：停在沿口上方 22mm
                                # F=0）。与"压沿口停滞"（F>0，真几何问题）区分。
                                # 判据三条件缺一不可：band 内 + xy 收敛 + 无接触。
                                # 机制分类由 physics.discriminators 给出（同一
                                # 物理，收敛到四合一引擎单一来源）。
                                if body and 0.0 <= end[2] - z_goal <= reach_limit_band \
                                        and xy_err < xy_tol:
                                    f_now = col.ev.f_at_end
                                    if f_now < f_free_n:
                                        return _done(True, "", t, end, xy_err,
                                                     mech, reach_limit=True,
                                                     force=f_now)
                                # z 停且 xy 偏 → xy 不可达，用 grip_failed 类因由
                                # 让反思提高 k_descend（而非 timeout 类降低 k）
                                reason = ("ik_descend_xy_drift" if xy_err >= xy_tol
                                          else "ik_descend_stalled")
                                return _done(False, reason, t, end, xy_err, mech)
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
                        return _done(False, "collision_risk", t, end, xy_err,
                                     coll_pair=pr, min_clearance=d)
            end = _eef(env, site)
            return _done(False, "ik_descend_timeout", timeout, end,
                         float(np.linalg.norm(end[:2] - pt[:2])))

        return {"success": False, "reason": f"ik_unknown_mode:{mode}", "steps": 0}


class PushMoveSkill(Skill):
    """平面推动：TCP 定在物体中高位，xy 向目标区伺服，物体入区即成功。

    平板类物体的刚体操作策略（goal:5 实证：plate 厚 19.1mm，指尖-TCP
    偏置 11.8mm 使力封闭不可建，任何 z 闭合都夹空；demo 真值夹爪全程
    张开——任务本体是推不是抓）。不做抓取：指尖侧面接触推物体滑入
    目标区，终态由 BDDL 谓词核验（与 place 同口径）。
    """

    spec = SkillSpec(
        name="push_move",
        description="Planar push: servo at object mid-height toward target region until body enters region box",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.PROBABLE,
        applies_when="推动平板/不可抓物体进入目标区",
    )

    def execute(self, env, target: List[float], body: str,
                region_site: Optional[str] = None,
                region_half: Optional[List[float]] = None,
                actor: str = "agent0", grip_site: Optional[str] = None,
                timeout: int = 240, k: float = _DEFAULT_K,
                vcap: float = 0.3, overshoot: float = 0.06,
                stall_window: int = _DEFAULT_STALL_WINDOW,
                stall_progress: float = 0.003) -> Dict[str, Any]:
        site = _grip_site(grip_site, actor)
        tgt = np.asarray(target, float)[:2]   # region 中心 xy（静态区够用）
        half = (np.asarray(region_half, float)[:2]
                if region_half is not None else None)

        def _region():
            """(中心xy, 半尺寸xy)：优先 live site 真值，静态区用传入值。"""
            if region_site is not None:
                try:
                    sid = int(env.mj_model.site_name2id(region_site))
                    c = np.asarray(env.get_site_pos(region_site), float)
                    xm = np.asarray(env.mj_data.site_xmat[sid],
                                    float).reshape(3, 3)
                    sz = np.asarray(env.mj_model.site_size[sid], float)
                    hf = np.abs(xm @ sz)[:2]
                    return c[:2], hf
                except Exception:
                    pass
            return tgt, half

        d_body = None      # 窗口起点 body→区 距离
        best_d = float("inf")
        for t in range(timeout):
            end = _eef(env, site)
            try:
                bpos = np.asarray(env.get_body_pos(body), float)
            except Exception:
                return {"success": False, "reason": "push_no_body",
                        "steps": t}
            c, hf = _region()
            d_vec = np.abs(bpos[:2] - c)
            in_region = bool(hf is not None
                             and d_vec[0] < hf[0] - 0.003
                             and d_vec[1] < hf[1] - 0.003)
            d = float(np.linalg.norm(bpos[:2] - c))
            best_d = min(best_d, d)
            if in_region:
                # 入区成功：抬爪退出，防后续步骤拖动物体
                for _ in range(15):
                    servo_step(env, site, end + [0, 0, 0.05], gripper=-1,
                               k=k, vcap=vcap, actor=actor)
                return {"success": True, "steps": t, "dist": d}
            # 接触守卫：TCP 沿推过方向越过物心 2cm → 物体绕爪滑脱
            to_tgt = tgt - bpos[:2]
            n = float(np.linalg.norm(to_tgt))
            if n > 1e-6 and float(np.dot(end[:2] - bpos[:2],
                                         to_tgt / n)) > 0.02:
                return {"success": False, "reason": "push_lost_contact",
                        "steps": t, "dist": d}
            # 伺服目标：区中心 + 过冲余量（推到底，防停在区口）
            aim = np.array([tgt[0], tgt[1], end[2]])
            u = to_tgt / n if n > 1e-6 else np.zeros(2)
            aim[:2] = tgt + u * overshoot
            servo_step(env, site, aim, gripper=-1, k=k, vcap=vcap,
                       actor=actor)
            if t % stall_window == 0:
                if d_body is not None and t >= stall_window \
                        and d_body - d < stall_progress:
                    return {"success": False, "reason": "push_stalled",
                            "steps": t, "dist": d, "best_dist": best_d}
                d_body = d
        return {"success": False, "reason": "push_timeout", "steps": timeout,
                "dist": best_d}


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
                timeout: int = 160, k: float = _DEFAULT_K, vcap: float = 1.0,
                carry_margin: float = 0.0
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
        # 绕障 waypoint：水平段与障碍膨胀 AABB 相交时，提升飞越（受 cap
        # 约束）或插侧向绕点。carry_margin 来自反思（stall 后逐步加大）。
        # 信息源：env.obstacle_bodies（auto_obstacles 清单）+ object_bounds，
        # 取不到任一信息时回退旧固定 L 形（行为不变）。
        wps = None
        try:
            from .avoidance import bounds_to_aabb, plan_carry_waypoints
            obs_names = getattr(env, "obstacle_bodies", None)
            if obs_names and hasattr(env, "object_bounds"):
                aabbs = []
                for nm in obs_names:
                    try:
                        aabbs.append(bounds_to_aabb(
                            env.object_bounds(nm), margin=0.03 + carry_margin))
                    except Exception:
                        continue
                if aabbs:
                    wps = plan_carry_waypoints(
                        end0, tgt, aabbs, cruise=cruise, hover=hover,
                        z_cap=float(phys_get(env, "carry_z_cap_m", 1.0)),
                        lateral_step=0.16, margin=0.03 + carry_margin)
        except Exception:
            wps = None
        if wps is None:
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
