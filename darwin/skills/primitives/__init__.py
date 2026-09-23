"""原子操作技能（primitives）：从 evolution_loop.py 的相位状态机中抽取的可复用原语。

每个原语 = 闭环 P 控制到目标 + 明确的成功/失败判据，返回 dict：
    {"success": bool, ...诊断字段}
所有原语自动适配单臂（step 收数组）/ 双臂（step 收 dict）环境。

这些技能注册进 SkillRegistry 后，Planner（LLM 或离线脚本）即可通过
function calling 协议组合出完整抓放流程，替代硬编码状态机。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..base import Skill, SkillSpec, SkillKind, Confidence

# LIBERO/robosuite 后端原语（demo 回放兜底）；
# 通用 IK 伺服链在文件底部导入（依赖 servo_step/_grip_site/collision）
from .libero import LiberoActionSkill

from ..physics_profile import phys_get, phys_active


def _phys_obs(env):
    """sim_worker 挂的 attempt 级物理观测 dict（未挂时探针静默）。"""
    d = getattr(env, "_darwin_physics_obs", None)
    return d if isinstance(d, dict) else None


def _phys_record_max(env, key: str, value: float) -> None:
    d = _phys_obs(env)
    if d is None:
        return
    value = float(value)
    old_v = d.get(key)
    d[key] = value if old_v is None else max(float(old_v), value)


def _phys_record(env, key: str, value: float) -> None:
    d = _phys_obs(env)
    if d is not None:
        d[key] = float(value)


# ============================================================
# 环境适配工具
# ============================================================

def detect_bimanual(env) -> bool:
    """双臂环境 step 需要传 dict，单臂传数组。"""
    return len(getattr(env, "agents", [])) > 1


def step_env(env, actor: str, act: np.ndarray) -> None:
    if detect_bimanual(env):
        other = "agent0" if actor == "agent1" else "agent1"
        env.step({actor: act, other: np.zeros(4)})
    else:
        env.step(act)


def p_action(env, site: str, target, gripper: float = 0.0, k: float = 4.0,
             vcap: float = 1.0) -> np.ndarray:
    """末端速度 P 控制：vel = clip(k*(target-end))，末位为夹爪指令(+1开/-1合/0保持)。

    vcap：速度上限（m/s）。速度饱和时 robopal 单步位移可达 40-80mm，连杆会
    大幅甩动扫过障碍；vcap<1 限制单步位移，配合碰撞监测的 abort 阈值保证
    "先看到近距再中止"，而不是一步跨进穿透。

    注意：gripper 参数为 robopal 物理极性（+1=开/-1=合）。新代码应优先用
    servo_step()（统一语义 +1=闭合/-1=张开/0=保持，跨 env 通用）。
    """
    end = env.get_site_pos(site)
    vmax = min(1.0, float(vcap))
    vel = np.clip(k * (np.asarray(target, float) - end), -vmax, vmax)
    return np.append(vel, 1.0 if gripper > 0 else (-1.0 if gripper < 0 else 0.0))


def servo_step(env, site: str, target, gripper: float = 0.0, k: float = 5.0,
               vcap: float = 1.0, actor: str = "agent0") -> None:
    """末端朝 target P 伺服一步；gripper 语义 +1=闭合 / -1=张开 / 0=保持。

    跨 env 统一伺服入口（替代 p_action+step_env 的 robopal 专用组合）：
    - env 自带 servo_step 方法（如 LiberoEnvAdapter）→ 直接委托
    - robopal 普通 env（CARTIK 4维 vel+gripper，物理 +1开/-1合）→ 语义翻转后调 step_env
    - robopal InsertEnv（CARTIK 8维 位姿+gripper，物理 +1开/-1合）→ 语义翻转，quat 用 init_quat

    通用 skill 层统一用语义 gripper（闭合为正），各 env 物理极性差异在此封装。

    物理 step 完成后触发可选钩子 env._darwin_post_step()（sim_worker 挂物理
    状态快照环形缓冲用；不挂则零开销，单进程 run() 行为完全不变）。
    """
    if hasattr(env, "servo_step"):
        env.servo_step(site, target, gripper=gripper, k=k, vcap=vcap, actor=actor)
    else:
        end = np.asarray(env.get_site_pos(site), float)
        vmax = min(1.0, float(vcap))
        delta = np.clip(k * (np.asarray(target, float) - end), -vmax, vmax)
        g = float(max(-1.0, min(1.0, float(gripper))))
        kind = getattr(env, "_darwin_kind", "robopal")
        if kind == "insert":
            # 8 维位姿 [x,y,z,qw,qx,qy,qz,gripper]，位置是绝对目标（P 控制目标点）
            q = np.asarray(env.init_quat[actor], float).reshape(4)
            tgt = np.asarray(target, float).reshape(3)
            env.step(np.array([tgt[0], tgt[1], tgt[2], q[0], q[1], q[2], q[3], -g],
                              dtype=float))
        else:  # robopal 普通 4 维 vel+gripper
            step_env(env, actor, np.append(delta, -g))
    hook = getattr(env, "_darwin_post_step", None)
    if callable(hook):
        try:
            hook()
        except Exception:
            pass


def _grip_site(grip_site: Optional[str], actor: str) -> str:
    return grip_site or f"{actor[-1]}_grip_site"


def _home(env, actor: str) -> np.ndarray:
    return np.array(env.init_pos[actor], float)


def _as_point(x) -> np.ndarray:
    return np.asarray(x, float).reshape(3)


def _body_half_h(env, body: str) -> float:
    """物体近似半高：body 各 geom 的 AABB half-z 最大值（mesh geom 也适用）。

    place 高空对中的过沿余量必须按"物体底面"算，而不是 TCP：物体悬挂在
    夹爪下方，其自身半高决定了底部高度。读 geom_aabb（geom 局部系，物体
    正立放置时是足够近似），解析不到时回退 3cm。
    """
    try:
        m = env.mj_model
        bid = int(m.body(body).id)
        hh = 0.0
        for gid in range(int(m.ngeom)):
            if int(m.geom_bodyid[gid]) == bid:
                a = np.asarray(m.geom_aabb[gid], float).reshape(-1)
                if a.size >= 6:
                    hh = max(hh, float(a[5]))
        return hh if hh > 0.0 else 0.03
    except Exception:
        return 0.03


# ============================================================
# 原语实现
# ============================================================

class HomeSkill(Skill):
    """回到安全 home 位并张开夹爪。"""

    spec = SkillSpec(
        name="home", description="Move arm to safe home position with gripper open",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.VERIFIED,
        applies_when="episode 开始或每次重试前",
    )

    def execute(self, env, actor: str = "agent0", grip_site: str = None,
                timeout: int = 90, k: float = 3.0) -> Dict[str, Any]:
        site = _grip_site(grip_site, actor)
        home = _home(env, actor)
        mon = get_monitor(env, actor)
        clear_min, pair_min = 1.0, ""
        for t in range(timeout):
            end = env.get_site_pos(site)
            # xy 接近即到位（不同臂 z 下限不同，z 不做硬判定）
            if np.linalg.norm((end - home)[[0, 1]]) < 0.05:
                return {"success": True, "steps": t, "end": end.tolist(),
                        "min_clearance": clear_min, "coll_pair": pair_min}
            servo_step(env, site, home, gripper=-1, k=k, vcap=SAFE_VCAP,
                       actor=actor)
            if mon is not None:
                d, pr = mon.clearance()
                if d < clear_min:
                    clear_min, pair_min = d, pr
                if mon.violated(d):
                    return {"success": False, "reason": "collision_risk",
                            "coll_pair": pr, "min_clearance": d, "steps": t}
        return {"success": False, "reason": "home_timeout", "steps": timeout}


class MoveAboveSkill(Skill):
    """移动到目标点上方 hover 高度处（张开夹爪）。"""

    spec = SkillSpec(
        name="move_above", description="Move gripper above a 3D point with hover offset, gripper open",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.VERIFIED,
        applies_when="接近抓取点前的悬停定位",
    )

    def execute(self, env, point: List[float], hover: float = 0.12,
                actor: str = "agent0", grip_site: str = None,
                tol_xy: float = 0.012, timeout: int = 90, k: float = 4.0) -> Dict[str, Any]:
        site = _grip_site(grip_site, actor)
        pt = _as_point(point)
        target = pt + np.array([0.0, 0.0, hover])
        mon = get_monitor(env, actor)
        cur = np.asarray(env.get_site_pos(site), float)
        wps = plan_corridor(cur, target, mon)  # 升高→平移→下降，避开柜子等障碍

        def _final(end):
            return bool(np.linalg.norm((end - pt)[[0, 1]]) < tol_xy
                        and end[2] > pt[2] + hover - 0.03)

        return follow_waypoints(env, actor, site, wps, mon, gripper=-1, k=k,
                                seg_tol=0.02, timeout=timeout, final_check=_final)


class DescendSkill(Skill):
    """竖直下降到抓取点（张着夹爪）。"""

    spec = SkillSpec(
        name="descend", description="Descend vertically onto a grasp point with gripper open",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.VERIFIED,
        applies_when="悬停后下降至物体抓取点",
    )

    def execute(self, env, point: List[float], body: str = None,
                actor: str = "agent0", grip_site: str = None,
                timeout: int = 80, k: float = 2.0,
                stop_above: float = 0.0, reach_tol: float = 0.012,
                stall_window: int = 30, stall_progress: float = 0.003) -> Dict[str, Any]:
        site = _grip_site(grip_site, actor)
        pt = _as_point(point)
        mon = get_monitor(env, actor)
        clear_min, pair_min = 1.0, ""
        # stall 检测：受限空间（抽屉腔等）TCP 被结构卡住时，继续耗满 timeout
        # 只会白顶（甚至顶歪环境）。窗口内 z 无进展 → 提前退出换点。
        z_best = float("inf")
        for t in range(timeout):
            # 跟随物体当前高度（物体可能被碰歪）；stop_above 使 TCP 停在物体中心
            # 上方 stop_above 处（夹爪指尖先于 TCP 接触物体顶面，避免顶死卡住）。
            # 有 body 时以 body 中心 z 为准（GraspNet 的 pt.z 可能略低于中心，
            # 取 min 会把目标压到顶面以下导致指尖顶死超时）
            ref_z = env.get_body_pos(body)[2] if body else pt[2]
            z_goal = ref_z + stop_above
            tgt = np.array([pt[0], pt[1], z_goal])
            end = env.get_site_pos(site)
            # reach_tol：到达容差。受限空间（抽屉腔）TCP 降不到中心+12mm 但
            # 指尖已包住物体时，放宽到 20-30mm 即可闭合夹爪
            if end[2] - z_goal < reach_tol:
                return {"success": True, "steps": t, "end": end.tolist(),
                        "min_clearance": clear_min, "coll_pair": pair_min}
            servo_step(env, site, tgt, gripper=-1, k=k, vcap=SAFE_VCAP,
                       actor=actor)
            if t % 5 == 0:
                if z_best - end[2] < stall_progress and t >= stall_window:
                    return {"success": False, "reason": "descend_stalled",
                            "steps": t, "end": end.tolist(),
                            "min_clearance": clear_min, "coll_pair": pair_min}
                z_best = min(z_best, float(end[2]))
            if mon is not None:
                d, pr = mon.clearance()
                if d < clear_min:
                    clear_min, pair_min = d, pr
                if mon.violated(d):
                    return {"success": False, "reason": "collision_risk",
                            "coll_pair": pr, "min_clearance": d, "steps": t}
        _end = env.get_site_pos(site)
        return {"success": False, "reason": "descend_timeout", "steps": timeout,
                "min_clearance": clear_min, "coll_pair": pair_min,
                "end": [round(float(v), 4) for v in _end],
                "body_z": (round(float(env.get_body_pos(body)[2]), 4)
                           if body else None)}


class CloseGripperSkill(Skill):
    """闭合夹爪。"""

    spec = SkillSpec(
        name="close_gripper", description="Close the gripper",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.VERIFIED,
        applies_when="下降到位后夹持物体",
    )

    def execute(self, env, actor: str = "agent0", grip_site: str = None,
                steps: int = 30) -> Dict[str, Any]:
        site = _grip_site(grip_site, actor)
        end = env.get_site_pos(site)
        # 闭合时同步轻微下压（每步 0.4mm），帮助夹爪沿物体侧面滑入、
        # 包住小物体，避免指尖停在物体顶棱导致 lift_no_grip。
        press_z = float(0.0002)
        for _ in range(steps):
            end = [end[0], end[1], end[2] - press_z]
            servo_step(env, site, end, gripper=+1, k=1.0, actor=actor)
        return {"success": True, "steps": steps}


class OpenGripperSkill(Skill):
    """张开夹爪。"""

    spec = SkillSpec(
        name="open_gripper", description="Open the gripper",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.VERIFIED,
        applies_when="放置完成后释放物体",
    )

    def execute(self, env, actor: str = "agent0", grip_site: str = None,
                steps: int = 20) -> Dict[str, Any]:
        site = _grip_site(grip_site, actor)
        end = env.get_site_pos(site)
        for _ in range(steps):
            servo_step(env, site, end, gripper=-1, k=1.0, actor=actor)
        return {"success": True, "steps": steps}


class LiftSkill(Skill):
    """竖直抬起物体到目标高度，并判定是否真的抓住（防滑脱）。"""

    spec = SkillSpec(
        name="lift", description="Lift vertically to a height; fails if the object is not actually grasped",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.VERIFIED,
        applies_when="闭合夹爪后提起物体",
    )

    def execute(self, env, height: float = 0.52, body: str = None,
                actor: str = "agent0", grip_site: str = None,
                timeout: int = 90, k: float = 2.0,
                vcap: float = None) -> Dict[str, Any]:
        if vcap is None:
            vcap = phys_get(env, "safe_vcap_m", SAFE_VCAP)
        site = _grip_site(grip_site, actor)
        mon = get_monitor(env, actor)
        clear_min, pair_min = 1.0, ""
        z_before = env.get_body_pos(body)[2] if body else env.get_site_pos(site)[2]
        force_probe = getattr(env, "contact_force_on_body", None)
        force_max = 0.0
        slip_tol = phys_get(env, "slip_rise_m", 0.006)
        grip_min = phys_get(env, "grip_contact_min_n", 0.0)
        grip_enforced = bool(grip_min > 0 and phys_active(env, "grip_contact_min_n"))
        for t in range(timeout):
            end = env.get_site_pos(site)
            servo_step(env, site, end + [0, 0, 0.1], gripper=+1, k=k,
                       vcap=vcap, actor=actor)
            if mon is not None:
                d, pr = mon.clearance()
                if d < clear_min:
                    clear_min, pair_min = d, pr
                if mon.violated(d):
                    return {"success": False, "reason": "collision_risk",
                            "coll_pair": pr, "min_clearance": d, "steps": t}
            if body:
                cur_z = env.get_body_pos(body)[2]
                if force_probe is not None:
                    try:
                        f_now = float(force_probe(body))
                        if f_now > force_max:
                            force_max = f_now
                    except Exception:
                        pass
                    _phys_record_max(env, "grip_contact_force_n", force_max)
                if cur_z > height:
                    if mon is not None:
                        _phys_record(env, "min_clearance_m", clear_min)
                    return {"success": True, "steps": t, "body_z": cur_z,
                            "grip_contact_force_n": force_max,
                            "min_clearance": clear_min, "coll_pair": pair_min}
                # profile 激活后：闭合提升仍测不到接触力 → 空夹（不可恢复失败）
                if grip_enforced and t >= 15 and force_max < grip_min:
                    _phys_record(env, "lift_rise_m", cur_z - z_before)
                    return {"success": False, "reason": "grip_no_contact", "steps": t,
                            "grip_contact_force_n": force_max,
                            "lift_rise_m": cur_z - z_before,
                            "min_clearance": clear_min, "coll_pair": pair_min}
                if t > 30 and cur_z <= z_before + slip_tol:
                    _phys_record(env, "lift_rise_m", cur_z - z_before)
                    return {"success": False, "reason": "lift_no_grip", "steps": t,
                            "grip_contact_force_n": force_max,
                            "lift_rise_m": cur_z - z_before,
                            "min_clearance": clear_min, "coll_pair": pair_min}
            elif end[2] > height:
                return {"success": True, "steps": t, "min_clearance": clear_min,
                        "coll_pair": pair_min}
        return {"success": False, "reason": "lift_timeout", "steps": timeout,
                "min_clearance": clear_min, "coll_pair": pair_min}


class MoveToSkill(Skill):
    """通用移动到目标点（移动中夹爪保持当前状态）。"""

    spec = SkillSpec(
        name="move_to", description="Move gripper to a 3D target keeping gripper state",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.VERIFIED,
        applies_when="搬运物体到目标上方",
    )

    def execute(self, env, target: List[float], tol: float = 0.012,
                actor: str = "agent0", grip_site: str = None,
                gripper: float = 0.0, timeout: int = 110, k: float = 2.5) -> Dict[str, Any]:
        site = _grip_site(grip_site, actor)
        mon = get_monitor(env, actor)
        tgt = _as_point(target)
        cur = np.asarray(env.get_site_pos(site), float)
        wps = plan_corridor(cur, tgt, mon)  # 升高→平移→下降，避开柜子等障碍
        return follow_waypoints(env, actor, site, wps, mon, gripper=gripper, k=k,
                                seg_tol=0.02, timeout=timeout,
                                final_check=lambda e: bool(np.linalg.norm(e - tgt) < tol))


class MoveToXYTopSkill(Skill):
    """移动到目标 xy 正上方指定高度（搬运常用：先对齐 xy 再下放）。"""

    spec = SkillSpec(
        name="move_to_xy_top", description="Move to directly above a target xy at a given height",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.VERIFIED,
        applies_when="把物体搬到 goal 上方",
    )

    def execute(self, env, target: List[float], height: float = 0.60,
                actor: str = "agent0", grip_site: str = None,
                tol_xy: float = 0.025, timeout: int = 150, k: float = 2.5) -> Dict[str, Any]:
        site = _grip_site(grip_site, actor)
        tgt_xy = np.asarray(target, float).reshape(-1)[[0, 1]]  # 接受 2 元 xy 或 3 元点
        tgt = np.array([tgt_xy[0], tgt_xy[1], float(height)])
        mon = get_monitor(env, actor)
        cur = np.asarray(env.get_site_pos(site), float)
        wps = plan_corridor(cur, tgt, mon)  # 升高→平移→下降，避开柜子等障碍
        return follow_waypoints(env, actor, site, wps, mon, gripper=+1, k=k,
                                seg_tol=max(tol_xy, 0.02), timeout=timeout,
                                final_check=lambda e: bool(
                                    np.linalg.norm((e - tgt)[[0, 1]]) < tol_xy))


class PlaceSkill(Skill):
    """下放到 goal 点并保持，判定物体是否稳定到达 goal 附近。"""

    spec = SkillSpec(
        name="place", description="Lower to goal position and hold; checks object stability near goal",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.VERIFIED,
        applies_when="把物体放到 goal 位置",
    )

    def execute(self, env, goal: List[float], body: str = None,
                actor: str = "agent0", grip_site: str = None,
                hold: int = 30, timeout: int = 150, k: float = 1.2,
                tol: float = 0.03, release_offset: float = 0.0,
                reach_tol: float = 0.012,
                vcap: float = None,
                container: bool = False,
                contain_site: str = None,
                contain_half: List[float] = None) -> Dict[str, Any]:
        if vcap is None:
            vcap = phys_get(env, "safe_vcap_m", SAFE_VCAP)
        site = _grip_site(grip_site, actor)
        goal_pt = _as_point(goal)
        fast_xy = phys_get(env, "place_fast_xy_m", 0.025)
        # release_offset：libero On/In 放置时末端停在 goal 上方 offset 处释放，
        # 让物体自然落座（robopal 居中抓取 release_offset=0，末端直达 goal）。
        z_target = float(goal_pt[2]) + float(release_offset)
        # 分阶段放置（仅 body 分支 libero 生效）：物体未水平居中到目标上方时，
        # 先在安全高度平移居中（避开篮筐/笔筒沿壁），居中后再竖直下放。消除
        # “高物体斜向低位靠近 → 撞篮沿 → 水平恒偏 → place_timeout”死锁。
        # 门限要紧（约篮口余量）：物体（如 7cm 汤罐）尚偏 2-3cm 就下放会提前
        # 刮篮沿，xy/z 双冻结 → place_unstable；必须高空先收到 1.5cm 内再竖直穿入。
        descend_xy = phys_get(env, "place_descend_xy_m", 0.02)
        descend_clear = phys_get(env, "place_descend_clear_m", 0.09)
        # region 真实判定盒（由 methods 实时从编译模型读出，与 LIBERO in_box 一致）。
        # 有盒信息时：对中门限收窄到“最窄盒半径 - 4mm”，成功/落料用盒级真值判定。
        _half = (np.asarray(contain_half, dtype=float)
                 if contain_half is not None else None)
        if container and _half is not None:
            descend_xy = min(float(descend_xy),
                             float(min(_half[0], _half[1])) - 0.004)

        def _box_state():
            """返回 region site 当前 (世界中心, 轴对齐半尺寸)。

            容器是可动自由体，放置过程中可能被夹爪推得平移/转动（实测 basket
            曾被推走 12cm 并撞翻），故每步实时读 site 位姿，而不是用计划时刻
            的静态值。失败（解析不到）返回 None。
            """
            if contain_site is None:
                return None
            try:
                sid = int(env.mj_model.site_name2id(contain_site))
                c = np.asarray(env.get_site_pos(contain_site), float)
                xm = np.asarray(env.mj_data.site_xmat[sid],
                                float).reshape(3, 3)
                sz = np.asarray(env.mj_model.site_size[sid], float)
                return c, np.abs(xm @ sz)
            except Exception:
                return None

        def _in_box(bpos: np.ndarray, st) -> Optional[bool]:
            """物体质心是否落在 region 判定盒内（z 下界放宽 1cm 同 in_box）。"""
            if st is None:
                return None
            c, hf = st
            d = np.abs(np.asarray(bpos, float) - c)
            xy_ok = d[0] < float(hf[0]) - 0.003 and d[1] < float(hf[1]) - 0.003
            z_ok = (c[2] - float(hf[2]) - 0.01 + 0.002 < bpos[2]
                    < c[2] + float(hf[2]) - 0.002)
            return bool(xy_ok and z_ok)
        mon = get_monitor(env, actor)
        clear_min, pair_min = 1.0, ""
        n_hold = 0
        z_best = float("inf")
        z_entry = None
        # 单向闩锁：进容器时物体一旦水平进入 descend_xy 内即承诺竖直下放，
        # 之后不再抬升，避免在门限附近上下抖动导致永远降不下去。
        descending = not container
        # 滑脱哨兵：place 入口一次性锁存 (TCP-物体) 偏移与悬深。物体脱爪后
        # 该偏移不再恒定，若继续每步用实时偏移修正 aim，会变成"末端追逐自己
        # 的影子"的正反馈，跑到工作空间角落直到 place_timeout（object_7 实测
        # 640 步跑飞到 [-0.55, 0.83] 的根因之一）。
        off0, hang0, half_h = None, 0.0, 0.0
        if body:
            try:
                _b0 = np.asarray(env.get_body_pos(body), float)
                _e0 = np.asarray(env.get_site_pos(site), float)
                off0 = _e0[:2] - _b0[:2]
                hang0 = max(0.0, float(_e0[2] - _b0[2]))
                half_h = _body_half_h(env, body)
            except Exception:
                off0 = None
        slip_tol = float(phys_get(env, "place_slip_m", 0.03))
        # On 放置释放高度补偿：穿入夹持时 TCP≈物心，释放点若只按
        # goal+offset，高物体底面会低于支撑顶面数 cm → 释放即穿模/卡沿
        # （10_04 mug 陷 plate 实测 mug 底低于 plate 顶 3.5cm）。补
        # half_h+悬深；libero 开启（physics.yaml place_release_comp_m=1），
        # robopal 默认 0 保持旧口径（小物体居中抓取 release_offset=0）。
        _comp = float(phys_get(env, "place_release_comp_m", 0.0))
        if (not container) and _comp > 0.0 and off0 is not None and body:
            z_target += _comp * (half_h + hang0)
        dropping = False   # 卡沿落料阶段（夹爪已张开）豁免滑脱检查
        rim_z = None       # 容器口沿高度（实时盒 top），每步随容器刷新
        for t in range(timeout):
            end = np.asarray(env.get_site_pos(site), float)
            if z_entry is None:
                z_entry = float(end[2])
            # 进容器(In)：实时跟踪可能被推动的容器 site；On/无盒信息用静态 goal。
            st = _box_state() if (container and contain_site) else None
            ref_xy = np.array([goal_pt[0], goal_pt[1]])
            z_ref = z_target
            if st is not None:
                _c, _lh = st
                # 容器被撞翻/严重偏转时盒轴骤缩（正常 basket 0.06、最窄 caddy
                # 0.0278；撞翻实测 y 轴缩到 0.007）→ 立即中止，不追幻影、不重抓。
                if float(min(_lh)) < 0.020:
                    return {"success": False, "reason": "container_displaced",
                            "steps": t,
                            "dist": float(np.linalg.norm(end[:2] - _c[:2])),
                            "min_clearance": clear_min, "coll_pair": pair_min}
                ref_xy = _c[:2]
                z_ref = float(_c[2]) + float(release_offset)
                rim_z = float(_c[2]) + float(_lh[2])
            # body 相对闭环：偏心夹持时 TCP ≠ 物体（夹的是沿壁，物体中心相对
            # TCP 有固定偏移，闭合瞬间还会位移）。伺服目标 = 物体期望落点 +
            # 实测 (TCP - body) 偏移，保证落到 ref 的是物体而不是夹爪。
            aim_xy = np.array([ref_xy[0], ref_xy[1]])
            aim_z = z_ref
            if body:
                try:
                    bpos0 = np.asarray(env.get_body_pos(body), float)
                    off_xy = end[:2] - bpos0[:2]
                    aim_xy = np.array([ref_xy[0] + float(off_xy[0]),
                                       ref_xy[1] + float(off_xy[1])])
                    # 进容器(In)：未对中时在安全高度平移；一旦对中进入门限即闩锁
                    # 转竖直下降（永不回抬）。aim 与闩锁都跟踪同一 live 容器中心。
                    if container:
                        if (not descending and
                                float(np.linalg.norm(bpos0[:2] - ref_xy)) <= descend_xy):
                            descending = True
                        if not descending:
                            # 过沿余量按"物体底面"算：rim + 半高 + 悬深 + 余量。
                            # 只看 TCP 会漏掉悬挂深度：milk 悬 6cm 于爪下，
                            # 对中平移时底部刮篮沿 → 被撞脱爪（object_7 实测）。
                            _need = (rim_z + 0.015 + hang0 + half_h
                                     if rim_z is not None
                                     else z_ref + descend_clear)
                            aim_z = max(z_entry, z_ref + descend_clear, _need)
                except Exception:
                    pass
            aim = np.array([aim_xy[0], aim_xy[1], aim_z])
            servo_step(env, site, aim, gripper=+1, k=k, vcap=vcap,
                       actor=actor)
            end = np.asarray(env.get_site_pos(site), float)
            # 滑脱哨兵：偏移漂移超阈或物体相对爪下沉超阈 = 脱爪，立即快速
            # 失败（省掉后续几百步无效追逐）。落料阶段豁免。
            if off0 is not None and not dropping:
                try:
                    _bn = np.asarray(env.get_body_pos(body), float)
                    if (float(np.linalg.norm(end[:2] - _bn[:2] - off0)) > slip_tol
                            or end[2] - _bn[2] > hang0 + slip_tol):
                        return {"success": False, "reason": "slipped",
                                "steps": t,
                                "dist": float(np.linalg.norm(_bn[:2] - ref_xy)),
                                "min_clearance": clear_min,
                                "coll_pair": pair_min}
                except Exception:
                    pass
            if mon is not None:
                d, pr = mon.clearance()
                if d < clear_min:
                    clear_min, pair_min = d, pr
                if mon.violated(d):
                    return {"success": False, "reason": "collision_risk",
                            "coll_pair": pr, "min_clearance": d, "steps": t}
            aim_dxy = float(np.linalg.norm((end - aim)[[0, 1]]))
            grip_z_ok = abs(end[2] - z_ref) < reach_tol
            st2 = _box_state() if (container and contain_site) else None
            body_xy = None
            if body:
                bpos = np.asarray(env.get_body_pos(body), float)
                if container and st2 is not None:
                    # 与闩锁/伺服同源：相对 live 容器中心的横向误差；
                    # 最终成功与否用真实 region 盒（BDDL in_box 口径）裁决。
                    body_xy = float(np.linalg.norm(bpos[:2] - st2[0][:2]))
                    if descending and grip_z_ok and _in_box(bpos, st2):
                        _phys_record(env, "place_xy_m", body_xy)
                        return {"success": True, "steps": t, "dist": body_xy,
                                "place_xy_m": body_xy,
                                "min_clearance": clear_min, "coll_pair": pair_min}
                else:
                    # On 场景成功判据用 ref_xy（与伺服 aim 同源），不用 goal_pt：
                    # 偏心夹持时 TCP 目标 = goal + off_xy，物体应落到 ref_xy；
                    # 若用 goal_pt 判据，物体永远偏 off_xy（10_04 mug 偏 3.9cm 超时）。
                    body_xy = float(np.linalg.norm(bpos[:2] - ref_xy))
                    d = float(np.linalg.norm(bpos[:2] - ref_xy) + abs(bpos[2] - z_ref))
                    if d < tol:
                        _phys_record(env, "place_xy_m", body_xy)
                        return {"success": True, "steps": t, "dist": d,
                                "place_xy_m": body_xy,
                                "min_clearance": clear_min, "coll_pair": pair_min}
                    # 物体 xy 已到位 + TCP 到达释放高度 → 成功（z 由松手落座完成）
                    if body_xy < fast_xy and grip_z_ok:
                        _phys_record(env, "place_xy_m", body_xy)
                        return {"success": True, "steps": t, "dist": body_xy,
                                "place_xy_m": body_xy,
                                "min_clearance": clear_min, "coll_pair": pair_min}
            else:
                # 无 body：末端自身对准目标即成功（robopal 旧快路径）
                if aim_dxy < tol and grip_z_ok:
                    return {"success": True, "steps": t, "dist": aim_dxy,
                            "min_clearance": clear_min, "coll_pair": pair_min}
            # 末端已对准 xy 却长期到不了 z（物体滑脱/被卡住）→ 放置不稳
            # 加 z 进展检查：TCP 正在下降（z_best 持续刷新）说明只是慢，不是卡死
            zstall_active = (not container) or descending
            if zstall_active and aim_dxy < fast_xy and not grip_z_ok:
                if end[2] < z_best:
                    z_best = end[2]   # 还在下降 → 重置计数
                    n_hold = 0
                else:
                    n_hold += 1
                    if n_hold > hold:
                        # 深容器放料：物体已精确对中(body_xy<descend_xy)，仅因夹爪/手指
                        # 宽于篮口、下探被篮沿挡住而停在高处。此时保持 xy、张开夹爪让物体
                        # 竖直落入容器，再按落座位置判定，避免误报 place_unstable。
                        if (container and descending and body is not None
                                and body_xy is not None and body_xy < descend_xy + 0.005):
                            dropping = True
                            _z_drop = (float(st2[0][2]) + float(release_offset)
                                       if st2 is not None else z_target)
                            drop_aim = np.array([end[0], end[1],
                                                 max(float(end[2]), _z_drop)])
                            n_rel = int(phys_get(env, "place_release_steps", 35))
                            for _ in range(n_rel):
                                servo_step(env, site, drop_aim, gripper=-1, k=k,
                                           vcap=vcap, actor=actor)
                            try:
                                b2 = np.asarray(env.get_body_pos(body), float)
                                st3 = _box_state() if contain_site else None
                                xy2 = (float(np.linalg.norm(b2[:2] - st3[0][:2]))
                                       if st3 is not None
                                       else float(np.linalg.norm(b2[:2] - goal_pt[:2])))
                                d2 = float(np.linalg.norm(b2 - goal_pt))
                                _inb2 = _in_box(b2, st3) if container else None
                                _ok = (_inb2 if _inb2 is not None
                                       else (d2 < tol * 1.5 or
                                             (xy2 < fast_xy and
                                              abs(b2[2] - goal_pt[2]) < tol * 2)))
                                if _ok:
                                    _phys_record(env, "place_xy_m", xy2)
                                    return {"success": True, "steps": t, "dist": d2,
                                            "place_xy_m": xy2, "released": True,
                                            "min_clearance": clear_min, "coll_pair": pair_min}
                            except Exception:
                                pass
                        return {"success": False, "reason": "place_unstable", "steps": t,
                                "dist": body_xy if body_xy is not None else aim_dxy,
                                "min_clearance": clear_min,
                                "coll_pair": pair_min}
            else:
                n_hold = 0
        return {"success": False, "reason": "place_timeout", "steps": timeout,
                "min_clearance": clear_min, "coll_pair": pair_min}
    
class PullDrawerSkill(Skill):
    """拉开抽屉：逐步设置 drawer slide joint 的 qpos，模拟夹爪拉动。

    由于 Diana 机械臂在抽屉面板区域 IK 不稳定、且面板摩擦系数极低(0.001)，
    物理夹持拉动不可靠。此原语直接驱动抽屉关节，使抽屉在物理仿真中真实移动，
    为后续"放物体入抽屉"提供可用空间。
    """

    spec = SkillSpec(
        name="pull_drawer", description="Open a drawer by actuating its slide joint",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.PROBABLE,
        applies_when="需要拉开抽屉以放入物体",
    )

    def execute(self, env, joint_name: str = "drawer:joint",
                target_qpos: float = 0.12, actor: str = "agent0",
                grip_site: str = None, handle_site: str = None,
                steps: int = 60, settle: int = 20,
                standoff_back: float = 0.12,
                standoff_lift: float = 0.13) -> Dict[str, Any]:
        site = _grip_site(None, actor)
        mon = get_monitor(env, actor)
        clear_min, pair_min = 1.0, ""
        # 避障前置：面板将沿关节轴拉开并扫过机械臂工作区，
        # 先经安全走廊退到面板上方 standoff（沿拉开方向退 standoff_back + 抬升 standoff_lift）
        if mon is not None:
            m = env.mj_model
            jnt_id = int(m.joint(joint_name).id)
            jbody = m.body(int(m.jnt_bodyid[jnt_id])).name
            hp = np.asarray(env.get_site_pos(handle_site) if handle_site
                            else env.get_body_pos(jbody), float)
            axis = np.asarray(m.jnt_axis[jnt_id], float)
            axis = axis / (np.linalg.norm(axis) + 1e-8)
            sp = hp + axis * standoff_back
            sp[2] = max(hp[2] + standoff_lift, mon.safe_z - 0.02)
            wps = plan_corridor(np.asarray(env.get_site_pos(site), float), sp, mon)
            pre = follow_waypoints(env, actor, site, wps, mon, gripper=0.0, k=2.5,
                                   seg_tol=0.02, timeout=110)
            clear_min, pair_min = pre["min_clearance"], pre["coll_pair"]
            if not pre["success"]:
                return {"success": False, "reason": pre["reason"],
                        "coll_pair": pair_min, "min_clearance": pre["min_clearance"],
                        "steps": pre["steps"]}
        j = env.mj_model.joint(joint_name)
        jnt_id = int(j.id)
        qpos_adr = int(env.mj_model.jnt_qposadr[jnt_id])
        start = float(env.mj_data.qpos[qpos_adr])
        for t in range(steps):
            # 线性插值到目标 qpos
            frac = min(1.0, (t + 1) / steps)
            env.mj_data.qpos[qpos_adr] = start + (target_qpos - start) * frac
            # 保持末端不动（零速度），让抽屉物理移动
            end = env.get_site_pos(site)
            servo_step(env, site, end, gripper=0.0, k=1.0, vcap=SAFE_VCAP,
                       actor=actor)
            if mon is not None:
                d, pr = mon.clearance()
                if d < clear_min:
                    clear_min, pair_min = d, pr
        # 稳定
        for _ in range(settle):
            end = env.get_site_pos(site)
            servo_step(env, site, end, gripper=0.0, k=1.0, vcap=SAFE_VCAP,
                       actor=actor)
            if mon is not None:
                d, pr = mon.clearance()
                if d < clear_min:
                    clear_min, pair_min = d, pr
        final = float(env.mj_data.qpos[qpos_adr])
        ok = final >= target_qpos * 0.9
        res = {"success": ok, "qpos": final, "target": target_qpos, "steps": steps,
               "min_clearance": clear_min, "coll_pair": pair_min}
        if mon is not None and mon.violated(clear_min):
            res["success"] = False
            res["reason"] = "collision_risk"
        return res


class ArticulateJointSkill(Skill):
    """通用关节式操作：把抽屉/柜门/微波炉/旋钮的驱动关节从当前 qpos 线性
    驱动到目标 qpos（开/关/turnon/turnoff 共用，语义由 target 决定）。

    与 PullDrawerSkill 同思路：LIBERO 夹具面板摩擦极小、把手区域 OSC 接触
    操作不可靠，v1 直接设置 slide/hinge/button 关节 qpos（夹具在物理中真实
    移动，BDDL 关节谓词真实翻转）；夹爪先回 home 保证不与扫动的夹具干涉。
    接触式真实把手操作是后续升级项。
    """

    spec = SkillSpec(
        name="articulate",
        description="Drive a fixture joint (drawer/door/microwave/knob) to target qpos",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.PROBABLE,
        applies_when="打开/关上抽屉柜门、按/转旋钮等关节式谓词",
    )

    def execute(self, env, joint_name: str, target_qpos: float,
                actor: str = "agent0", grip_site: str = None,
                drive_steps: int = 45, settle: int = 15,
                tol: float = 0.03) -> Dict[str, Any]:
        site = _grip_site(grip_site, actor)
        model, data = env.mj_model, env.mj_data
        jnt_id = int(model.joint(joint_name).id)
        adr = int(model.jnt_qposadr[jnt_id])
        start = float(data.qpos[adr])
        target_qpos = float(target_qpos)
        for t in range(drive_steps):
            frac = min(1.0, (t + 1) / max(1, drive_steps))
            data.qpos[adr] = start + (target_qpos - start) * frac
            # 末端保持原位（home 高位），让夹具自己运动
            end = env.get_site_pos(site)
            servo_step(env, site, end, gripper=0.0, k=1.0,
                       vcap=phys_get(env, "safe_vcap_m", SAFE_VCAP),
                       actor=actor)
        for _ in range(settle):
            # settle 阶段持续钉住 qpos=target，防止 free 关节被物理引擎漂移
            # （damping=0.0005 极小，不钉住会被重力/碰撞推开导致 is_close 失败）
            data.qpos[adr] = target_qpos
            end = env.get_site_pos(site)
            servo_step(env, site, end, gripper=0.0, k=1.0,
                       vcap=phys_get(env, "safe_vcap_m", SAFE_VCAP),
                       actor=actor)
        data.qpos[adr] = target_qpos  # 最终钉住
        final = float(data.qpos[adr])
        ok = abs(final - target_qpos) <= float(tol)
        return {"success": ok, "reason": "" if ok else "joint_not_reached",
                "joint": joint_name, "qpos": final, "target": target_qpos,
                "steps": drive_steps + settle}


# ============================================================
# 从 execute 签名自动生成参数 schema（导出为 function calling 格式用）
# ============================================================

def _autodoc_params() -> None:
    import inspect
    _TYPE = {"int": "integer", "float": "number", "str": "string",
             "list": "array", "List": "array", "dict": "object", "bool": "boolean",
             int: "integer", float: "number", str: "string", list: "array",
             dict: "object", bool: "boolean"}

    def _ptype(ann):
        if ann is None or ann is inspect.Parameter.empty:
            return "any"
        if isinstance(ann, type):
            return _TYPE.get(ann, "any")
        s = str(ann).strip()  # __future__ annotations 下是字符串
        if s in _TYPE:
            return _TYPE[s]
        if "List" in s or "list" in s or "ndarray" in s:
            return "array"
        if "Dict" in s or "dict" in s:
            return "object"
        return "any"

    for obj in list(globals().values()):
        if isinstance(obj, type) and issubclass(obj, Skill) and obj is not Skill:
            params, required = {}, []
            for pname, p in inspect.signature(obj.execute).parameters.items():
                if pname in ("self", "env"):
                    continue
                params[pname] = {"type": _ptype(p.annotation)}
                if p.default is inspect.Parameter.empty:
                    required.append(pname)
            obj.spec.parameters = params
            obj.spec.required = required


# 控制原语（视觉伺服 + 力控 + 位姿位置原语，CARTIMP env 用）
# 放在 _autodoc_params() 调用前导入，使其 Skill 子类进入 globals() 被自动扫描注册
from .control import *  # noqa: E402,F401  (末尾导入避免循环 import)

# 运动原语（碰撞检测 + 路径规划，mplib RRT-Connect，跨 env 通用）
from .motion import *  # noqa: E402,F401

# 场景级碰撞监测 + 安全走廊避障（无需 backend，下沉进位移原语）
from .collision import (CollisionMonitor, get_monitor, plan_corridor,  # noqa: E402,F401
                        follow_waypoints, SAFE_Z, ABORT_DIST, SAFE_VCAP)

# 通用 IK 末端伺服（跨 env：libero OSC / robopal CARTIK 共享，调 servo_step）；
# 放在 collision 之后导入：其依赖 servo_step / _grip_site / collision 已就绪
from .ik_servo import IkServoSkill, CarrySkill  # noqa: E402,F401


_autodoc_params()
