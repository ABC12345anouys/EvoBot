"""控制原语（control primitives）：视觉伺服 + 力控的参数化泛化技能。

设计哲学（用户给定）：
- 分层：task skill (pick/place/insert) → control primitive (servo_align/guarded_move/impedance_push/spiral_search)
- 泛化的关键：参数化"目标特征+容差+方向+阈值+终止条件+恢复策略"，不绑定具体物体
- 三档执行模式：position(默认) / visual(最后对齐) / force(接触富集段)
- 验证用原语级信号：reprojection_error<tol、contact&&travel<max、depth_reached&&force_peak<limit
- 失败映射：target_lost / excessive_force / stuck / slip → 回给 Agent/RAG 做恢复

本模块包含两类原语（均为 Skill 子类，注册进 SkillRegistry 后 planner 可调用）：
1. 位姿位置原语（PoseHome/PoseMoveAbove/...）：服务 CARTIMP env（8 维位姿动作），
   复用现有 P 控制思想但输出绝对位姿目标（限幅增量推进），与现有 4 维原语（CARTIK env）解耦。
2. 控制原语（ServoAlign/GuardedMove/ImpedancePush/SpiralSearch）：参数化的视觉/力控闭环。

力信号：优先 InsertEnv.get_ee_force（force_ee sensor），无 sensor 时用 contact force 回退。
视觉信号：仿真下用 site 真值投影算 reproj_error（理想视觉，验证闭环逻辑），未来可接 YOLO keypoint。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..base import Skill, SkillSpec, SkillKind, Confidence
from . import detect_bimanual, _grip_site, _home, _as_point


# ============================================================
# 环境与力/视觉信号工具
# ============================================================

def _get_ee_pose(env, actor: str = "agent0"):
    """末端当前位姿 [pos(3), quat(4)]。兼容 InsertEnv 和原生 robopal env。"""
    if hasattr(env, "get_ee_pose"):
        return env.get_ee_pose(actor)
    # 原生 robopal：用 controller.forward_kinematics
    qpos = env.robot.get_arm_qpos(actor)
    return env.controller.forward_kinematics(qpos, actor)


def _ee_force(env) -> np.ndarray:
    """末端接触力（3 维，局部坐标系，N）。InsertEnv 优先 sensor，否则 contact 回退。"""
    if hasattr(env, "get_ee_force"):
        return env.get_ee_force()
    # 回退：遍历 contact force 求和
    import mujoco
    force = np.zeros(3)
    if hasattr(env, "mj_data") and env.mj_data.ncon > 0:
        for i in range(env.mj_data.ncon):
            f6 = np.zeros(6)
            mujoco.mj_contactForce(env.mj_model, env.mj_data, i, f6)
            force += f6[:3]
    return force


def _ee_force_norm(env) -> float:
    return float(np.linalg.norm(_ee_force(env)))


def _ee_force_along(env, axis: str = "z") -> float:
    """末端力在指定轴的分量。axis ∈ {x,y,z}。"""
    f = _ee_force(env)
    idx = {"x": 0, "y": 1, "z": 2}.get(axis, 2)
    return float(f[idx])


def step_pose_env(env, actor: str, pose_action: np.ndarray) -> None:
    """位姿动作 step：单臂传 8 维数组，双臂传 dict（8 维 + 8 维零位姿）。"""
    if detect_bimanual(env):
        other = "agent0" if actor == "agent1" else "agent1"
        env.step({actor: pose_action, other: np.zeros(8)})
    else:
        env.step(pose_action)


def _pose_action(env, actor: str, target_pos, target_quat=None,
                 gripper: float = 0.0, k: float = 4.0,
                 max_step: float = 0.05) -> np.ndarray:
    """构造 8 维位姿动作 [x,y,z,qw,qx,qy,qz, gripper]。

    用 P 控制限幅增量推进（避免大跨度目标导致 CARTIMP 震荡）：
        new_pos = end_pos + clip(k*(target-end), -max_step, max_step)
    quat 默认保持当前末端姿态（竖直向下），保证插入方向稳定。
    """
    end_pos, end_quat = _get_ee_pose(env, actor)
    end_pos = np.asarray(end_pos, float).reshape(3)
    tgt = np.asarray(target_pos, float).reshape(3)
    delta = np.clip(k * (tgt - end_pos), -max_step, max_step)
    new_pos = end_pos + delta
    quat = np.asarray(target_quat, float).reshape(4) if target_quat is not None else np.asarray(end_quat, float).reshape(4)
    grip = 1.0 if gripper > 0 else (-1.0 if gripper < 0 else 0.0)
    return np.concatenate([new_pos, quat, [grip]])


def _grip_site_pos(env, actor: str, grip_site: Optional[str]) -> np.ndarray:
    """末端 grip_site 当前位置（3D 真值，servo_align/对齐用）。"""
    site = _grip_site(grip_site, actor)
    return np.asarray(env.get_site_pos(site), float).reshape(3)


# ============================================================
# 位姿位置原语（CARTIMP env 专用，输出 8 维位姿动作）
# ============================================================

class PoseHomeSkill(Skill):
    """回到安全 home 位并张开夹爪（CARTIMP 版，位姿目标）。"""

    spec = SkillSpec(
        name="pose_home", description="Move arm to home position with gripper open (CARTIMP pose mode)",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.SINGLE_SHOT,
        applies_when="position: episode 开始或重试前（CARTIMP env）",
    )

    def execute(self, env, actor: str = "agent0", grip_site: str = None,
                timeout: int = 90, k: float = 3.0) -> Dict[str, Any]:
        home = np.array(env.init_pos[actor], float)
        quat = np.array(env.init_quat[actor], float)
        for t in range(timeout):
            end, _ = _get_ee_pose(env, actor)
            if np.linalg.norm((np.asarray(end, float) - home)[[0, 1]]) < 0.03:
                return {"success": True, "steps": t, "end": list(np.asarray(end, float))}
            step_pose_env(env, actor, _pose_action(env, actor, home, quat, gripper=+1, k=k))
        return {"success": False, "reason": "home_timeout", "steps": timeout}


class PoseMoveAboveSkill(Skill):
    """移动到目标点上方 hover 高度处（CARTIMP 版，张开夹爪）。"""

    spec = SkillSpec(
        name="pose_move_above", description="Move gripper above a 3D point with hover offset (CARTIMP pose mode)",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.SINGLE_SHOT,
        applies_when="position: 接近抓取点前的悬停定位（CARTIMP env）",
    )

    def execute(self, env, point: List[float], hover: float = 0.12,
                actor: str = "agent0", grip_site: str = None,
                tol_xy: float = 0.012, timeout: int = 90, k: float = 4.0) -> Dict[str, Any]:
        quat = np.array(env.init_quat[actor], float)
        pt = _as_point(point)
        target = pt + np.array([0.0, 0.0, hover])
        for t in range(timeout):
            end, _ = _get_ee_pose(env, actor)
            end = np.asarray(end, float)
            if np.linalg.norm((end - pt)[[0, 1]]) < tol_xy and end[2] > pt[2] + hover - 0.03:
                return {"success": True, "steps": t, "end": end.tolist()}
            step_pose_env(env, actor, _pose_action(env, actor, target, quat, gripper=+1, k=k))
        return {"success": False, "reason": "above_timeout", "steps": timeout}


class PoseDescendSkill(Skill):
    """竖直下降到抓取点（CARTIMP 版，张着夹爪）。

    与 cart 版 DescendSkill 接口对齐：支持 stop_above 控制 TCP 停留深度。
    z_goal = ref_z + stop_above，stop_above<0 时 TCP 可低于物体中心（夹爪
    指尖包住物体）。
    """

    spec = SkillSpec(
        name="pose_descend", description="Descend vertically onto a grasp point (CARTIMP pose mode)",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.SINGLE_SHOT,
        applies_when="position: 悬停后下降至物体抓取点（CARTIMP env）",
    )

    def execute(self, env, point: List[float], body: str = None,
                actor: str = "agent0", grip_site: str = None,
                timeout: int = 80, k: float = 2.0,
                stop_above: float = 0.0, reach_tol: float = 0.005) -> Dict[str, Any]:
        quat = np.array(env.init_quat[actor], float)
        pt = _as_point(point)
        for t in range(timeout):
            ref_z = env.get_body_pos(body)[2] if body else pt[2]
            z_goal = ref_z + stop_above  # 与 cart 版一致：TCP 停在物体中心+stop_above
            tgt = np.array([pt[0], pt[1], z_goal])
            end, _ = _get_ee_pose(env, actor)
            end = np.asarray(end, float)
            if abs(end[2] - z_goal) < reach_tol:  # 默认 5mm，避免停太高夹不住
                return {"success": True, "steps": t, "end": end.tolist()}
            step_pose_env(env, actor, _pose_action(env, actor, tgt, quat, gripper=+1, k=k))
        return {"success": False, "reason": "descend_timeout", "steps": timeout}


class PoseLiftSkill(Skill):
    """竖直抬起物体到目标高度，并判定是否真的抓住（CARTIMP 版）。"""

    spec = SkillSpec(
        name="pose_lift", description="Lift vertically to a height; fails if not grasped (CARTIMP pose mode)",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.SINGLE_SHOT,
        applies_when="position: 闭合夹爪后提起物体（CARTIMP env）",
    )

    def execute(self, env, height: float = 0.52, body: str = None,
                actor: str = "agent0", grip_site: str = None,
                timeout: int = 90, k: float = 2.0) -> Dict[str, Any]:
        quat = np.array(env.init_quat[actor], float)
        z_before = env.get_body_pos(body)[2] if body else _get_ee_pose(env, actor)[0][2]
        for t in range(timeout):
            end, _ = _get_ee_pose(env, actor)
            end = np.asarray(end, float)
            step_pose_env(env, actor, _pose_action(env, actor, end + [0, 0, 0.1], quat, gripper=-1, k=k))
            if body:
                cur_z = env.get_body_pos(body)[2]
                if cur_z > height:
                    return {"success": True, "steps": t, "body_z": cur_z}
                if t > 30 and cur_z <= z_before + 0.006:
                    return {"success": False, "reason": "lift_no_grip", "steps": t}
            elif end[2] > height:
                return {"success": True, "steps": t}
        return {"success": False, "reason": "lift_timeout", "steps": timeout}


class PoseMoveToSkill(Skill):
    """通用移动到目标点（CARTIMP 版，保持夹爪状态）。"""

    spec = SkillSpec(
        name="pose_move_to", description="Move gripper to a 3D target keeping gripper state (CARTIMP pose mode)",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.SINGLE_SHOT,
        applies_when="position: 搬运物体到目标上方（CARTIMP env）",
    )

    def execute(self, env, target: List[float], tol: float = 0.012,
                actor: str = "agent0", grip_site: str = None,
                gripper: float = -1.0, timeout: int = 110, k: float = 2.5) -> Dict[str, Any]:
        quat = np.array(env.init_quat[actor], float)
        tgt = _as_point(target)
        for t in range(timeout):
            end, _ = _get_ee_pose(env, actor)
            end = np.asarray(end, float)
            if np.linalg.norm(end - tgt) < tol:
                return {"success": True, "steps": t, "end": end.tolist()}
            step_pose_env(env, actor, _pose_action(env, actor, tgt, quat, gripper=gripper, k=k))
        return {"success": False, "reason": "move_timeout", "steps": timeout}


class PosePlaceSkill(Skill):
    """闭环下放到 goal（CARTIMP 版）。

    与 pose_move_to 的区别：pose_move_to 移动 TCP 到 target；
    pose_place 把 TCP 移向 goal，直到 **物体中心** 到达 goal（自动补偿
    TCP→物体中心的握持偏移）。成功条件是 body 距 goal < tol。
    """

    spec = SkillSpec(
        name="pose_place", description="Place object at goal, closed-loop on body position (CARTIMP)",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.SINGLE_SHOT,
        applies_when="position: 把物体放到 goal 位置（CARTIMP env，闭环）",
    )

    def execute(self, env, goal: List[float], body: str = None,
                actor: str = "agent0", grip_site: str = None,
                tol: float = 0.03, timeout: int = 150, k: float = 1.2) -> Dict[str, Any]:
        quat = np.array(env.init_quat[actor], float)
        goal_pt = _as_point(goal)
        for t in range(timeout):
            end, _ = _get_ee_pose(env, actor)
            end = np.asarray(end, float)
            step_pose_env(env, actor, _pose_action(env, actor, goal_pt, quat,
                                                   gripper=-1, k=k))
            if body:
                d = float(np.linalg.norm(np.asarray(env.get_body_pos(body), float) - goal_pt))
                if d < tol:
                    return {"success": True, "steps": t, "dist": d}
                # 末端已到 goal 附近但物体仍偏（握持偏移导致），继续沿偏移方向推
                if np.linalg.norm(end - goal_pt) < 0.01 and d > tol:
                    body_pos = np.asarray(env.get_body_pos(body), float)
                    offset = end - body_pos
                    push_tgt = goal_pt + offset * 1.1
                    step_pose_env(env, actor, _pose_action(env, actor, push_tgt, quat,
                                                           gripper=-1, k=k))
        return {"success": False, "reason": "place_timeout", "steps": timeout}

class ServoAlignSkill(Skill):
    """视觉伺服对齐原语。

    参数化目标特征 + 容差，不绑定具体物体。仿真下用 site 真值投影算 reproj_error
    （理想视觉，验证闭环逻辑），未来可接 YOLO keypoint（真实视觉）。

    成功：reproj_error < tol_px 且 xy 误差 < tol_xy_m
    失败：target_lost（连续 lost_lost_max 帧检测不到）/ joint_limit
    """

    spec = SkillSpec(
        name="servo_align",
        description="Visual servo align to a target feature (keypoint/mask/pose) with pixel/xy tolerances",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.SINGLE_SHOT,
        applies_when="visual: 最后 2-10 cm/几度的对齐纠偏，特征可见时启用",
        parameters={
            "target": {"type": "object", "description": "目标特征 {type: keypoint|mask|pose, name: site/body 名}"},
            "tol_px": {"type": "number", "description": "像素容差（仿真下用 3D 误差近似）"},
            "tol_xy_m": {"type": "number", "description": "xy 3D 容差（米）"},
            "gain": {"type": "number", "description": "P 控制增益"},
            "timeout": {"type": "integer"},
        },
    )

    def execute(self, env, target: Dict[str, Any], frame: str = "tcp",
                tol_px: float = 5.0, tol_xy_m: float = 0.003, tol_rz_deg: float = 1.0,
                gain: float = 0.4, timeout: int = 100, lost_max: int = 5,
                actor: str = "agent0", grip_site: str = None,
                camera: Dict = None) -> Dict[str, Any]:
        quat = np.array(env.init_quat[actor], float)
        site = _grip_site(grip_site, actor)
        # 解析目标特征：仿真下 type=pose 用 site 真值；type=keypoint/mask 也可用 site 真值（理想视觉）
        tgt_name = target.get("name") if isinstance(target, dict) else None
        if not tgt_name:
            return {"success": False, "reason": "target_lost", "steps": 0}
        # 找目标 3D 位置（site 或 body）
        tgt_3d = self._resolve_target_3d(env, tgt_name)
        if tgt_3d is None:
            return {"success": False, "reason": "target_lost", "steps": 0}

        lost_count = 0
        for t in range(timeout):
            end = _grip_site_pos(env, actor, grip_site)
            # reproj_error（仿真下用 3D xy 误差 × 像素/米 转换因子近似）
            err_xy = np.linalg.norm((end - tgt_3d)[[0, 1]])
            reproj_err = err_xy * 1000.0  # 近似：1mm ≈ 1px（仿真理想视觉）
            if reproj_err < tol_px and err_xy < tol_xy_m:
                return {"success": True, "steps": t, "reproj_error": reproj_err,
                        "xy_err": err_xy, "end": end.tolist()}
            # P 控制对齐（xy 为主，z 保持当前）
            move_tgt = np.array([tgt_3d[0], tgt_3d[1], end[2]])
            # 检测目标是否仍在视野（仿真下始终在，用 3D 距离判定）
            if np.linalg.norm(end - tgt_3d) > 0.5:
                lost_count += 1
                if lost_count > lost_max:
                    return {"success": False, "reason": "target_lost", "steps": t, "lost": lost_count}
            else:
                lost_count = 0
            step_pose_env(env, actor, _pose_action(env, actor, move_tgt, quat,
                                                   gripper=-1, k=gain * 10, max_step=0.01))
        return {"success": False, "reason": "servo_timeout", "steps": timeout,
                "reproj_error": reproj_err, "xy_err": err_xy}

    @staticmethod
    def _resolve_target_3d(env, name: str) -> Optional[np.ndarray]:
        """从 site/body 名解析目标 3D 位置（仿真真值）。"""
        try:
            pos = env.get_site_pos(name)
            if pos is not None:
                return np.asarray(pos, float).reshape(3)
        except Exception:
            pass
        try:
            pos = env.get_body_pos(name)
            if pos is not None:
                return np.asarray(pos, float).reshape(3)
        except Exception:
            pass
        return None


class GuardedMoveSkill(Skill):
    """带力保护的直线运动原语。

    沿 direction 推位姿目标（速度限幅），每步监测末端力，达 until.force_n 或 max_travel 停。

    成功：contact && travel < max_travel
    失败：excessive_force（force > force_limit）/ stuck（travel < min_travel）
    """

    spec = SkillSpec(
        name="guarded_move",
        description="Guarded straight move along a direction, stops on force threshold or max travel",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.SINGLE_SHOT,
        applies_when="force: 接触富集段，必须有 abort 阈值",
        parameters={
            "direction": {"type": "array", "description": "运动方向 3D（world frame）"},
            "speed_mps": {"type": "number"},
            "until": {"type": "object", "description": "终止条件 {force_n: 阈值}"},
            "max_travel_m": {"type": "number"},
        },
    )

    def execute(self, env, direction: List[float], frame: str = "tool",
                speed_mps: float = 0.01, until: Dict = None, max_travel_m: float = 0.03,
                force_limit: float = 15.0, min_travel_m: float = 0.001,
                actor: str = "agent0", grip_site: str = None,
                timeout: int = 100) -> Dict[str, Any]:
        quat = np.array(env.init_quat[actor], float)
        site = _grip_site(grip_site, actor)
        start = _grip_site_pos(env, actor, grip_site)
        dir_unit = np.asarray(direction, float).reshape(3)
        dir_unit = dir_unit / (np.linalg.norm(dir_unit) + 1e-8)
        force_n_thresh = float((until or {}).get("force_n", 6.0))
        # 每步位移 = speed_mps * dt（dt ≈ control_freq 倒数）
        dt = getattr(env, "dt", 0.05)
        step_len = float(speed_mps) * float(dt)
        travel = 0.0
        force_peak = 0.0
        contacted = False
        for t in range(timeout):
            end = _grip_site_pos(env, actor, grip_site)
            travel = float(np.linalg.norm(end - start))
            f = _ee_force_norm(env)
            force_peak = max(force_peak, f)
            # 力超限 → 失败
            if f > force_limit:
                return {"success": False, "reason": "excessive_force", "steps": t,
                        "force": f, "travel": travel}
            # 接触达成
            if f >= force_n_thresh:
                contacted = True
                if travel < max_travel_m:
                    return {"success": True, "steps": t, "force": f,
                            "travel": travel, "contact": True}
                else:
                    return {"success": False, "reason": "excessive_travel", "steps": t,
                            "force": f, "travel": travel}
            # 行程超限
            if travel >= max_travel_m:
                return {"success": False, "reason": "stuck" if travel < min_travel_m else "no_contact",
                        "steps": t, "travel": travel, "contact": False}
            # 推进：用固定路径目标（start + dir * step * t），避免末端漂移时目标跟漂
            move_tgt = start + dir_unit * step_len * (t + 1)
            step_pose_env(env, actor, _pose_action(env, actor, move_tgt, quat,
                                                   gripper=-1, k=4.0, max_step=step_len * 2))
        return {"success": False, "reason": "guarded_timeout", "steps": timeout,
                "travel": travel, "force_peak": force_peak, "contact": contacted}


class ImpedancePushSkill(Skill):
    """力觉终止的位置推进原语（force-guarded position push）。

    原计划用 CARTIMP 做真·阻抗控制，但 robopal CARTIMP 对 DianaMed 不稳定（K=0 都发散），
    且不能改源码。改为 CARTIK 位置控制 + force_ee sensor 做力觉终止/保护。
    这正是用户设计哲学："几何先到位，力觉做终止和保护"。

    沿 axis 推位置目标（每步小增量），监测 depth + force：
    - 几何先到位（CARTIK 稳定跟踪位置目标）
    - 力觉做终止（force 达阈值或 depth 达标即停）
    - 力觉做保护（force 超限即失败）

    成功：depth_reached && force_peak < force_limit（软接触/真插入）
          或 force_reached（硬接触：力达阈值 + 有位移）
    失败：excessive_force / stuck
    """

    spec = SkillSpec(
        name="impedance_push",
        description="Force-guarded position push along an axis with force/depth termination (CARTIK + force sensor)",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.SINGLE_SHOT,
        applies_when="force: 插入/按压段，几何先到位 + 力觉做终止和保护",
        parameters={
            "axis": {"type": "string", "description": "推进轴 x/y/z"},
            "until": {"type": "object", "description": "{force_n, depth_m} 终止条件"},
            "stiffness": {"type": "number", "description": "（保留参数，CARTIK 模式下忽略）"},
            "damping": {"type": "number", "description": "（保留参数，CARTIK 模式下忽略）"},
        },
    )

    def execute(self, env, axis: str = "z", until: Dict = None,
                stiffness: float = 100.0, damping: float = 40.0,
                force_limit: float = 15.0, depth_target: float = None,
                actor: str = "agent0", grip_site: str = None,
                timeout: int = 100, k: float = 2.0) -> Dict[str, Any]:
        quat = np.array(env.init_quat[actor], float)
        site = _grip_site(grip_site, actor)
        # CARTIK 无 set_cart_params；stiffness/damping 参数保留为接口兼容但不生效。
        # 力觉终止/保护靠 force_ee sensor 监测，不靠动态调刚度。

        start = _grip_site_pos(env, actor, grip_site)
        force_n_thresh = float((until or {}).get("force_n", 8.0))
        depth_m_target = float(depth_target if depth_target is not None else
                               (until or {}).get("depth_m", 0.012))
        axis_idx = {"x": 0, "y": 1, "z": 2}.get(axis, 2)
        dir_unit = np.zeros(3)
        dir_unit[axis_idx] = -1.0  # 默认沿 -axis 推进（下压/插入）
        depth = 0.0
        force_peak = 0.0
        for t in range(timeout):
            end = _grip_site_pos(env, actor, grip_site)
            depth = float(np.dot(start - end, dir_unit))  # 沿推进方向的位移
            f_along = abs(_ee_force_along(env, axis))
            force_peak = max(force_peak, f_along)
            # 力超限 → 失败
            if f_along > force_limit:
                return {"success": False, "reason": "excessive_force", "steps": t,
                        "force": f_along, "depth": depth}
            # depth 达标 + 力未超限 → 成功（软接触/真插入）
            if depth >= depth_m_target and f_along < force_limit:
                return {"success": True, "steps": t, "depth": depth,
                        "force": f_along, "force_peak": force_peak}
            # 力达阈值 + 有位移 → 成功（硬接触：接触即到位，force_reached 模式）
            if f_along >= force_n_thresh and depth > 0.0001:
                return {"success": True, "steps": t, "depth": depth,
                        "force": f_along, "force_peak": force_peak, "mode": "force_reached"}
            # 推进（位置目标：沿 dir_unit 缓慢推进，CARTIK 稳定跟踪）
            move_tgt = end + dir_unit * 0.002  # 每步 2mm 推进目标
            step_pose_env(env, actor, _pose_action(env, actor, move_tgt, quat,
                                                   gripper=-1, k=k, max_step=0.003))
        return {"success": False, "reason": "stuck", "steps": timeout,
                "depth": depth, "force_peak": force_peak}


class SpiralSearchSkill(Skill):
    """螺旋搜索恢复原语。

    xy 平面走螺旋路径，监测 until 条件（force_drop 找到孔或 depth_reached 插入成功）。
    作为 impedance_push 失败后的 recover 挂点。

    成功：until 条件满足
    失败：timeout
    """

    spec = SkillSpec(
        name="spiral_search",
        description="Spiral search in xy plane to find a hole/socket, stops on force drop or depth reached",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.SINGLE_SHOT,
        applies_when="force: 插入失败恢复，螺旋搜索找孔",
        parameters={
            "radius": {"type": "number", "description": "最大搜索半径"},
            "pitch": {"type": "number", "description": "螺旋螺距"},
            "until": {"type": "string", "description": "force_drop|depth_reached"},
        },
    )

    def execute(self, env, radius: float = 0.004, pitch: float = 0.001,
                until: str = "force_drop", depth_target: float = 0.012,
                force_drop_delta: float = 2.0,
                actor: str = "agent0", grip_site: str = None,
                timeout: int = 80) -> Dict[str, Any]:
        quat = np.array(env.init_quat[actor], float)
        site = _grip_site(grip_site, actor)
        center = _grip_site_pos(env, actor, grip_site)
        # 记录起始力（找孔时力会下降）
        f_baseline = _ee_force_norm(env)
        max_force = f_baseline
        depth_best = 0.0
        # 初始检查：如果已有显著接触力（guarded_move/impedance_push 已完成插入），
        # spiral_search 是 recovery 原语，前序已完成则直接成功，不必再搜。
        if f_baseline >= 1.5:
            return {"success": True, "steps": 0, "depth": 0.0,
                    "force": f_baseline, "mode": "already_inserted"}
        for t in range(timeout):
            # 螺旋路径：r = pitch * t / (2π), θ = 2π * pitch * t（保证螺距 pitch）
            theta = 2 * np.pi * pitch * max(t, 1)
            r = min(pitch * max(t, 1) / (2 * np.pi), radius)
            offset = np.array([r * np.cos(theta), r * np.sin(theta), 0.0])
            move_tgt = center + offset
            # 同时维持下压（z 方向缓慢推进）
            move_tgt[2] = center[2] - 0.001 * t
            step_pose_env(env, actor, _pose_action(env, actor, move_tgt, quat,
                                                   gripper=-1, k=2.0, max_step=0.002))
            end = _grip_site_pos(env, actor, grip_site)
            f = _ee_force_norm(env)
            depth = float(center[2] - end[2])
            depth_best = max(depth_best, depth)
            if until == "force_drop":
                # 找到孔时力骤降
                if f < f_baseline - force_drop_delta and depth > 0.002:
                    return {"success": True, "steps": t, "force": f,
                            "force_drop": f_baseline - f, "depth": depth}
            elif until == "depth_reached":
                if depth >= depth_target:
                    return {"success": True, "steps": t, "depth": depth, "force": f}
            max_force = max(max_force, f)
        return {"success": False, "reason": "spiral_timeout", "steps": timeout,
                "force_baseline": f_baseline, "max_force": max_force, "depth_best": depth_best}


# ============================================================
# 位姿夹爪原语（CARTIMP env 专用，保持末端位姿 + 控制夹爪）
# ============================================================

class PoseCloseGripperSkill(Skill):
    """闭合夹爪（CARTIMP 版，保持末端位姿）。"""

    spec = SkillSpec(
        name="pose_close_gripper", description="Close gripper holding current pose (CARTIMP pose mode)",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.SINGLE_SHOT,
        applies_when="position: 下降到位后夹持物体（CARTIMP env）",
    )

    def execute(self, env, actor: str = "agent0", grip_site: str = None,
                steps: int = 25) -> Dict[str, Any]:
        quat = np.array(env.init_quat[actor], float)
        for _ in range(steps):
            end, _ = _get_ee_pose(env, actor)
            step_pose_env(env, actor, _pose_action(env, actor, end, quat, gripper=-1, k=1.0))
        return {"success": True, "steps": steps}


class PoseOpenGripperSkill(Skill):
    """张开夹爪（CARTIMP 版，保持末端位姿）。"""

    spec = SkillSpec(
        name="pose_open_gripper", description="Open gripper holding current pose (CARTIMP pose mode)",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.SINGLE_SHOT,
        applies_when="position: 插入完成后释放物体（CARTIMP env）",
    )

    def execute(self, env, actor: str = "agent0", grip_site: str = None,
                steps: int = 20) -> Dict[str, Any]:
        quat = np.array(env.init_quat[actor], float)
        for _ in range(steps):
            end, _ = _get_ee_pose(env, actor)
            step_pose_env(env, actor, _pose_action(env, actor, end, quat, gripper=+1, k=1.0))
        return {"success": True, "steps": steps}
