"""运动原语(motion primitives):碰撞检测 + 路径规划。

设计哲学(延续 control.py):
- 分层:task skill (pick/place/insert) → control primitive (servo_align/guarded_move/...)
  → motion primitive (path_plan/collision_check)  ← 本模块
- 泛化:参数化"目标位姿+约束+终止条件",不绑定具体物体
- 用现成库 mplib(RRT-Connect + Pinocchio + FCL),不自己手搓规划器
- 失败映射:no_path / joint_limit / collision / track_failed → 回给 Agent/RAG 做恢复

与现有 P 控制原语的关系:
- pose_move_to(P 控制笛卡尔直线)= 不避障、不查碰撞、近距离/无障碍迁移
- path_plan(mplib RRT-Connect 关节空间规划)= 避自碰撞、远距离/复杂构型迁移
- 两者互补:path_plan 失败(no_path)可回退 pose_move_to;远距离用 path_plan,最后几 cm 用 P 控制

机型解耦(本模块不再硬编码任何机型/SDK):
- 所有 mujoco/robopal/mplib 细节下沉到 env.backend(RobotBackend 抽象)
- Skill 只调 backend 高层接口(plan_path/execute_trajectory/check_self/env_collision/get_*)
- 换机型只写 Backend 子类 + Profile YAML,本模块一行不改
- backend 未注入(无 profile)时返回 planner_unavailable,不崩

关键决策见 darwin/robot/mujoco_robopal.py(mplib URDF/SRDF/convex/base_pose/eef 偏移 等)。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..base import Skill, SkillSpec, SkillKind, Confidence
from . import _grip_site  # noqa: F401  (__init__ 仍 import motion *,保持 _grip_site 可用)


def _get_backend(env):
    """从 env 取 backend;未注入(无 profile)返回 None。"""
    return getattr(env, "backend", None)


def _fmt_cols(cols) -> List[str]:
    """mplib collision 对象 → 'link1<->link2' 字符串列表。"""
    return [getattr(c, "link_name1", "?") + "<->" + getattr(c, "link_name2", "?")
            for c in (cols or [])]


# ============================================================
# 碰撞检测原语
# ============================================================

class CollisionCheckSkill(Skill):
    """碰撞检测原语(基于 backend FCL)。

    检测给定/当前关节构型下的自碰撞 + 环境碰撞。
    mplib 的 check_for_self_collision 用 SRDF 排除相邻 link 对(不会误报 base-link1)。
    env_collision 需要 planning_world 里有障碍物(normal_objects/point_cloud);
    MVP 不自动同步 mujoco 障碍,故 env_collision 默认返回空(接口已留)。

    成功:collision_free=True
    失败:self_collision / env_collision(返回碰撞 link 对列表)
    """

    spec = SkillSpec(
        name="collision_check",
        description="Check self/environment collision at a given or current joint configuration (mplib FCL)",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.SINGLE_SHOT,
        applies_when="motion: 规划前/执行中验证候选构型是否碰撞(自碰撞+环境)",
        parameters={
            "qpos": {"type": "array", "description": "7 维关节角;None=用当前 mujoco qpos"},
            "use_attach": {"type": "boolean", "description": "是否含末端附着夹爪(默认 True)"},
            "with_point_cloud": {"type": "boolean", "description": "是否对 point_cloud 障碍检测(默认 False)"},
        },
    )

    def execute(self, env, qpos: Optional[List[float]] = None,
                use_attach: bool = True, with_point_cloud: bool = False,
                actor: str = "agent0", **_) -> Dict[str, Any]:
        backend = _get_backend(env)
        if backend is None:
            return {"success": False, "reason": "planner_unavailable",
                    "error": "env 未注入 backend(无 profile 或未在 runner 注入)"}
        q_arm = (np.asarray(qpos, float).reshape(-1)[:7] if qpos is not None
                 else backend.get_arm_qpos(actor))
        self_cols = backend.check_self_collision(actor=actor, qpos_arm=q_arm)
        env_cols = backend.check_env_collision(actor=actor, qpos_arm=q_arm,
                                               use_attach=use_attach, with_pc=with_point_cloud)
        self_list = _fmt_cols(self_cols)
        env_list = _fmt_cols(env_cols)
        collision_free = not (self_list or env_list)
        return {
            "success": collision_free,
            "collision_free": collision_free,
            "reason": "ok" if collision_free else
                      ("self_collision" if self_list else "env_collision"),
            "self_collisions": self_list,
            "env_collisions": env_list,
        }


# ============================================================
# 路径规划原语
# ============================================================

class PathPlanSkill(Skill):
    """路径规划原语(backend RRT-Connect 关节空间规划 + 准静态执行)。

    规划:backend.plan_qpos_to_pose(goal_pose, current_qpos, wrt_world=True)
        → 输出关节角序列 (N, 7),RRT-Connect 采样 + TOPPRA 时间最优化
        → 自碰撞感知(planning_world 含 robot 自碰撞对)
    执行:逐 waypoint mj_forward 准静态(保留关节角,避障有效)
        → 跳过 CARTIK 控制器(IK 重解会丢避障),直接设 qpos
        → 夹爪保持当前 qpos(夹持状态保持)

    成功:planned && executed && final pose 到位
    失败:no_path(RRT 失败)/ joint_limit / collision(执行后验证)/ track_failed
    """

    spec = SkillSpec(
        name="path_plan",
        description="Plan a collision-free joint-space path to a target pose (mplib RRT-Connect) and execute quasi-statically",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.SINGLE_SHOT,
        applies_when="motion: 远距离避障迁移(替代 pose_move_to 的无障碍直线段)",
        parameters={
            "goal_pose": {"type": "array", "description": "目标位姿 [x,y,z,qw,qx,qy,qz](world frame)"},
            "execute": {"type": "boolean", "description": "True=规划后执行;False=只规划返回 trajectory"},
            "planning_time": {"type": "number", "description": "RRT 时间预算(秒,默认 1.0)"},
            "rrt_range": {"type": "number", "description": "RRT 步长(默认 0.1)"},
            "frame": {"type": "string", "description": "goal_pose frame: world(默认)/base"},
        },
    )

    def execute(self, env, goal_pose: List[float], execute: bool = True,
                planning_time: float = 1.0, rrt_range: float = 0.1,
                time_step: float = 0.1, frame: str = "world",
                planner_name: str = "RRTConnect",
                tol: float = 0.01, skip: int = 2,
                actor: str = "agent0", grip_site: str = None,
                timeout: int = 200, **_) -> Dict[str, Any]:
        backend = _get_backend(env)
        if backend is None:
            return {"success": False, "reason": "planner_unavailable",
                    "error": "env 未注入 backend(无 profile 或未在 runner 注入)"}
        goal = np.asarray(goal_pose, float).reshape(-1)[:7]
        cur = backend.get_arm_qpos(actor)

        # ---- 规划 ----
        res = backend.plan_path(
            goal_pose=goal, actor=actor, frame=frame, current_qpos=cur,
            planning_time=planning_time, rrt_range=rrt_range,
            time_step=time_step, planner_name=planner_name,
        )
        if not res.get("ok"):
            return {"success": False, "reason": res.get("reason", "no_path"),
                    "status": res.get("status", ""), "error": res.get("error", ""),
                    "current_qpos": cur.tolist(), "goal_pose": goal.tolist()}
        traj = np.asarray(res["trajectory"], float)  # (N, 7)
        duration = float(res.get("duration", 0.0))

        if not execute:
            return {"success": True, "reason": "ok_plan_only",
                    "trajectory": traj.tolist(), "duration": duration}

        # ---- 执行(准静态 mj_forward,降采样 skip)----
        ex = backend.execute_trajectory(traj, actor=actor, skip=skip, timeout=timeout)
        if ex.get("reason") == "exec_timeout":
            return {"success": False, "reason": "exec_timeout",
                    "executed": ex["executed"], "trajectory": traj.tolist()}
        executed = ex["executed"]

        # ---- 到位验证(eef 当前位姿 vs eef 目标;等价 link7 误差,刚体偏移抵消)----
        final_q = backend.get_arm_qpos(actor)
        final_pos, _ = backend.get_eef_pose(actor)
        err = float(np.linalg.norm(final_pos - goal[:3]))

        # 执行后 backend 验证自碰撞(基于 SRDF + convex hull,保守)
        cols = backend.check_self_collision(actor=actor, qpos_arm=final_q)
        if cols:
            return {"success": False, "reason": "collision",
                    "collisions": _fmt_cols(cols),
                    "final_err": err, "executed": executed}
        if err > tol and err >= 0:
            return {"success": False, "reason": "track_failed", "final_err": err,
                    "executed": executed, "final_pos": final_pos.tolist()}

        return {"success": True, "reason": "ok", "executed": executed,
                "final_err": err, "duration": duration,
                "trajectory_len": int(len(traj))}
