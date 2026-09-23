"""objectives：目标条件谓词 + entry→条件 适配层。

从 runner_dynamic.py 抽出，消除循环依赖根因（chain_registry/skill_creator/
experience_store 不再需要 import runner_dynamic 来拿 GoalCond）。
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


class GoalCond:
    """目标条件基类：check 判定是否达成，measure 返回观测数值（供日志/规划）。"""
    kind = "base"

    def describe(self) -> str:
        return self.kind

    def measure(self, env) -> Dict[str, Any]:
        raise NotImplementedError

    def check(self, env) -> bool:
        raise NotImplementedError


class JointAtLeast(GoalCond):
    """滑动/转动关节达到阈值（如抽屉拉开 qpos >= 0.08）。"""
    kind = "joint_ge"

    def __init__(self, joint: str, threshold: float, *,
                 target_qpos: Optional[float] = None,
                 handle_site: Optional[str] = None) -> None:
        self.joint = joint
        self.threshold = float(threshold)
        self.target_qpos = float(target_qpos if target_qpos is not None else threshold)
        self.handle_site = handle_site

    def describe(self) -> str:
        return f"{self.joint}>={self.threshold:g}"

    def _qpos(self, env) -> float:
        try:
            return float(env.mj_data.joint(self.joint).qpos[0])
        except Exception:
            return 0.0

    def measure(self, env) -> Dict[str, Any]:
        return {"qpos": round(self._qpos(env), 4), "threshold": self.threshold}

    def check(self, env) -> bool:
        return self._qpos(env) >= self.threshold


class BodyLifted(GoalCond):
    """物体被抬升到高度以上（grasp-only 任务的成功条件）。"""
    kind = "body_lifted"

    def __init__(self, body: str, height: float) -> None:
        self.body = body
        self.height = float(height)

    def describe(self) -> str:
        return f"{self.body}.z>={self.height:g}"

    def measure(self, env) -> Dict[str, Any]:
        z = float(env.get_body_pos(self.body)[2])
        return {"body_z": round(z, 4), "height": self.height}

    def check(self, env) -> bool:
        return float(env.get_body_pos(self.body)[2]) >= self.height


class BodyNearSite(GoalCond):
    """物体进入目标 site 邻域（抓放/插装的终态条件）。

    flavor: "cart" 笛卡尔抓放链；"pose" 位姿+力控插装链。
    """
    kind = "body_near_site"

    def __init__(self, body: str, site: str, tol: float, *,
                 flavor: str = "cart", stop_above: float = 0.0,
                 place_timeout: int = 150, place_k: float = 1.2,
                 descend_tol: float = 0.012) -> None:
        self.body = body
        self.site = site
        self.tol = float(tol)
        self.flavor = flavor
        self.stop_above = float(stop_above)
        self.place_timeout = int(place_timeout)
        self.place_k = float(place_k)
        # descend 到达容差：受限空间（抽屉腔等）TCP 无法降到中心+12mm，
        # 但指尖早已包住物体，可放宽退出条件
        self.descend_tol = float(descend_tol)

    def describe(self) -> str:
        return f"{self.body}~{self.site}<{self.tol:g}({self.flavor})"

    def measure(self, env) -> Dict[str, Any]:
        bp = np.asarray(env.get_body_pos(self.body), float)
        gp = np.asarray(env.get_site_pos(self.site), float)
        d = float(np.linalg.norm(bp - gp))
        return {"dist_mm": round(d * 1000.0, 1), "tol_mm": round(self.tol * 1000.0, 1),
                "body": [round(float(v), 4) for v in bp],
                "goal": [round(float(v), 4) for v in gp]}

    def check(self, env) -> bool:
        return float(np.linalg.norm(
            np.asarray(env.get_body_pos(self.body), float)
            - np.asarray(env.get_site_pos(self.site), float))) < self.tol


class LiberoGoal(GoalCond):
    """LIBERO 任务终态：BDDL 谓词全部满足（env.check_success）。"""
    kind = "libero_bddl"

    def __init__(self, description: str = "libero_bddl_success") -> None:
        self._desc = description

    def describe(self) -> str:
        return self._desc

    def measure(self, env) -> Dict[str, Any]:
        return {"bddl_success": self.check(env)}

    def check(self, env) -> bool:
        fn = getattr(env, "check_success", None)
        return bool(fn()) if callable(fn) else False


class LiberoSubGoal(GoalCond):
    """LIBERO 形式化任务的**单个**谓词子目标（On/In/Open/Close/Turnon）。

    check 优先走 env.eval_subgoal（与 BDDL _check_success 同口径的逐谓词
    求值）；隐式 Open 不在 goal 段时走 env.fixture_open 关节兜底。
    """
    kind = "libero_subgoal"

    def __init__(self, subgoal: Dict[str, Any]) -> None:
        self.predicate = str(subgoal.get("predicate", ""))
        self.skind = str(subgoal.get("kind", "place"))
        self.object = subgoal.get("object") or None
        self.target = subgoal.get("target") or None
        self.implicit = bool(subgoal.get("implicit", False))

    def describe(self) -> str:
        if self.object:
            return f"{self.predicate}({self.object},{self.target})"
        return f"{self.predicate}({self.target})"

    def measure(self, env) -> Dict[str, Any]:
        return {"subgoal": self.describe(), "done": self.check(env)}

    def check(self, env) -> bool:
        try:
            fn = getattr(env, "eval_subgoal", None)
            if callable(fn):
                return bool(fn(self.predicate, self.object, self.target))
        except KeyError:
            pass
        except Exception:
            return False
        # 隐式/兜底：关节态夹具（is_open / is_close）
        if self.skind == "articulate" and self.target:
            try:
                if self.predicate == "Open":
                    return bool(env.fixture_open(self.target))
                if self.predicate == "Close":
                    return bool(env.fixture_close(self.target))
            except Exception:
                return False
        return False


# ---- 声明适配层：entry（dict）→ 条件对象。唯一允许按任务字段映射的地方 ----

_COND_BUILDERS = {
    "joint_ge": lambda e, env: JointAtLeast(
        e["joint"], e["threshold"],
        target_qpos=e.get("target_qpos"), handle_site=e.get("handle_site")),
    "body_lifted": lambda e, env: BodyLifted(e["body"], e["height"]),
    "body_near_site": lambda e, env: BodyNearSite(
        e["body"], e["site"], e["tol"], flavor=e.get("flavor", "cart"),
        stop_above=e.get("stop_above", 0.0),
        place_timeout=e.get("place_timeout", 150), place_k=e.get("place_k", 1.2)),
}


def objectives_from_entry(entry: Dict[str, Any], env) -> List[GoalCond]:
    """把任务 entry 解析成有序目标条件。

    优先读 entry["objectives"]（数据声明，新任务不改代码）；
    否则从旧 mode 字段适配（grasp/full/insert/drawer）。
    """
    decl = entry.get("objectives")
    if decl is not None:
        conds = []
        for d in decl:
            b = _COND_BUILDERS.get(d.get("kind"))
            if b is None:
                raise ValueError(f"unknown objective kind: {d.get('kind')}")
            conds.append(b(d, env))
        return conds

    body, mode = entry["body"], entry["mode"]
    if mode == "libero":
        # LIBERO：形式化 spec 给出有序谓词子目标（多谓词任务顺序执行）；
        # 无 spec 的旧入口回退单一 BDDL 终态条件。
        spec = entry.get("libero_spec") or {}
        subs = [sg for sg in (spec.get("subgoals") or [])
                if sg.get("kind") != "unsupported"]
        if subs:
            return [LiberoSubGoal(sg) for sg in subs]
        return [LiberoGoal(entry.get("libero_language", "libero_bddl_success"))]
    if mode == "drawer":
        return [
            JointAtLeast(entry.get("drawer_joint", "drawer:joint"), 0.08,
                         target_qpos=0.12,
                         handle_site=entry.get("drawer_site", "drawer")),
            # stop_above 必须=0：descend 带 body 时 ref_z 已是物体中心，
            # 再抬高会让 TCP 停在方块上沿以上 → 闭爪夹空（lift_no_grip）。
            # （peg 场景的 0.025 是补偿 GraspNet pt.z 低于中心，语义不同）
            BodyNearSite(body, entry.get("cube_goal_site", "cube_goal"), 0.05,
                         flavor="cart", stop_above=0.0, descend_tol=0.02,
                         place_timeout=300, place_k=2.0),
        ]
    if mode == "insert":
        return [BodyNearSite(body, entry.get("goal_site", "goal_site"), 0.03,
                             flavor="pose")]
    if mode == "full":
        # InsertEnv 等 CARTIK 位姿环境只能用 pose 原语，place 用 pose_move_to+open
        flavor = "pose_cart" if entry.get("env_id") == "InsertEnv" else "cart"
        return [BodyNearSite(body, entry.get("goal_site", "goal_site"), 0.05,
                             flavor=flavor)]
    # grasp-only
    return [BodyLifted(body, 0.52)]


from typing import Optional  # noqa: E402  (放在末尾避免破坏上面结构)

__all__ = ["GoalCond", "JointAtLeast", "BodyLifted", "BodyNearSite",
           "LiberoGoal", "LiberoSubGoal", "objectives_from_entry"]
