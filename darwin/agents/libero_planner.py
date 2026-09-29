"""LIBERO 确定性规划器：task_spec 子目标 → 技能调用计划。

职责边界：
- 排序来自 task_spec._order_subgoals（BDDL 依赖拓扑，确定性）；
- 本模块只做**谓词种类 → 技能**的映射（place→grasp+place_at，
  articulate→articulate，toggle→toggle），与任务名零耦合；
- 满足判定调 adapter 的 BDDL 真值谓词（eval_subgoal / fixture_open），
  已满足的子目标自动跳过（重规划时从世界现状继续，不重跑前缀）。

重规划（replan_hook）：技能失败返回 mechanism，本模块把它映射为
下一次执行的候选参数（grasp_miss → 下一个几何抓取候选；
contact_blocked/ik_unreachable → 同一子目标重来，世界已变）——
机制级映射表，不是任务级 if/else。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from . import libero_skills as S

# 谓词 → 开合方向 / 旋钮方向（BDDL 词汇级映射）
_DIRECTION = {"Open": "open", "Close": "close",
              "Turnon": "turnon", "Turnoff": "turnoff"}


def plan_subgoal(sub: Dict[str, Any]) -> List[Dict[str, Any]]:
    """一个子目标 → 有序技能步骤 [{skill, args}]。"""
    kind = sub.get("kind")
    if kind == "place":
        return [
            {"skill": "grasp", "args": {"obj": sub["object"]}},
            {"skill": "place_at",
             "args": {"obj": sub["object"], "target": sub["target"],
                      "predicate": sub["predicate"]}},
        ]
    if kind == "articulate":
        return [{"skill": "articulate",
                 "args": {"target": sub["target"],
                          "direction": _DIRECTION[sub["predicate"]]}}]
    if kind == "toggle":
        return [{"skill": "toggle",
                 "args": {"target": sub["target"],
                          "direction": _DIRECTION[sub["predicate"]]}}]
    return []  # unsupported：跳过（由 runner 记录）


def plan_subgoal_alternative(sub: Dict[str, Any]) -> List[Dict[str, Any]]:
    """子目标的替代物理实现（析取规划的第二支）。

    place 子目标（On/In 只约束物体最终位置）在抓取候选空间被实测
    耗尽（grasp 返回 no_candidate——几何事实，非执行失败）时，退回
    推/拨归约。机制级映射，与任务名零耦合。
    """
    if sub.get("kind") == "place":
        return [{"skill": "push_to",
                 "args": {"obj": sub["object"], "target": sub["target"],
                          "predicate": sub["predicate"]}}]
    return []


def subgoal_satisfied(adapter, sub: Dict[str, Any]) -> bool:
    """子目标满足判定：BDDL 真值谓词（与 _check_success 同口径）。

    隐式 Open 不在 goal_state 中（KeyError）→ 回退 fixture_open。
    """
    pred = sub.get("predicate")
    try:
        if sub.get("kind") == "place":
            return bool(adapter.eval_subgoal(pred, sub.get("object"),
                                             sub.get("target")))
        return bool(adapter.eval_subgoal(pred, sub.get("target")))
    except KeyError:
        try:
            if pred == "Open":
                return bool(adapter.fixture_open(sub["target"]))
            if pred == "Close":
                return bool(adapter.fixture_close(sub["target"]))
        except Exception:
            return False
    except Exception:
        return False
    return False


def execute_step(adapter, step: Dict[str, Any], *,
                 cand: int = 0, strategy: int = 0,
                 mem: Optional[Dict[str, dict]] = None) -> Dict[str, Any]:
    """执行一个技能步骤（cand/strategy 由重规划按 mechanism 注入）。"""
    name = step["skill"]
    args = dict(step.get("args") or {})
    if name == "grasp":
        return S.grasp(adapter, cand=cand, mem=mem, **args)
    if name == "place_at":
        return S.place_at(adapter, mem=mem, **args)
    if name == "articulate":
        return S.articulate(adapter, strategy=strategy, **args)
    if name == "toggle":
        return S.toggle(adapter, strategy=strategy, **args)
    if name == "move_to":
        return S.move_to(adapter, **args)
    if name == "push_to":
        return S.push_to(adapter, **args)
    return {"success": False, "reason": f"未知技能 {name}",
            "mechanism": "wrong_state", "measures": {}}


def replan_hint(failures: int, mechanism: Optional[str]) -> Dict[str, int]:
    """失败机制 → 下次执行的候选参数（机制级，不是任务级）。

    grasp_miss（夹空）：换几何抓取候选；
    ik_unreachable（目标点被挡/够不到）：同样换候选——目标点邻域内
    找可行点是几何问题的标准响应；
    其余机制：世界已被动作改变，用新感知重跑同一子目标（cand 保持）。
    """
    if mechanism in ("grasp_miss", "ik_unreachable"):
        return {"cand": failures, "strategy": 0}
    return {"cand": min(failures, 0), "strategy": 0}
