"""Agent Loop：感知 → 规划 → 执行 → 谓词验证 → 重规划的通用闭环。

不含任何任务名分支：任务差异完全由 instruction、谓词与技能序列承载。
新任务复用本闭环，只需能在场景图上被感知、被技能操作。

run() 是生成器（yield 帧动作），结束时 return 结果摘要。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from . import predicates as P
from .robodojo_skills import SKILLS, SkillContext
from .world_state import WorldState


class Failure:
    """轻量失败描述（避免 import llm 侧 dataclass 造成环依赖）。"""

    def __init__(self, step_index: int, step: dict, result: dict):
        self.step_index = step_index
        self.step = step
        self.result = result

    def brief(self) -> str:
        r = self.result
        return (f"step#{self.step_index} {self.step.get('skill')} "
                f"args={self.step.get('args')} 失败：{r.get('reason')} "
                f"mechanism={r.get('mechanism', '-')} measures={r.get('measures')}")

    # llm.planner.Failure 兼容字段
    def as_dict(self) -> Dict[str, Any]:
        return {"step_index": self.step_index, "step": self.step,
                "result": self.result}


def _goals_met(goals, state: WorldState):
    measures = {}
    for g in goals:
        ok, m = P.evaluate(g, state)
        measures[P.describe(g)] = m
        if not ok:
            return False, measures
    return True, measures


class AgentLoop:
    def __init__(self, ctx: SkillContext, planner: Any,
                 instruction: str, max_replans: int = 4):
        self.ctx = ctx
        self.planner = planner
        self.instruction = instruction
        self.max_replans = int(max_replans)

    def run(self):
        ctx = self.ctx
        state = ctx.perceive(True)
        plan = self.planner.plan(self.instruction, state)
        print(f"[agent] 计划: goals={[P.describe(g) for g in plan.goals]} "
              f"steps={len(plan.steps)}")

        history: list = []
        for round_ in range(self.max_replans + 1):
            met, gms = _goals_met(plan.goals, state)
            if met:
                return self._success(gms)

            failure = None
            i = 0
            while i < len(plan.steps):
                st = plan.steps[i]
                expect = st.get("expect")
                if expect is not None and P.evaluate(expect, state)[0]:
                    i += 1
                    continue
                fn = SKILLS[st["skill"]]
                result = yield from fn(ctx, **(st.get("args") or {}))
                history.append({"skill": st["skill"],
                                "args": st.get("args"),
                                "ok": bool(result.get("success")),
                                "result": result})
                if not result.get("success"):
                    failure = Failure(i, st, result)
                    print(f"[agent] {failure.brief()}")
                    break
                # 世界改变 → 全量感知；否则轻量刷新保证 expect 用新数据
                state = ctx.perceive(bool(result.get("world_change")))
                if expect is not None:
                    ok_e, m_e = P.evaluate(expect, state)
                    if not ok_e:
                        r = {"success": False,
                             "reason": f"执行成功但期望未达成 {P.describe(expect)}",
                             "mechanism": "goal_not_reached",
                             "measures": m_e}
                        failure = Failure(i, st, r)
                        print(f"[agent] {failure.brief()}")
                        break
                i += 1

            if failure is None:
                # 步骤走完但目标未达
                r = {"success": False,
                     "reason": "全部步骤执行完毕但 goal_predicates 未满足",
                     "mechanism": "goal_not_reached", "measures": gms}
                failure = Failure(len(plan.steps) - 1,
                                  plan.steps[-1] if plan.steps else {}, r)

            if round_ >= self.max_replans:
                return self._give_up(failure, history)
            print(f"[agent] 第 {round_ + 1} 次重规划（原因机制："
                  f"{failure.result.get('mechanism', '-')}）")
            plan = self.planner.replan(self.instruction, state,
                                       failure, history)
            state = ctx.perceive(True)
            print(f"[agent] 新计划: goals={[P.describe(g) for g in plan.goals]}"
                  f" steps={len(plan.steps)}")
        return self._give_up(None, history)

    @staticmethod
    def _success(goal_measures):
        print(f"[agent] SUCCESS goal_measures={goal_measures}")
        return {"success": True, "goal_measures": goal_measures}

    @staticmethod
    def _give_up(failure: Optional[Failure], history):
        reason = failure.brief() if failure else "unknown"
        print(f"[agent] GIVE UP after {len(history)} 次技能尝试: {reason}")
        return {"success": False, "fail": reason,
                "mechanism": failure.result.get("mechanism") if failure else None}
