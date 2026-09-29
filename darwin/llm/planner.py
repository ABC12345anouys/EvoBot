"""LLM 规划器：指令 + WorldState → 结构化 Plan；失败 → 带证据重规划。

任务无关：任务理解全部在 LLM 与谓词/技能清单里，本模块不含任何
任务名分支。LLM 不可用或输出非法时，OfflinePlanner 给出确定性的
通用堆叠计划，保证链路端到端可跑（也用于离线测试）。

Plan 协议（LLM 只输出 JSON）：
    {"goal_predicates": [{"pred": "on", "args": ["p2", "p1"]}],
     "steps": [{"skill": "grasp", "args": {"target": "p2"}},
               {"skill": "place_on", "args": {...},
                "expect": {"pred": "on", "args": ["p2", "p1"]}}]}
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..agents import predicates as P
from ..agents.robodojo_skills import MANIFEST
from ..agents.world_state import WorldState
from .client import LLMClient

_SKILL_NAMES = [m["skill"] for m in MANIFEST]


class PlanError(RuntimeError):
    pass


@dataclass
class Plan:
    goals: List[Dict[str, Any]]
    steps: List[Dict[str, Any]]
    rationale: str = ""


@dataclass
class Failure:
    step_index: int
    step: Dict[str, Any]
    result: Dict[str, Any]

    def brief(self) -> str:
        r = self.result
        return (f"step#{self.step_index} {self.step.get('skill')} "
                f"args={self.step.get('args')} 失败："
                f"{r.get('reason')} "
                f"mechanism={r.get('mechanism', '-')} "
                f"measures={r.get('measures')}")


# ---------------- 校验 ----------------

def validate(plan: Plan, state: WorldState) -> List[str]:
    """grounding 校验：实体 label、技能名、参数完整、goal 谓词合法。"""
    issues: List[str] = []
    labels = set(state.labels())
    for g in plan.goals:
        if g.get("pred") not in P.predicate_docs():
            issues.append(f"goal 未知谓词: {g.get('pred')}")
            continue
        for a in (g.get("args") or []):
            if a not in labels:
                issues.append(f"goal {g.get('pred')} 引用未知实体 {a!r}")
    for i, s in enumerate(plan.steps):
        if s.get("skill") not in _SKILL_NAMES:
            issues.append(f"step#{i} 未知技能 {s.get('skill')}")
            continue
        args = s.get("args") or {}
        for key in ("target", "support"):
            if key in args and args[key] not in labels:
                issues.append(f"step#{i} {key}={args[key]!r} 不存在")
        if "expect" in s:
            try:
                P.evaluate(s["expect"], state)
            except (KeyError, TypeError) as e:
                issues.append(f"step#{i} expect 非法: {e}")
    return issues


# ---------------- LLM 规划器 ----------------

_SYSTEM = """你是双臂机器人的操作规划大脑。根据任务指令和当前世界状态，输出一个 JSON 计划。

可用谓词（goal_predicates，描述任务目标）：
{p_preds}

可用技能（steps 的 skill）：
{p_skills}

输出规则：
1. 只输出 JSON，不要解释文字、不要 markdown 代码块。
2. 只能引用世界状态 entities 中存在的 label；目标按物理依赖从下到上排列。
3. 每个 place_on 前必须先 grasp 同一实体；place_on 后加 "expect" 谓词用于验证。
4. 被夹持的实体（held=true）已经在手中，不要重复 grasp。
5. 重规划时必须针对失败原因更换维度（不可达→换 arm；夹空→换 cand 或 sign），
   绝不重复“已尝试”列表中的方案。
JSON 格式：
{{"goal_predicates": [{{"pred": "on", "args": ["p2", "p1"]}}],
  "steps": [{{"skill": "grasp", "args": {{"target": "p2"}}}},
            {{"skill": "place_on", "args": {{"target": "p2", "support": "p1"}},
              "expect": {{"pred": "on", "args": ["p2", "p1"]}}}}]}}"""


def _format_manifests() -> str:
    pred_lines = "\n".join(f"- {k}: {v}" for k, v in P.predicate_docs().items())
    sk_lines = []
    for m in MANIFEST:
        args = "; ".join(f"{k}: {v}" for k, v in m["args"].items())
        sk_lines.append(f"- {m['skill']}（{args}）—— {m['desc']}")
    return pred_lines, "\n".join(sk_lines)


class LLMPlanner:
    def __init__(self, client: Optional[LLMClient] = None, *,
                 temperature: float = 0.2, max_repairs: int = 1):
        self.client = client
        self.temperature = temperature
        self.max_repairs = max_repairs

    def available(self) -> bool:
        return bool(self.client and self.client.available())

    def plan(self, instruction: str, state: WorldState) -> Plan:
        pred_s, sk_s = _format_manifests()
        messages = [
            {"role": "system",
             "content": _SYSTEM.format(p_preds=pred_s, p_skills=sk_s)},
            {"role": "user",
             "content": f"任务指令：{instruction}\n"
                        f"世界状态：{json.dumps(state.to_json(), ensure_ascii=False)}\n"
                        "输出 JSON 计划："}]
        return self._chat(messages, state)

    def replan(self, instruction: str, state: WorldState,
               failure: Failure, history: List[Dict[str, Any]]) -> Plan:
        pred_s, sk_s = _format_manifests()
        tried = json.dumps(
            [{"skill": h.get("skill"), "args": h.get("args"),
              "ok": h.get("ok"),
              "reason": (h.get("result") or {}).get("reason")}
             for h in history], ensure_ascii=False)
        messages = [
            {"role": "system",
             "content": _SYSTEM.format(p_preds=pred_s, p_skills=sk_s)},
            {"role": "user",
             "content": f"任务指令：{instruction}\n"
                        f"当前世界状态：{json.dumps(state.to_json(), ensure_ascii=False)}\n"
                        f"已尝试：{tried}\n"
                        f"最近失败 → {failure.brief()}\n"
                        "输出修正后的完整 JSON 计划（已满足的 expect 会被自动跳过，"
                        "可保留这些步骤）："}]
        return self._chat(messages, state)

    def _chat(self, messages, state) -> Plan:
        raw = self.client.chat(
            messages, temperature=self.temperature, max_tokens=1500)
        plan = _parse_plan(raw)
        issues = validate(plan, state)
        if not issues:
            return plan
        # 一次修复机会：把校验错误喂回去
        for _ in range(self.max_repairs):
            messages += [{"role": "assistant", "content": raw},
                         {"role": "user",
                          "content": "计划非法，请修正后重新输出完整 JSON：\n"
                                     + "\n".join(issues)}]
            raw = self.client.chat(messages, temperature=self.temperature,
                                   max_tokens=1500)
            plan = _parse_plan(raw)
            issues = validate(plan, state)
            if not issues:
                return plan
        raise PlanError("LLM 计划 grounding 校验失败: " + "; ".join(issues))


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def _parse_plan(raw: str) -> Plan:
    text = raw.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1)
    else:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            text = text[i:j + 1]
    try:
        d = json.loads(text)
    except json.JSONDecodeError as e:
        raise PlanError(f"LLM 输出不是合法 JSON: {e}; 原文: {raw[:200]}")
    return Plan(goals=list(d.get("goal_predicates") or []),
                steps=list(d.get("steps") or []),
                rationale=str(d.get("rationale", "")))


# ---------------- 确定性离线规划器 ----------------

_WS_CENTER = (0.0, -0.12)


class OfflinePlanner:
    """无 LLM 时的通用规划：堆叠动词 → 选底→大件先摞→小件后摞。

    重规划按 (arm, cand) 组合轮转：每次失败把失败实体的 grasp 换成
    下一个组合。不是任务定制规则，只是穷举通用技能参数空间。
    """

    def __init__(self):
        self._replans = 0

    def available(self) -> bool:
        return True

    def plan(self, instruction: str, state: WorldState) -> Plan:
        return self._stack_plan(state, fail_target=None)

    def replan(self, instruction: str, state: WorldState,
               failure: Failure, history) -> Plan:
        self._replans += 1
        target = (failure.step.get("args") or {}).get("target")
        return self._stack_plan(state, fail_target=target)

    def _combos(self):
        # 与 grasp 的 arm/cand 参数对应
        return [("auto", 0), ("auto", 1), ("left", 2),
                ("right", 0), ("right", 1)]

    def _stack_plan(self, state: WorldState,
                    fail_target: Optional[str]) -> Plan:
        ents = sorted(
            state.entities,
            key=lambda e: (-float(e.half[0]),
                           (e.x - _WS_CENTER[0]) ** 2
                           + (e.y - _WS_CENTER[1]) ** 2,
                           e.label))
        if not ents:
            return Plan([], [])
        base = ents[0]
        movers = ents[1:]
        movers.sort(key=lambda e: (-float(e.half[0]), e.label))  # 大件先放

        combo = self._combos()[min(self._replans, len(self._combos()) - 1)]
        goals, steps = [], []
        support = base.label
        chain = [(movers[i].label,
                  movers[i - 1].label if i > 0 else base.label)
                 for i in range(len(movers))]
        for mov, sup in chain:
            g = {"pred": "on", "args": [mov, sup]}
            goals.append(g)
            gargs: Dict[str, Any] = {"target": mov}
            if mov == fail_target:
                gargs["arm"] = combo[0]
                gargs["cand"] = combo[1]
            steps.append({"skill": "grasp", "args": gargs})
            steps.append({"skill": "place_on",
                          "args": {"target": mov, "support": sup},
                          "expect": g})
            support = mov
        return Plan(goals, steps)
