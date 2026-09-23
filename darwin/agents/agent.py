"""ManipulationAgent：自进化机械臂操作智能体。

架构（去分发，agent 直接持有 skill 实例）：
    skills: Dict[str, Skill]   —— 感知 + 控制原语 + forged 技能
    rag: RAGMemory             —— 经验记忆（成功/失败沉淀 + 检索 + UCB 排序）
    llm: Optional[LLMClient]   —— LLM 决策（可选，无则走规则 plan）

自进化闭环（run_episode）：
    perceive → retrieve(RAG) → decide(LLM/规则) → execute(直接调 skill)
    → reflect(沉淀 RAG + forge 新 skill) → tick() + decay()

跨任务泛化：rel_offset = (grasp_point - object_center) / half_size，不同物体经验可迁移。
成功率自适应探索：rank_candidates 用 success_rate + UCB bonus，低成功率 band 多试。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from ..skills import Skill, load_skills
from ..skills.forged import ForgedSkill
from ..memory import RAGMemory
from ..llm.client import LLMClient


@dataclass
class EpisodeResult:
    success: bool
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    info: Dict[str, Any] = field(default_factory=dict)


def _parse_action(response: str) -> Optional[Dict[str, Any]]:
    """解析 LLM 输出的 <action>name</action><params>json</params>。"""
    m = re.search(r"<action>\s*(.*?)\s*</action>", response, re.DOTALL)
    if not m:
        return None
    name = m.group(1).strip()
    params: Dict[str, Any] = {}
    pm = re.search(r"<params>\s*(.*)\s*</params>", response, re.DOTALL)
    if pm:
        try:
            params = json.loads(pm.group(1).strip())
        except json.JSONDecodeError:
            s = pm.group(1)
            lo, hi = s.find("{"), s.rfind("}")
            if lo != -1 and hi > lo:
                try:
                    params = json.loads(s[lo:hi + 1])
                except json.JSONDecodeError:
                    pass
    return {"action": name, "params": params}


class ManipulationAgent:
    """自进化操作智能体。"""

    FINISH = "finish"

    def __init__(self, skills: Optional[Dict[str, Skill]] = None,
                 rag: Optional[RAGMemory] = None,
                 llm: Optional[LLMClient] = None,
                 task_name: str = "task", seed: str = "s0",
                 max_turns: int = 30) -> None:
        self.skills = skills or load_skills()
        # forged skill 绑定完整 skills 表，使其能调用原语
        for s in self.skills.values():
            if isinstance(s, ForgedSkill):
                s.bind(self.skills)
        self.rag = rag or RAGMemory()
        self.llm = llm if llm is not None else (LLMClient() if LLMClient().available() else None)
        self.task_name = task_name
        self.seed = seed
        self.max_turns = max_turns

    # ============================================================
    # 1. 感知
    # ============================================================

    def perceive(self, env, task: str,
                 candidates: Optional[List[Dict]] = None) -> tuple:
        """感知物体特征 + 抓取候选。

        优先用传入的 candidates（调用方已感知）；否则尝试 detect + grasp 技能。
        返回 (feat, candidates)。
        """
        feat: Dict[str, Any] = {"shape": "unknown", "size": [0.02, 0.02, 0.02]}
        # 尝试用环境直接取物体信息（仿真 ground truth）
        if hasattr(env, "get_body_pos") and hasattr(env, "object_names"):
            for name in getattr(env, "object_names", []):
                try:
                    pos = env.get_body_pos(name)
                    feat = {"shape": name.split("_")[0], "size": [0.02, 0.02, 0.02],
                            "center": pos.tolist() if hasattr(pos, "tolist") else list(pos)}
                    break
                except Exception:
                    continue
        # 感知技能（GraspNet / detect）
        if candidates is None:
            grasp = self.skills.get("grasp_pose")
            if grasp is not None:
                try:
                    res = grasp.execute(env=env)
                    if res.get("success"):
                        candidates = res.get("candidates") or [{"score_band": "mid", "rel_offset": [0, 0, 0]}]
                except Exception:
                    pass
        if candidates is None:
            candidates = [{"score_band": "mid", "rel_offset": [0.0, 0.0, 0.0]}]
        return feat, candidates

    # ============================================================
    # 2. RAG 检索
    # ============================================================

    def retrieve(self, task: str, feat: Dict, candidates: List[Dict]) -> Dict[str, Any]:
        """RAG 检索：成功经验 + 失败过滤 + UCB 重排候选。"""
        # 过滤近期失败候选
        valid = [c for c in candidates
                 if self.rag.is_failed(task, feat, c.get("rel_offset", [0, 0, 0])) is None]
        if not valid:
            valid = candidates  # 全失败时保留，让 UCB 探索
        # UCB 重排
        ranked = self.rag.rank_candidates(task, feat, valid)
        # 语义检索成功经验
        best = self.rag.best_success(task, feat)
        ctx = self.rag.retrieve(f"{task} {feat.get('shape')}", feat, k=3)
        return {
            "ranked_candidates": ranked,
            "best_success": best,
            "context": ctx,
            "stats": self.rag.stats,
        }

    # ============================================================
    # 3. 决策
    # ============================================================

    def _skill_specs_text(self) -> str:
        lines = []
        for name, s in self.skills.items():
            spec = s.spec
            lines.append(f"- {name}: {spec.description} | 参数: {json.dumps(spec.parameters, ensure_ascii=False)}")
        return "\n".join(lines)

    def decide(self, task: str, feat: Dict, rag_ctx: Dict,
               plan: Union[List[Dict], Callable, None] = None) -> List[Dict]:
        """决策：有 LLM 用 LLM，否则用规则 plan。"""
        if plan is not None:
            return plan(feat, rag_ctx) if callable(plan) else plan
        if self.llm is None:
            return self._default_plan(task, feat, rag_ctx)

        sys_prompt = (
            f"你是机械臂操作规划器。任务：{task}\n"
            f"物体特征：{json.dumps(feat, ensure_ascii=False)}\n"
            f"历史经验（RAG）：{json.dumps(rag_ctx.get('context', []), ensure_ascii=False, default=str)[:2000]}\n"
            f"可用技能：\n{self._skill_specs_text()}\n\n"
            "每轮只输出一个动作，格式：\n"
            "<action>skill_name</action>\n<params>{\"参数名\": 值}</params>\n"
            f"完成后输出 <action>{self.FINISH}</action><params>{{\"success\": true/false}}</params>"
        )
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": f"请规划完成 {task} 的动作序列。"}]
        actions: List[Dict] = []
        for _ in range(self.max_turns):
            try:
                resp = self.llm.chat(messages)
            except Exception:
                break
            messages.append({"role": "assistant", "content": resp})
            act = _parse_action(resp)
            if act is None:
                messages.append({"role": "user", "content": "请严格按 <action>/<params> 格式输出。"})
                continue
            if act["action"] == self.FINISH:
                break
            if act["action"] not in self.skills:
                messages.append({"role": "user", "content": f"技能 {act['action']} 不存在。"})
                continue
            actions.append(act)
            messages.append({"role": "user", "content": "继续下一个动作，或 finish。"})
        return actions or self._default_plan(task, feat, rag_ctx)

    def _default_plan(self, task: str, feat: Dict, rag_ctx: Dict) -> List[Dict]:
        """规则默认 plan：pick → lift → [place]。"""
        cand = (rag_ctx.get("ranked_candidates") or [{}])[0]
        off = np.array(cand.get("rel_offset", [0, 0, 0]), float)
        center = np.array(feat.get("center", [0.4, 0.0, 0.5]), float)
        size = np.array(feat.get("size", [0.02, 0.02, 0.02]), float)
        grasp_pt = (center + off * size).tolist()
        plan = [
            {"action": "home", "params": {"actor": "agent0"}},
            {"action": "move_above", "params": {"point": grasp_pt, "actor": "agent0"}},
            {"action": "descend", "params": {"point": grasp_pt, "actor": "agent0"}},
            {"action": "close_gripper", "params": {"actor": "agent0"}},
            {"action": "lift", "params": {"height": 0.55, "actor": "agent0"}},
        ]
        if "place" in task:
            plan.append({"action": "move_to_xy_top", "params": {"target": feat.get("goal", [0.5, 0.0]), "actor": "agent0"}})
            plan.append({"action": "place", "params": {"goal": feat.get("goal", [0.5, 0.0, 0.5]), "actor": "agent0"}})
            plan.append({"action": "open_gripper", "params": {"actor": "agent0"}})
        return plan

    # ============================================================
    # 4. 执行（去分发：直接调 skill）
    # ============================================================

    def execute(self, plan: List[Dict], env) -> tuple:
        """按动作序列执行，返回 (trajectory, success)。"""
        trajectory: List[Dict] = []
        for i, act in enumerate(plan):
            name = act["action"]
            params = act.get("params", {})
            skill = self.skills.get(name)
            if skill is None:
                trajectory.append({"action": name, "params": params,
                                   "result": {"success": False, "error": f"unknown skill: {name}"},
                                   "step_idx": i})
                return trajectory, False
            try:
                result = skill.execute(env=env, **params)
            except Exception as e:
                result = {"success": False, "error": str(e)}
            trajectory.append({"action": name, "params": params, "result": result, "step_idx": i})
            if not result.get("success", False):
                return trajectory, False
        return trajectory, True

    # ============================================================
    # 5. 沉淀 + forge
    # ============================================================

    def reflect(self, task: str, feat: Dict, trajectory: List[Dict],
                success: bool, grasp_pt=None, control_mode: Optional[str] = None,
                control_hints: Optional[Dict] = None, fail_cat: str = "") -> None:
        """沉淀经验到 RAG；成功则尝试 forge 新 skill。

        fail_cat: 失败分类（grip_failed/timeout/collision/force_exceed），
                  由 runner_dynamic 传入，存入 RAG 供精准过滤。
        """
        center = np.array(feat.get("center", [0, 0, 0]), float)
        size = np.array(feat.get("size", [0.02, 0.02, 0.02]), float)
        if grasp_pt is not None:
            rel = (np.array(grasp_pt, float) - center) / np.maximum(size, 1e-6)
        else:
            rel = [0.0, 0.0, 0.0]
        # sanity check：边界异常不写
        if np.linalg.norm(rel) > RAGMemory.OFFSET_SANITY:
            rel = [0.0, 0.0, 0.0]
        band = "mid"
        # 失败归因：找第一个失败的 phase
        fail_phase = ""
        if not success:
            for step in trajectory:
                if not step["result"].get("success", False):
                    fail_phase = step["action"]
                    break
        if success:
            self.rag.add_success(task, feat, rel.tolist() if hasattr(rel, "tolist") else list(rel),
                                 band, attempts=1, exp=f"{task} via {trajectory[0]['action'] if trajectory else '?'}",
                                 control_mode=control_mode, control_hints=control_hints)
            self._maybe_forge(task, trajectory, feat)
        else:
            self.rag.add_failure(task, feat, rel.tolist() if hasattr(rel, "tolist") else list(rel),
                                 band, fail_phase=fail_phase,
                                 exp=f"failed at {fail_phase}",
                                 fail_cat=fail_cat,
                                 control_mode=control_mode, control_hints=control_hints)

    def _maybe_forge(self, task: str, trajectory: List[Dict], feat: Dict) -> None:
        """成功轨迹自动沉淀为 forged skill（去分发版：调用 skills dict）。"""
        from ..evolution.skill_forge import forge_skill_from_trajectory
        try:
            forge_skill_from_trajectory(trajectory, task_name=task, feat=feat)
        except Exception:
            pass  # forge 失败不影响主流程

    # ============================================================
    # 主闭环
    # ============================================================

    def run_episode(self, env, task: str, feat: Optional[Dict] = None,
                    candidates: Optional[List[Dict]] = None,
                    plan: Union[List[Dict], Callable, None] = None,
                    grasp_pt=None, control_mode: Optional[str] = None,
                    control_hints: Optional[Dict] = None) -> EpisodeResult:
        """执行一次自进化闭环。"""
        # 1. 感知
        if feat is None:
            feat, candidates = self.perceive(env, task, candidates)
        # 2. RAG 检索
        rag_ctx = self.retrieve(task, feat, candidates or [])
        # 3. 决策
        actions = self.decide(task, feat, rag_ctx, plan)
        # 4. 执行
        trajectory, success = self.execute(actions, env)
        # 5. 沉淀 + forge
        self.reflect(task, feat, trajectory, success, grasp_pt=grasp_pt,
                     control_mode=control_mode, control_hints=control_hints)
        # 6. RAG 时钟
        self.rag.tick()
        self.rag.decay()
        summary = "success" if success else f"failed: {trajectory[-1].get('result', {}).get('reason', '?') if trajectory else '?'}"
        return EpisodeResult(success=success, trajectory=trajectory, summary=summary,
                             info={"task": task, "feat": feat, "rag_stats": self.rag.stats})
