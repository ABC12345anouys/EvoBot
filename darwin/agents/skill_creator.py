"""SkillCreator：面对新任务（无匹配 method）时自动生成新技能并注册。

参考 EvoAgentX：
- AgentGenerator：LLM 生成 agent 规格 + schema 验证（inputs 必须出现在 prompt）；
- AFlow optimizer：LLM 基于经验生成新 graph，check_modification 防重复；
- WorkflowGenerator：目标→子任务→分配/生成 agent。

darwin 版：
- 触发时机：registry.select(cond) 抛出 ValueError（无方法能达成该条件）；
- 输入：cond 描述 + 现有方法列表 + 可用 skill 名录 + ExperienceStore 历史失败；
- 输出：一个新的 YAML method（match + steps），DeclarativeMethod 校验通过后
  register 到当前 registry + 写入 methods/<name>.yaml（下次进程自动加载）；
- LLM 不可用时走规则回退：对已知 cond.kind 生成一个通用链模板。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .chain_registry import (
    ChainRegistry, DeclarativeMethod, TemplateError, METHODS_DIR,
)
from .objectives import GoalCond
from .experience_store import ExperienceStore


# 规则回退：对已知条件类型生成一个基础方法（LLM 不可用时兜底）
_RULE_FALLBACK = {
    "body_near_site": {
        "match": {"kind": "body_near_site"},
        # flavor 未指定时按 cart 处理；pose 需 LLM 生成
        "steps": [
            {"skill": "home"},
            {"skill": "move_above", "params": {"point": "$grasp_pt", "hover": "$cfg.hover"}},
            {"skill": "descend",
             "params": {"point": "$grasp_pt", "body": "$cond.body",
                        "k": "$cfg.k_descend", "stop_above": "$cond.stop_above"}},
            {"skill": "close_gripper"},
            {"skill": "lift", "params": {"height": "$cfg.lift_height", "body": "$cond.body"}},
            {"skill": "move_to_xy_top",
             "params": {"target": ["$goal_site.x", "$goal_site.y"],
                        "height": "$cfg.lift_height"}},
            {"skill": "place",
             "params": {"goal": "$goal_site", "body": "$cond.body",
                        "timeout": "$cond.place_timeout", "k": "$cond.place_k"}},
            {"skill": "open_gripper"},
        ],
    },
    "body_lifted": {
        "match": {"kind": "body_lifted"},
        "steps": [
            {"skill": "move_above", "params": {"point": "$grasp_pt", "hover": "$cfg.hover"}},
            {"skill": "descend",
             "params": {"point": "$grasp_pt", "body": "$cond.body", "k": "$cfg.k_descend"}},
            {"skill": "close_gripper"},
            {"skill": "lift", "params": {"height": "$cond.height", "body": "$cond.body"}},
        ],
    },
}


class SkillCreator:
    """自动生成并注册新 method 的组件。"""

    def __init__(self, registry: ChainRegistry,
                 llm=None,
                 experience: Optional[ExperienceStore] = None,
                 available_skills: Optional[List[str]] = None) -> None:
        self.registry = registry
        self.llm = llm
        self.experience = experience
        self.available_skills = available_skills or []

    # ---------- 公开入口 ----------

    def create_for(self, cond: GoalCond, task: str = "task") -> Optional[DeclarativeMethod]:
        """为一个无匹配方法的条件生成新 method 并注册。

        返回注册成功的 DeclarativeMethod；生成/校验失败返回 None（调用方应
        回退到 episode 失败，由 reflect 记录经验）。
        """
        # 1. 经验防重复：若该条件已有失败方法且参数与规则回退一致，不浪费 LLM 调用
        if self.experience is not None:
            fail_methods = self.experience.known_methods(task)
            if fail_methods and self._all_failed_with_rule(cond, task):
                return None  # 规则回退已试过且失败，交给 LLM 或放弃

        # 2. 优先 LLM 生成
        spec = None
        if self.llm is not None and getattr(self.llm, "available", lambda: False)():
            spec = self._llm_generate(cond, task)

        # 3. LLM 不可用/失败 → 规则回退
        if spec is None:
            spec = self._rule_fallback(cond)

        if spec is None:
            return None

        name = str(spec.get("name", ""))
        # 规则回退没 name 时按条件生成
        if not name:
            name = f"auto_{cond.kind}_{re.sub(r'[^a-z0-9]', '', cond.describe())[:12]}"
        spec["name"] = name

        # 4. 校验 + 注册
        try:
            method = DeclarativeMethod(spec, source=f"yaml:auto_gen:{name}")
        except TemplateError as e:
            self._log(f"生成的方法校验失败: {e}")
            return None

        # 校验 skill 名存在（防止 LLM 编造不存在的 skill）
        if self.available_skills:
            for s in method.raw_steps:
                if s["skill"] not in self.available_skills:
                    self._log(f"方法引用未知 skill: {s['skill']}（可用: {self.available_skills}）")
                    return None

        self.registry.register(method, replace=True)
        self._persist(name, spec)
        self._log(f"已自动生成并注册新方法: {name} "
                  f"(match={method.match_kind}, steps={len(method.raw_steps)})")
        return method

    # ---------- LLM 生成 ----------

    def _llm_generate(self, cond: GoalCond, task: str) -> Optional[Dict[str, Any]]:
        """用 LLM 生成 YAML method spec。失败返回 None。"""
        methods_desc = "\n".join(
            f"- {m['name']}: match={m['kind']} source={m['source']}"
            for m in self.registry.list_methods())
        skills_desc = ", ".join(self.available_skills) or "（未知）"
        exp_text = ""
        if self.experience is not None:
            exp_text = self.experience.format_for_creator(task, cond)

        prompt = f"""你是机械臂操作技能生成器。当前任务遇到一个目标条件，现有方法库无法达成它，
需要你生成一个新的声明式方法（YAML）。

## 目标条件
- kind: {cond.kind}
- 描述: {cond.describe()}

## 现有方法（不要重复，且不能匹配此条件）
{methods_desc}

## 可用 skill 名录（steps[].skill 只能从这里选）
{skills_desc}

## 历史经验
{exp_text}

## YAML 方法格式
```yaml
name: <唯一名字，kebab-case，与条件相关>
match:
  kind: {cond.kind}
  flavor: cart        # 仅 body_near_site 需要：cart 或 pose
include_home: false   # 若链首是 home 则 true，否则引擎自动补
steps:
  - skill: <skill 名>
    params:            # 可选，支持 $变量
      <key>: <$变量或字面量>
```

可用 $变量（白名单，无 eval）：
  $safe_z, $grasp_pt(.x/.y/.z), $site:<名>(.axis),
  $goal_site(.axis), $cond.<属性>, $cfg.<键>, $entry.<键>
actor/grip_site 自动注入，不要写。

只输出 YAML 代码块，不要其他文字。"""

        try:
            resp = self.llm.chat(
                [{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=1500)
            spec = self._parse_yaml_from_llm(resp, cond)
        except Exception as e:
            self._log(f"LLM 生成失败: {e}")
            spec = None
        return spec

    @staticmethod
    def _parse_yaml_from_llm(text: str, cond: GoalCond) -> Optional[Dict[str, Any]]:
        """从 LLM 回复中提取 YAML 代码块并校验。"""
        # 取 ```yaml ... ``` 或 ``` ... ```
        m = re.search(r"```(?:ya?ml)?\s*\n(.*?)\n```", text, re.DOTALL)
        body = m.group(1) if m else text
        try:
            spec = yaml.safe_load(body)
        except yaml.YAMLError as e:
            print(f"[skill_creator] YAML 解析失败: {e}")
            return None
        if not isinstance(spec, dict):
            return None
        # 强制 match.kind 与条件一致，防止 LLM 跑偏
        match = spec.setdefault("match", {})
        match["kind"] = cond.kind
        return spec

    # ---------- 规则回退 ----------

    def _rule_fallback(self, cond: GoalCond) -> Optional[Dict[str, Any]]:
        """对已知条件类型生成通用链模板。"""
        tpl = _RULE_FALLBACK.get(cond.kind)
        if tpl is None:
            self._log(f"无规则回退模板 for cond.kind={cond.kind}")
            return None
        # body_near_site 的 flavor 适配：pose 用 pose_ 前缀 skill
        spec = {"name": f"auto_{cond.kind}", "match": dict(tpl["match"]),
                "steps": [dict(s) for s in tpl["steps"]]}
        if cond.kind == "body_near_site" and getattr(cond, "flavor", None) == "pose":
            spec["match"]["flavor"] = "pose"
            # 简单替换前缀
            pose_map = {"move_above": "pose_move_above", "descend": "pose_descend",
                        "close_gripper": "pose_close_gripper", "lift": "pose_lift",
                        "move_to_xy_top": "pose_move_to", "open_gripper": "pose_open_gripper"}
            for s in spec["steps"]:
                if s["skill"] in pose_map:
                    s["skill"] = pose_map[s["skill"]]
        return spec

    def _all_failed_with_rule(self, cond: GoalCond, task: str) -> bool:
        """规则回退生成的方法是否已被记录为失败（避免重复）。"""
        name = f"auto_{cond.kind}"
        return bool(self.experience.failures(task, name))

    # ---------- 持久化 ----------

    @staticmethod
    def _persist(name: str, spec: Dict[str, Any]) -> None:
        """写入 methods/<name>.yaml，下次进程自动加载。"""
        METHODS_DIR.mkdir(parents=True, exist_ok=True)
        path = METHODS_DIR / f"{name}.yaml"
        path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True),
                        encoding="utf-8")

    @staticmethod
    def _log(msg: str) -> None:
        print(f"[skill_creator] {msg}", flush=True)

    # ---------- 方法变异（参考 EvoAgentX AFlow：对成功 method 做小变异）----------

    def mutate_existing(self, method_name: str, *, max_mutations: int = 2) -> Optional[str]:
        """对已有方法做 1~max_mutations 处小变异，生成新方法并注册。

        变异策略（随机选）：
        - 数值参数 ±10% 扰动（hover/k_descend/lift_height/stiffness/...）
        - 替换某个 skill 为同义 skill（如 move_to↔move_to_xy_top）
        - 调整某 step 的顺序（仅相邻交换）

        返回新方法名；失败返回 None。新方法命名为 <name>_mutant_<ts>。
        """
        import copy
        import random
        import time

        method = self.registry.get(method_name)
        if method is None or not isinstance(method, DeclarativeMethod):
            self._log(f"变异失败：方法 {method_name} 不存在或非 YAML 声明式")
            return None

        spec = copy.deepcopy(method.spec)
        steps = spec.get("steps", [])
        if not steps:
            return None

        mutations = random.randint(1, max_mutations)
        for _ in range(mutations):
            strategy = random.choice(["param_jitter", "skill_swap", "step_swap"])
            idx = random.randint(0, len(steps) - 1)
            step = steps[idx]

            if strategy == "param_jitter":
                params = step.get("params") or {}
                num_keys = [k for k, v in params.items()
                            if isinstance(v, (int, float)) and not isinstance(v, bool)]
                if num_keys:
                    k = random.choice(num_keys)
                    old = float(params[k])
                    params[k] = round(old * random.uniform(0.9, 1.1), 4)
            elif strategy == "skill_swap":
                swaps = {"move_to": "move_to_xy_top", "move_to_xy_top": "move_to",
                         "descend": "descend", "lift": "lift"}
                if step["skill"] in swaps and swaps[step["skill"]] != step["skill"]:
                    if swaps[step["skill"]] in self.available_skills:
                        step["skill"] = swaps[step["skill"]]
            elif strategy == "step_swap" and len(steps) > 1:
                j = (idx + 1) % len(steps)
                steps[idx], steps[j] = steps[j], steps[idx]

        new_name = f"{method_name}_mutant_{int(time.time()) % 100000}"
        spec["name"] = new_name
        try:
            new_method = DeclarativeMethod(spec, source=f"yaml:mutant:{new_name}")
        except TemplateError as e:
            self._log(f"变异方法校验失败: {e}")
            return None

        if self.available_skills:
            for s in new_method.raw_steps:
                if s["skill"] not in self.available_skills:
                    self._log(f"变异方法引用未知 skill: {s['skill']}，丢弃")
                    return None

        self.registry.register(new_method, replace=True)
        self._persist(new_name, spec)
        self._log(f"变异生成新方法: {new_name} (源: {method_name}, {mutations} 处变异)")
        return new_name


__all__ = ["SkillCreator"]
