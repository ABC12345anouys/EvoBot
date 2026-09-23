"""forged 技能：成功轨迹自动沉淀的可重放技能。

forged skill 的 entry.py 定义 run(env, registry) 函数（历史签名保留）。
为兼容去分发架构，用 _RegistryProxy 把 registry.execute 转发到 skills dict。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Callable, Dict

import yaml

from ..base import Skill, SkillSpec, SkillKind, Confidence, Evidence

FORGED_DIR = Path(__file__).resolve().parent


class _RegistryProxy:
    """兼容旧 forged skill 的 registry.execute 签名，转发到 skills dict。"""

    def __init__(self, skills: Dict[str, Skill]) -> None:
        self._skills = skills

    def execute(self, name: str, env=None, **kwargs):
        skill = self._skills.get(name)
        if skill is None:
            return {"success": False, "error": f"unknown skill: {name}"}
        try:
            result = skill.execute(env=env, **kwargs) if env is not None else skill.execute(**kwargs)
            return result if isinstance(result, dict) else {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_skill_description(self, name: str):
        skill = self._skills.get(name)
        return skill.get_skill_description() if skill else None


class ForgedSkill(Skill):
    """forged 技能包装：持有 run 函数，execute 时注入 skills 代理。"""

    def __init__(self, name: str, run_fn: Callable, spec: SkillSpec) -> None:
        self.spec = spec
        self._run = run_fn
        self._skills: Dict[str, Skill] = {}

    def bind(self, skills: Dict[str, Skill]) -> None:
        """agent 加载后绑定完整 skills 表，使 forged skill 能调用原语。"""
        self._skills = skills

    def execute(self, env=None, **kwargs) -> Dict[str, Any]:
        proxy = _RegistryProxy(self._skills)
        result = self._run(env=env, registry=proxy, **kwargs)
        return result if isinstance(result, dict) else {"success": True, "result": result}


def _parse_skill_md(path: Path) -> SkillSpec:
    text = path.read_text()
    meta = {}
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end > 0:
            try:
                meta = yaml.safe_load(text[3:end]) or {}
            except Exception:
                meta = {}
    evm = meta.get("evidence") or {}
    evidence = Evidence(
        cells=list(evm.get("cells", [])),
        tasks=list(evm.get("tasks", [])),
        attempts=int(evm.get("attempts", 0)),
        solved_seeds=list(evm.get("solved_seeds", [])),
        failed_seeds=list(evm.get("failed_seeds", [])),
    )
    return SkillSpec(
        name=meta.get("name", path.parent.name),
        description=meta.get("description", ""),
        kind=SkillKind(meta.get("kind", "strategy")),
        confidence=Confidence(meta.get("confidence", "single-shot")),
        applies_when=meta.get("applies_when", ""),
        entry_file=meta.get("entry_file", "entry.py"),
        evidence=evidence,
    )


def load_forged_skills() -> Dict[str, ForgedSkill]:
    """加载所有 forged 技能，返回 {name: ForgedSkill}。"""
    skills: Dict[str, ForgedSkill] = {}
    if not FORGED_DIR.exists():
        return skills
    for d in FORGED_DIR.iterdir():
        if not (d.is_dir() and (d / "entry.py").exists()):
            continue
        entry = d / "entry.py"
        module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(d.name, entry))
        try:
            importlib.util.spec_from_file_location(d.name, entry).loader.exec_module(module)  # type: ignore
        except Exception:
            continue
        run_fn = getattr(module, "run", None) or getattr(module, "main", None)
        if run_fn is None:
            continue
        spec = _parse_skill_md(d / "SKILL.md")
        skills[spec.name] = ForgedSkill(spec.name, run_fn, spec)
    return skills


__all__ = ["ForgedSkill", "load_forged_skills", "FORGED_DIR"]
