"""skill_forge：成功轨迹 → 自动沉淀为 forged skill（去分发版）。

与 registry 无关：直接写文件到 darwin/skills/forged/<name>/，
生成的 entry.py 用 run(env, registry) 签名（registry 由 agent 注入代理）。

成功轨迹 → entry.py（动作序列重放）+ SKILL.md（元数据）。
失败轨迹 → 走 RAG failure 记忆（agent.reflect 已处理），这里不重复。
"""
from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..skills.forged import FORGED_DIR


def extract_skill_code(trajectory: List[Dict[str, Any]], task_name: str) -> Dict[str, Any]:
    """从成功轨迹中抽象出技能代码骨架。"""
    actions = [t for t in trajectory if t.get("action")]
    if not actions:
        return {"success": False, "error": "轨迹中无动作"}

    action_names = [a["action"] for a in actions]
    action_body = "\n".join(
        f"    result = registry.execute('{a['action']}', env=env, **{a.get('params') or {}})"
        for a in actions
    )
    code = (
        f'"""自动生成技能: {task_name}（skill_forge 抽象，成功轨迹重放模板）。"""\n'
        f"def run(env=None, registry=None):\n"
        f'    """由成功轨迹抽象而来：按验证过的动作序列重放（坐标为该次成功观测值）。"""\n'
        f"{action_body}\n"
        f"    return result\n"
    )
    description = f"任务 {task_name} 的成功轨迹抽象：{' -> '.join(action_names)}"
    applies_when = f"类似 {task_name} 的任务，涉及操作: {set(action_names)}"
    return {"success": True, "code": code, "description": description, "applies_when": applies_when}


def _save_forged_skill(name: str, description: str, code: str,
                       applies_when: str = "") -> Dict[str, Any]:
    """直接写 forged skill 文件（不依赖 registry）。"""
    safe_name = re.sub(r"[^\w\-]", "_", name).lower()
    skill_dir = FORGED_DIR / safe_name
    if skill_dir.exists():
        safe_name = f"{safe_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        skill_dir = FORGED_DIR / safe_name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "entry.py").write_text(code)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {safe_name}\ndescription: {description[:200]}\n"
        f"entry_file: entry.py\nkind: strategy\nconfidence: single-shot\n"
        f"applies_when: '{applies_when}'\n"
        f"evidence:\n  cells: []\n  tasks: []\n  attempts: 0\n  solved_seeds: []\n  failed_seeds: []\n---\n\n"
        f"# {safe_name}\n{description}\n"
    )
    return {"success": True, "skill_name": safe_name}


def _update_evidence(skill_name: str, cell: str, solved: bool = True,
                     task: Optional[str] = None) -> None:
    """更新 forged skill 的 SKILL.md evidence frontmatter。"""
    skill_md = FORGED_DIR / skill_name / "SKILL.md"
    if not skill_md.exists():
        return
    text = skill_md.read_text()
    if not text.startswith("---"):
        return
    end = text.find("\n---", 3)
    if end <= 0:
        return
    try:
        meta = yaml.safe_load(text[3:end]) or {}
    except Exception:
        meta = {}
    ev = meta.get("evidence") or {}
    cells = sorted({*ev.get("cells", []), cell})
    tasks = sorted({*ev.get("tasks", []), task} if task else ev.get("tasks", []))
    attempts = int(ev.get("attempts", 0)) + 1
    solved_seeds = sorted({*ev.get("solved_seeds", []), cell} if solved else ev.get("solved_seeds", []))
    failed_seeds = sorted(ev.get("failed_seeds", []) if solved else {*ev.get("failed_seeds", []), cell})
    meta["evidence"] = {
        "cells": cells, "tasks": tasks, "attempts": attempts,
        "solved_seeds": solved_seeds, "failed_seeds": failed_seeds,
    }
    n_tasks = len(tasks)
    meta["confidence"] = (
        "verified" if len(cells) >= 3 and n_tasks >= 2
        else "probable" if len(cells) >= 2
        else "single-shot"
    )
    new_fm = yaml.safe_dump(meta, sort_keys=False, allow_unicode=True)
    skill_md.write_text(f"---\n{new_fm}---{text[end + 4:]}")


def forge_skill_from_trajectory(trajectory: List[Dict], task_name: str,
                                feat: Optional[Dict] = None) -> Dict[str, Any]:
    """agent.reflect 调用：成功轨迹 → 沉淀/更新 forged skill。"""
    safe_name = re.sub(r"[^\w\-]", "_", task_name).lower()
    created = False
    if not (FORGED_DIR / safe_name / "entry.py").exists():
        result = extract_skill_code(trajectory, task_name)
        if not result.get("success"):
            return result
        saved = _save_forged_skill(safe_name, result["description"], result["code"],
                                   result.get("applies_when", ""))
        if not saved.get("success"):
            return saved
        safe_name = saved["skill_name"]
        created = True
    cell = f"{task_name}_{datetime.datetime.now().strftime('%H%M%S%f')}"
    _update_evidence(safe_name, cell, solved=True, task=task_name)
    return {"success": True, "skill_name": safe_name, "created": created}
