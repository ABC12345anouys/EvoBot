"""RPent 式记忆存储：YAML frontmatter + evidence 合并 + 自动置信度升级。

记忆元数据字段（来自 RPent memory/manager.py）：
- scope: global / suite
- kind: primitive / perception / strategy / failure / infra
- confidence: single-shot / probable / verified（evidence.cells≥3 且 tasks≥2 → verified）
- evidence: {cells, attempts, solved_seeds, failed_seeds, contradicted_by}
- applies_when: 触发条件
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..skills.base import Evidence, Confidence, SkillKind

SCOPES = {"global", "suite"}
KINDS = {k.value for k in SkillKind}
CONFIDENCE = {c.value for c in Confidence}


def _split_frontmatter(path: Path) -> tuple[Dict[str, Any], str]:
    text = path.read_text(errors="replace")
    if not text.startswith("---"):
        raise ValueError("missing YAML frontmatter")
    end = text.find("\n---", 3)
    if end < 0:
        raise ValueError("unterminated YAML frontmatter")
    metadata = yaml.safe_load(text[3:end]) or {}
    return metadata, text[end + 4:]


def _render(metadata: Dict[str, Any], body: str) -> str:
    return "---\n" + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True) + "---\n" + body


def merge_evidence(old: Dict, new: Dict) -> tuple[Dict, str]:
    """合并证据并自动升级置信度（来自 RPent _merge_evidence）。"""
    old_ev = old.get("evidence") or {}
    new_ev = new.get("evidence") or {}
    cells = sorted({*old_ev.get("cells", []), *new_ev.get("cells", [])})
    evidence = {
        **old_ev,
        "cells": cells,
        "attempts": int(old_ev.get("attempts") or 0) + int(new_ev.get("attempts") or 0),
    }
    for key in ("solved_seeds", "failed_seeds", "contradicted_by"):
        if key in old_ev or key in new_ev:
            evidence[key] = sorted({*old_ev.get(key, []), *new_ev.get(key, [])})
    tasks = {str(c).rsplit("_s", 1)[0] for c in cells}
    confidence = (
        Confidence.VERIFIED.value if len(cells) >= 3 and len(tasks) >= 2
        else Confidence.PROBABLE.value if len(cells) >= 2
        else Confidence.SINGLE_SHOT.value
    )
    return {**old, "evidence": evidence, "confidence": confidence}, confidence


class MemoryStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def add(self, name: str, metadata: Dict[str, Any], body: str) -> str:
        """写入一条记忆。metadata 必须含 scope/kind/confidence/evidence/applies_when。"""
        self._validate(metadata)
        path = self._root / f"{name}.md"
        if path.exists():
            old_meta, _ = _split_frontmatter(path)
            metadata, _ = merge_evidence(old_meta, metadata)
        path.write_text(_render(metadata, body))
        return str(path)

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        path = self._root / f"{name}.md"
        if not path.exists():
            return None
        meta, body = _split_frontmatter(path)
        return {"metadata": meta, "body": body}

    def list_all(self) -> List[Dict[str, Any]]:
        items = []
        for path in self._root.glob("*.md"):
            try:
                meta, body = _split_frontmatter(path)
                items.append({"name": path.stem, "metadata": meta, "body": body})
            except Exception:
                continue
        return items

    def search(self, kind: Optional[str] = None, min_confidence: Optional[str] = None) -> List[Dict]:
        conf_order = [Confidence.SINGLE_SHOT.value, Confidence.PROBABLE.value, Confidence.VERIFIED.value]
        min_idx = conf_order.index(min_confidence) if min_confidence else 0
        result = []
        for item in self.list_all():
            meta = item["metadata"]
            if kind and meta.get("kind") != kind:
                continue
            if min_confidence and conf_order.index(meta.get("confidence", "single-shot")) < min_idx:
                continue
            result.append(item)
        return result

    @staticmethod
    def _validate(metadata: Dict[str, Any]) -> None:
        scope = metadata.get("scope")
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {sorted(SCOPES)}, got {scope!r}")
        if metadata.get("kind") not in KINDS:
            raise ValueError(f"kind must be one of {sorted(KINDS)}")
        if metadata.get("confidence") not in CONFIDENCE:
            raise ValueError(f"confidence must be one of {sorted(CONFIDENCE)}")
        evidence = metadata.get("evidence") or {}
        if not isinstance(evidence.get("cells"), list) or not evidence["cells"]:
            raise ValueError("evidence.cells must be a non-empty list")
