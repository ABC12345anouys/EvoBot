"""形式化任务规格：LIBERO BDDL goal → 有序子目标序列。

BDDL 本身就是形式化规格（谓词 + 对象 + region）。本模块把 goal 段里的
**全部**谓词解析出来，并按物理依赖排序为 agent/sim 共用的 TaskSpec：

    On A B       → place：抓 A 放到 B（物体或 region site）
    In A R       → place：抓 A 放进 region R；若 R 是抽屉/柜门/微波炉
                   region 且 goal 未显式 Open → 补一个隐式 Open 前置
    Open R       → articulate（拉开）
    Close R      → articulate（关上），排序到同柜体 place 之后
    Turnon X     → toggle（旋钮/按钮），排序到 On(_, X_cook_region) 之前
    Turnoff X    → toggle

未知谓词进 unsupported（不崩），由 agent LLM 兜底或台账标 unsupported 跳过。
排序是确定性的（零 token）：依赖边 + BDDL 原文顺序做稳定拓扑。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

TASK_SPECS_DIR = (Path(__file__).resolve().parents[1]
                  / "skills" / "configs" / "task_specs")

# 已知谓词 → 子目标大类
_PLACE_PRED = {"On", "In"}
_ARTICULATE_PRED = {"Open", "Close"}
_TOGGLE_PRED = {"Turnon", "Turnoff"}
_KNOWN_PRED = _PLACE_PRED | _ARTICULATE_PRED | _TOGGLE_PRED | {"NextTo"}

# goal 段内所有谓词调用（大小写照 BDDL 实际写法：Turnon 小写 n）
_PRED_RE = re.compile(
    r"\((On|In|NextTo|Open|Close|Turnon|Turnoff)\b\s*"
    r"([A-Za-z0-9_]+)?(?:\s+([A-Za-z0-9_]+))?\s*\)")
# 兜底：抓 goal 段内所有大写开头的形如谓词的 token，用于发现未知谓词
_ANY_PRED_RE = re.compile(r"\(([A-Z][A-Za-z0-9_]*)\b")

# region/对象名 → 柜体关节夹具（drawer / cabinet / microwave）
_ARTICULATED_HINT = re.compile(r"(drawer|cabinet|microwave)", re.I)
_COOK_HINT = re.compile(r"(stove|cook)", re.I)

# region 名后缀归一化（用于 fixture 关联）
_SUFFIXES = ("_contain_region", "_cook_region", "_top_region", "_middle_region",
             "_bottom_region", "_front_region", "_back_region", "_left_region",
             "_right_region", "_default_site", "_region", "_site")


def _fixture_key(name: str) -> str:
    """把 region/site/对象名归一成夹具主键：flat_stove_1_cook_region→flat_stove_1。"""
    out = name
    changed = True
    while changed:
        changed = False
        for suf in _SUFFIXES:
            if out.endswith(suf):
                out = out[: -len(suf)]
                changed = True
                break
    return out


def _is_articulated_region(name: str) -> bool:
    return bool(_ARTICULATED_HINT.search(name))


def _kind_of(predicate: str) -> str:
    if predicate in _PLACE_PRED:
        return "place"
    if predicate in _ARTICULATE_PRED:
        return "articulate"
    if predicate in _TOGGLE_PRED:
        return "toggle"
    return "unsupported"  # NextTo 等：v1 不支持


def parse_bddl_spec(bddl_path: str, language: str = "") -> Dict[str, Any]:
    """解析 BDDL goal 为有序 TaskSpec（纯函数，不读缓存）。"""
    text = Path(bddl_path).read_text(encoding="utf-8")
    i = text.find("(:goal")
    goal = text[i:] if i >= 0 else text

    raw: List[Tuple[str, Optional[str], Optional[str]]] = []
    for m in _PRED_RE.finditer(goal):
        raw.append((m.group(1), m.group(2), m.group(3)))

    _LOGICAL = {"And", "Or", "Not"}  # BDDL 逻辑连接词，不是谓词
    known = {m.group(1) for m in _ANY_PRED_RE.finditer(goal)}
    unknown = sorted(p for p in known
                     if p not in _KNOWN_PRED and p not in _LOGICAL)

    subs: List[Dict[str, Any]] = []
    for idx, (pred, a1, a2) in enumerate(raw):
        kind = _kind_of(pred)
        if kind == "place":
            subs.append({"predicate": pred, "kind": kind,
                         "object": a1 or "", "target": a2 or "",
                         "implicit": False, "raw_index": idx})
        elif kind in ("articulate", "toggle"):
            subs.append({"predicate": pred, "kind": kind,
                         "object": None, "target": a1 or "",
                         "implicit": False, "raw_index": idx})
        else:  # NextTo 等
            subs.append({"predicate": pred, "kind": "unsupported",
                         "object": a1, "target": a2,
                         "implicit": False, "raw_index": idx})

    ordered = _order_subgoals(subs)
    unsupported_tokens = sorted({s["predicate"] for s in ordered
                                 if s["kind"] == "unsupported"} | set(unknown))
    for s in ordered:
        s.pop("raw_index", None)
        s.pop("anchor", None)
    return {
        "bddl": str(bddl_path),
        "language": language,
        "subgoals": ordered,
        "unsupported": unsupported_tokens,
        "source": "deterministic",
    }


def _order_subgoals(subs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """依赖排序：
    - place 进抽屉/柜/微波炉 region → 同 key 的 Open 必须在前（缺则补隐式）
    - Close(同 key) → 排在 place 之后；无 place 关联时保持原位
    - Turnon(同 key) → 排在 On(其 cook region) 之前
    - 其余保持 BDDL 原序（作者按执行顺序书写，如 stack 再入盘）
    """
    items = [dict(s, anchor=s["raw_index"]) for s in subs]
    supported = [s for s in items if s["kind"] != "unsupported"]
    # unsupported 不参与执行，排序后统一附加
    unsupported = [s for s in items if s["kind"] == "unsupported"]

    # 1) 补隐式 Open（place 进可开合 region 且没有显式 Open 同 key）
    existing_open = {_fixture_key(s["target"]) for s in supported
                     if s["kind"] == "articulate" and s["predicate"] == "Open"}
    implicit_nodes: List[Dict[str, Any]] = []
    for s in supported:
        if s["kind"] == "place" and _is_articulated_region(s["target"] or ""):
            key = _fixture_key(s["target"])
            if key not in existing_open:
                node = {"predicate": "Open", "kind": "articulate",
                        "object": None, "target": s["target"],
                        "implicit": True, "anchor": s["anchor"] - 0.5,
                        "raw_index": -1}
                implicit_nodes.append(node)
                existing_open.add(key)
    nodes = supported + implicit_nodes

    # 2) 依赖边（after 列表存 anchor）
    # 用对象承载顺序，简单稳定拓扑：给每个节点计算 after 集合
    by_anchor = {n["anchor"]: n for n in nodes}

    def _find(pred=None, kind=None, key=None):
        out = []
        for n in nodes:
            if pred is not None and n["predicate"] != pred:
                continue
            if kind is not None and n["kind"] != kind:
                continue
            if key is not None and _fixture_key(n["target"] or "") != key:
                continue
            out.append(n)
        return out

    after: Dict[float, List[float]] = {n["anchor"]: [] for n in nodes}

    def add_edge(u: Dict[str, Any], v: Dict[str, Any]) -> None:
        """u 必须在 v 前。"""
        if u["anchor"] != v["anchor"] and u["anchor"] not in after[v["anchor"]]:
            after[v["anchor"]].append(u["anchor"])

    for s in [n for n in nodes if n["kind"] == "place"]:
        key = _fixture_key(s["target"] or "")
        # Open → place（显式或隐式）
        for op in _find(pred="Open", key=key):
            add_edge(op, s)
        # place → Close
        for cl in _find(pred="Close", key=key):
            add_edge(s, cl)
        # Turnon → On 到炉面
        if _COOK_HINT.search(s["target"] or ""):
            for tn in _find(pred="Turnon", key=key):
                add_edge(tn, s)

    # 3) Kahn 稳定拓扑：每轮选 anchor 最小的入度零节点
    ordered: List[Dict[str, Any]] = []
    remaining = set(after.keys())
    # 入度
    indeg = {a: len([u for u in ups if u in remaining])
             for a, ups in after.items()}
    while remaining:
        ready = sorted(a for a in remaining if indeg[a] == 0)
        if not ready:  # 理论上无环；保险：强制取最小
            ready = [min(remaining)]
        a = ready[0]
        remaining.discard(a)
        ordered.append(by_anchor[a])
        for b in remaining:
            if a in after[b]:
                indeg[b] -= 1

    # 4) 同 anchor 顺序下 implicit 已在其 place 前（edge 保证）；unsupported 附尾
    return ordered + unsupported


# ----------------- 缓存（agent LLM 兜底产物可覆盖确定性解析）-----------------

def cache_path(suite: str, idx: int) -> Path:
    return TASK_SPECS_DIR / f"{suite}_{int(idx):02d}.yaml"


def load_or_parse(suite: str, idx: int, bddl_path: str,
                  language: str = "") -> Dict[str, Any]:
    """优先读 agent 沉淀的缓存 YAML，否则现场确定性解析。"""
    p = cache_path(suite, idx)
    if p.exists():
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if data.get("subgoals"):
                return data
        except yaml.YAMLError:
            pass
    return parse_bddl_spec(bddl_path, language)


def save_cache(suite: str, idx: int, spec: Dict[str, Any]) -> Path:
    TASK_SPECS_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(suite, idx)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(spec, allow_unicode=True,
                                  sort_keys=False, default_flow_style=False),
                   encoding="utf-8")
    tmp.replace(p)
    return p
