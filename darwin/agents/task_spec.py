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

**两条分解路径**（`load_or_parse` 决定走哪条）：

1. **LLM 分解（当前默认）**：把任务定义物化成一份自描述的形式化规格
   （`build_formal`：谓词表带 id + 隐式前置候选 + 词表 + 夹具属性 + 可用
   技能约束 + 确定性参考序），交 LLM 做**谓词级分解**——只决定执行顺序与
   是否插隐式前置（如"先开抽屉"），**不选技能、不造谓词、不改参数**。
   产物冻结进 `skills/configs/task_specs/<suite>_<idx>.yaml`，运行期只读：
   零 token、完全确定。
2. **确定性拓扑排序（fallback / validator / 参考序）**：`_order_subgoals`
   用依赖边 + BDDL 原文顺序做 Kahn 稳定拓扑。它同时是 LLM 输出的校验依据
   （`validate_order`：id 必须已知、不重复、不漏、add 只能取候选）和 diff 基准。
"""
from __future__ import annotations

import json
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


def _goal_text(bddl_path: str) -> str:
    """BDDL 的 (:goal ...) 段（找不到就整篇，容错）。"""
    text = Path(bddl_path).read_text(encoding="utf-8")
    i = text.find("(:goal")
    return text[i:] if i >= 0 else text


def _raw_predicates(goal: str) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """goal 段内的谓词调用原文，按 BDDL 书写顺序。"""
    return [(m.group(1), m.group(2), m.group(3)) for m in _PRED_RE.finditer(goal)]


def _unknown_predicates(goal: str) -> List[str]:
    """goal 段内出现但不在已知表里的谓词名（用于 unsupported 报告）。"""
    _LOGICAL = {"And", "Or", "Not"}  # BDDL 逻辑连接词，不是谓词
    known = {m.group(1) for m in _ANY_PRED_RE.finditer(goal)}
    return sorted(p for p in known if p not in _KNOWN_PRED and p not in _LOGICAL)


def _to_subgoals(raw: List[Tuple[str, Optional[str], Optional[str]]]
                 ) -> List[Dict[str, Any]]:
    """谓词原文 → 子目标项（带 raw_index，供拓扑排序锚定）。"""
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
    return subs


def _strip_meta(ordered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """去掉排序用的内部字段，得到可落盘的子目标。"""
    for s in ordered:
        s.pop("raw_index", None)
        s.pop("anchor", None)
    return ordered


def parse_bddl_spec(bddl_path: str, language: str = "") -> Dict[str, Any]:
    """确定性解析 BDDL goal 为有序 TaskSpec（纯函数，不读缓存）。

    这是**参考实现**：既是 LLM 不可用时的 fallback，也是 LLM 分解的
    参考序（用于 diff/回归），以及 LLM 输出的交叉校验依据。
    """
    goal = _goal_text(bddl_path)
    ordered = _strip_meta(_order_subgoals(_to_subgoals(_raw_predicates(goal))))
    unsupported_tokens = sorted({s["predicate"] for s in ordered
                                 if s["kind"] == "unsupported"}
                                | set(_unknown_predicates(goal)))
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
    """优先读冻结的任务定义 YAML（LLM 分解产物），否则回退确定性解析。

    缓存由 `scripts/gen_task_specs.py` 离线生成并提交仓库——运行期不调 LLM。
    """
    p = cache_path(suite, idx)
    if p.exists():
        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            if data.get("subgoals"):
                data.setdefault("source", "frozen")
                return data
        except yaml.YAMLError:
            pass
    return parse_bddl_spec(bddl_path, language)


def save_cache(suite: str, idx: int, spec: Dict[str, Any]) -> Path:
    """原子写任务定义 YAML（tmp + rename）。"""
    TASK_SPECS_DIR.mkdir(parents=True, exist_ok=True)
    p = cache_path(suite, idx)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(yaml.safe_dump(spec, allow_unicode=True,
                                  sort_keys=False, default_flow_style=False),
                   encoding="utf-8")
    tmp.replace(p)
    return p


# ----------------- 形式化任务定义（写到 skills/configs 里）与 LLM 分解 ----------------
#
# 设计取舍：BDDL goal 本身已经是形式化规格，所以"把任务定义写到 skill 里"
# 不是把 BDDL 抄一遍，而是**物化**成一份可被 LLM 读、可被程序校验的自描述
# 任务定义：谓词表（每项带稳定 id）+ 允许插入的隐式前置 + 对象/目标词表 +
# 夹具属性 + 可用技能约束 + 确定性参考序。
#
# LLM 的自由度被刻意压到最小：**只能排列给定 id 的执行顺序，并只能从
# implicit_candidates 里挑要插入的隐式前置**。不得新增/删除/改写谓词与
# 对象名，不得选技能、不得给参数。这样"LLM 分解"既可核查（validate_order
# 是纯函数、可单测），又不会把基准的结论建立在不可复现的自由生成上。
#
# 冻结后运行期零 token、完全确定：`load_or_parse` 只读 YAML。

FORMAL_VERSION = 1

# 可用技能约束（给 LLM 的"你能用哪些谓词类别"说明；与 libero_planner
# 的谓词→技能映射表同源——这里只做声明，不参与执行）
SKILL_VOCABULARY: List[Dict[str, Any]] = [
    {"kind": "place", "predicates": ["On", "In"], "args": ["object", "target"],
     "note": "抓 object 放到 target（物体或 region）；In 要求 target 是 region"},
    {"kind": "articulate", "predicates": ["Open", "Close"], "args": ["target"],
     "note": "开/合可动夹具（抽屉、柜门、微波炉门）"},
    {"kind": "toggle", "predicates": ["Turnon", "Turnoff"], "args": ["target"],
     "note": "拧/按炉灶旋钮或按钮"},
]

_SYSTEM_PROMPT = """你是机器人任务分解器。输入是一份 LIBERO 任务的**形式化规格**：
- items：BDDL goal 里的子目标谓词，每项有唯一 id（这些是**全部**子目标，不可增删）
- implicit_candidates：允许插入的隐式前置项（可开合夹具的 Open），id 形如 "open:<夹具>"
- vocabulary：出现的对象名、目标名与夹具属性
- skills：谓词类别与技能类别的对应关系

你唯一的职责是**决定执行顺序**，并决定**是否需要插入隐式前置**。

硬约束（违反即判为无效输出）：
1. order 必须恰好是 items 全部 id 的一个排列，不重不漏；
2. add 只能取 implicit_candidates 里已有的 id，原样照抄；不需要就留空数组；
3. 不得新增、删除、改写任何谓词名、对象名或参数；
4. 只输出 JSON，不要 markdown 代码块，JSON 之外不要有任何文字。

输出格式：
{"order": ["p0", "p1"], "add": ["open:drawer_1"], "rationale": "一句话理由"}

物理常识（用于排序）：
- 往抽屉/柜子/微波炉里放东西之前，必须先打开它（对应 add 里的 open:<夹具>）；
- 往炉面上放锅之前要先点火（Turnon 排在 On(_, *_cook_region) 之前）；
- 同一夹具的 Close 排在"把东西放进去"之后；
- 其余谓词若 BDDL 原文顺序已表达执行次序（如先叠放再整体放入），沿用原文顺序。"""


def build_formal(bddl_path: str, language: str = "") -> Dict[str, Any]:
    """物化任务定义（不排序）。LLM 与 validator 都只读这一份。"""
    goal = _goal_text(bddl_path)
    subs = _to_subgoals(_raw_predicates(goal))

    items = [{"id": f"p{i}", "predicate": s["predicate"], "kind": s["kind"],
              "object": s["object"], "target": s["target"], "implicit": False}
             for i, s in enumerate(subs)]

    # 隐式前置候选：place 进"可开合 region"时通常需要先打开它
    implicit_candidates: List[Dict[str, Any]] = []
    seen: set = set()
    for s in subs:
        if s["kind"] == "place" and _is_articulated_region(s["target"] or ""):
            key = _fixture_key(s["target"])
            if key in seen:
                continue
            seen.add(key)
            implicit_candidates.append({
                "id": f"open:{key}", "predicate": "Open", "kind": "articulate",
                "object": None, "target": s["target"], "implicit": True,
            })

    targets = sorted({(s["target"] or "") for s in subs} - {""})
    fixtures = {_fixture_key(t): {"articulated": _is_articulated_region(t),
                                  "cook": bool(_COOK_HINT.search(t))}
                for t in targets}
    reference = _strip_meta(_order_subgoals([dict(s) for s in subs]))

    return {
        "version": FORMAL_VERSION,
        "bddl": str(bddl_path),
        "language": language,
        "items": items,
        "implicit_candidates": implicit_candidates,
        "vocabulary": {
            "objects": sorted({s["object"] for s in subs if s["object"]}),
            "targets": targets,
            "fixtures": fixtures,
        },
        "skills": SKILL_VOCABULARY,
        "unsupported": sorted({s["predicate"] for s in subs
                               if s["kind"] == "unsupported"}),
        "reference_order": [
            {"predicate": r["predicate"], "kind": r["kind"],
             "object": r["object"], "target": r["target"],
             "implicit": r.get("implicit", False)}
            for r in reference
        ],
    }


def _llm_messages(formal: Dict[str, Any]) -> List[Dict[str, str]]:
    payload = {
        "task": formal.get("language") or "",
        "items": [{k: it[k] for k in ("id", "predicate", "kind", "object", "target")}
                  for it in formal["items"]],
        "implicit_candidates": [{k: c[k] for k in ("id", "predicate", "target")}
                                for c in formal["implicit_candidates"]],
        "vocabulary": formal["vocabulary"],
        "skills": formal["skills"],
    }
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False,
                                               indent=1)},
    ]


def _extract_json(text: str) -> Any:
    """容忍 LLM 输出：先剥 ```json 围栏，否则取首个 { 到末个 }。"""
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", t, re.S)
    if m:
        t = m.group(1)
    else:
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            t = t[i:j + 1]
    return json.loads(t)


def validate_order(reply: Any, formal: Dict[str, Any]) -> Tuple[bool, str]:
    """校验 LLM 回复（纯函数，可单测）：id 已知、不重复、不漏、add 只取候选。"""
    if not isinstance(reply, dict):
        return False, "回复不是 JSON 对象"
    order = reply.get("order")
    if not isinstance(order, list) or not all(isinstance(x, str) for x in order):
        return False, "order 不是字符串数组"
    known = {it["id"] for it in formal["items"]}
    if len(set(order)) != len(order):
        return False, "order 里有重复 id"
    unknown = [x for x in order if x not in known]
    if unknown:
        return False, f"order 含未知 id: {unknown}"
    skip = reply.get("skip") or []
    if not isinstance(skip, list) or any(x not in known for x in skip):
        return False, "skip 非法（必须是已知 id 数组）"
    missing = known - set(order) - set(skip)
    if missing:
        return False, f"order 漏了 id: {sorted(missing)}"
    cand = {c["id"] for c in formal["implicit_candidates"]}
    add = reply.get("add") or []
    if not isinstance(add, list) or any(x not in cand for x in add):
        return False, f"add 只能取 implicit_candidates: {sorted(cand)}"
    return True, ""


def materialize(formal: Dict[str, Any], reply: Dict[str, Any]) -> List[Dict[str, Any]]:
    """按 LLM 的 order 展开 subgoals，并把 add 的隐式前置插到对应夹具的子目标前。"""
    by_id = {it["id"]: it for it in formal["items"]}
    add_by_id = {c["id"]: c for c in formal["implicit_candidates"]}
    pending = list(reply.get("add") or [])
    out: List[Dict[str, Any]] = []

    def _emit(node: Dict[str, Any], implicit: bool) -> None:
        out.append({"predicate": node["predicate"], "kind": node["kind"],
                    "object": node["object"], "target": node["target"],
                    "implicit": implicit})

    for pid in reply["order"]:
        it = by_id[pid]
        for aid in list(pending):
            cand = add_by_id[aid]
            if _fixture_key(cand["target"] or "") == _fixture_key(it["target"] or ""):
                _emit(cand, True)
                pending.remove(aid)
        _emit(it, False)
    for aid in pending:  # 没匹配上任何子目标的（如给一个没有 place 的柜子开盖）
        _emit(add_by_id[aid], True)
    return out


def decompose_spec(bddl_path: str, language: str, client: Any,
                   temperature: float = 0.0) -> Dict[str, Any]:
    """LLM 分解 BDDL 任务定义 → TaskSpec（离线生成脚本用）。

    校验不通过或 LLM 不可用时**回退确定性解析**，并把原因写进 spec["llm"]，
    保证脚本永远能产出一份可用的规格，且失败可追溯。
    """
    formal = build_formal(bddl_path, language)
    spec: Dict[str, Any] = {
        "bddl": str(bddl_path),
        "language": language,
        "unsupported": formal["unsupported"],
        "formal": formal,
        "reference_order": formal["reference_order"],
    }
    if client is None or not client.available():
        base = parse_bddl_spec(bddl_path, language)
        spec.update({"subgoals": base["subgoals"], "source": "deterministic",
                     "llm": {"ok": False, "error": "LLM 不可用（缺 api_key/model）"}})
        return spec

    raw = ""
    reply: Any = None
    try:
        raw = client.chat(_llm_messages(formal), temperature=temperature,
                          max_tokens=800)
        reply = _extract_json(raw)
        ok, why = validate_order(reply, formal)
    except Exception as e:  # 网络/解析异常都回退，不让生成脚本崩
        ok, why = False, f"{type(e).__name__}: {e}"

    if not ok:
        base = parse_bddl_spec(bddl_path, language)
        spec.update({"subgoals": base["subgoals"], "source": "deterministic",
                     "llm": {"ok": False, "error": why, "raw": (raw or "")[:2000]}})
        return spec

    spec["subgoals"] = materialize(formal, reply)
    spec["source"] = "llm"
    spec["llm"] = {
        "ok": True,
        "model": getattr(client, "model", ""),
        "temperature": temperature,
        "order": reply["order"],
        "add": reply.get("add") or [],
        "skip": reply.get("skip") or [],
        "rationale": (reply.get("rationale") or "")[:500],
        "raw": (raw or "")[:2000],
    }
    return spec

