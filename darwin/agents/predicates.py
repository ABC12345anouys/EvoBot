"""通用几何谓词：在 WorldState 上求值。

谓词是任务目标的通用语言：LLM 把指令翻译成谓词表达式作为 goal，
执行端只做谓词求值，不按任务名分支。新任务（stack_blocks、
pour_into、hang_mugs…）复用同一组谓词。

每个谓词返回 (bool, measures)：measures 给出具体数值（间隙/距离），
既是日志也是重规划反馈。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from .world_state import WorldState

_PRED_DOCS = {
    "on": "实体 A 正放在 B 上（xy 对齐 + A 底部与 B 顶部贴合）",
    "near": "A 与 B 的水平距离在 tol 内",
    "above": "A 位于 B 正上方（xy 对齐，不要求贴合）",
    "lifted": "A 被抬离桌面至少 height 米",
    "in": "A 中心位于容器 B 的开口范围内",
}


def _ent(state: WorldState, label: str):
    e = state.get(label)
    if e is None:
        raise KeyError(f"谓词引用了未知实体 {label!r}（当前 {state.labels()}）")
    return e


def on(state: WorldState, a: str, b: str,
       xy_tol: float = 0.04, dz_tol: float = 0.06) -> Tuple[bool, Dict[str, Any]]:
    A, B = _ent(state, a), _ent(state, b)
    dxy = float(((A.x - B.x) ** 2 + (A.y - B.y) ** 2) ** 0.5)
    gap = A.z_bottom - B.z_top
    ok = dxy < xy_tol and -0.02 <= gap <= dz_tol
    return ok, {"dxy": round(dxy, 3), "xy_tol": xy_tol,
                "vertical_gap": round(gap, 3), "dz_tol": dz_tol}


def near(state: WorldState, a: str, b: str,
         tol: float = 0.05) -> Tuple[bool, Dict[str, Any]]:
    A, B = _ent(state, a), _ent(state, b)
    dxy = float(((A.x - B.x) ** 2 + (A.y - B.y) ** 2) ** 0.5)
    return dxy < tol, {"dxy": round(dxy, 3), "tol": tol}


def above(state: WorldState, a: str, b: str,
          xy_tol: float = 0.04) -> Tuple[bool, Dict[str, Any]]:
    A, B = _ent(state, a), _ent(state, b)
    dxy = float(((A.x - B.x) ** 2 + (A.y - B.y) ** 2) ** 0.5)
    return (dxy < xy_tol and A.z_bottom >= B.z_top - 0.01,
            {"dxy": round(dxy, 3), "xy_tol": xy_tol})


def lifted(state: WorldState, a: str,
           height: float = 0.05) -> Tuple[bool, Dict[str, Any]]:
    A = _ent(state, a)
    h = A.z_bottom - state.table_z
    return h >= height, {"lift_height": round(h, 3), "height": height}


def in_(state: WorldState, a: str, b: str,
        margin: float = 0.0) -> Tuple[bool, Dict[str, Any]]:
    A, B = _ent(state, a), _ent(state, b)
    ex = abs(A.x - B.x) - (B.half[0] - margin)
    ey = abs(A.y - B.y) - (B.half[1] - margin)
    return (ex < 0 and ey < 0,
            {"edge_x": round(ex, 3), "edge_y": round(ey, 3)})


_FUNCS = {"on": on, "near": near, "above": above,
          "lifted": lifted, "in": in_}


def predicate_docs() -> Dict[str, str]:
    return dict(_PRED_DOCS)


def evaluate(spec: Dict[str, Any], state: WorldState) -> Tuple[bool, Dict[str, Any]]:
    """求值一个谓词声明：{"pred": "on", "args": ["p2","p1"], "params": {...}}。"""
    name = spec.get("pred")
    fn = _FUNCS.get(name)
    if fn is None:
        raise KeyError(f"未知谓词 {name!r}（可用 {sorted(_FUNCS)}）")
    args = spec.get("args") or []
    params = spec.get("params") or {}
    return fn(state, *args, **params)


def describe(spec: Dict[str, Any]) -> str:
    name = spec.get("pred", "?")
    args = ", ".join(str(a) for a in (spec.get("args") or []))
    return f"{name}({args})"
