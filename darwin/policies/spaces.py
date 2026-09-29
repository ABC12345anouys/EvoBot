"""ParamSpace：约束层的显式表示。

每个参数的可行域 = PARAM_SPEC 范围 ∩（可选）anchor 邻域。policy 提案
超出域时被裁剪并记录——"脚链"的落点，见 docs/architecture_step_planner.md
§2/§11。Phase A 只做范围裁剪（与现状 PARAM_SPEC 校验同语义）。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ..skills.skill_config import PARAM_SPEC


class ParamSpace:
    def __init__(self, anchor: Optional[Dict[str, Any]] = None):
        self.anchor = dict(anchor or {})

    def clip(self, params: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """把提案裁剪进可行域，返回 (裁剪后, 被裁剪的参数名)。"""
        out, clipped = dict(params), []
        for k, v in out.items():
            spec = PARAM_SPEC.get(k)
            if spec is None or not isinstance(v, (int, float)):
                continue
            typ, _, lo, hi, _ = spec
            cv = min(hi, max(lo, float(v)))
            if cv != float(v):
                out[k] = typ(cv)
                clipped.append(k)
        return out, clipped

    def neighborhood(self, key: str, ratio: float = 0.5,
                     floor: float = 0.0) -> Tuple[Any, Any]:
        """anchor 邻域（反思 _DRIFT_FLOOR 同语义），供重试探索采样。"""
        spec = PARAM_SPEC.get(key)
        if spec is None:
            return None, None
        _, default, lo, hi, _ = spec
        a = self.anchor.get(key, default)
        try:
            a = float(a)
        except (TypeError, ValueError):
            a = float(default)
        w = max(ratio * abs(a), floor)
        return max(lo, a - w), min(hi, a + w)
