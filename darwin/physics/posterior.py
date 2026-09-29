"""Θ 后验：未知物理标量的半空间交 + 置信度 + per-task 持久化。

设计（§15.1）：知识 = 约束实例 (判据, Θ, 裕度)。只有 Θ 里的标量需要
学（μ、质量级、可达短缩、感知偏置，k≤3~5），其余一切解析推导。
每个失败贡献一条不等式（cuts.py），可行集单调收缩，无一步浪费。

持久化：ik_servo.<env>.yaml 的 `theta:` 段（与 params 平级）。
语义与 params 的根本区别：params 是"本次怎么跑"，theta 是"这个世界
的物理事实"——后者跨 attempt、跨任务（经记忆库检索）复用。
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

# 默认先验域（宽，物理意义约束的夹紧域；Θ 才是 learned 的部分）
_DEFAULT_PRIORS: Dict[str, tuple] = {
    "mu": (0.05, 2.0),            # 摩擦系数物理域
    "reach_shortfall": (0.0, 0.15),  # OSC 到达短缩（m，构型相关）
    "timeout_need": (60, 2000),   # 步数预算下界推导域
}


class ThetaPosterior:
    """一个任务的 Θ 后验。lo/hi 单调收缩；conf ∈ (0,1] 加权更新。"""

    def __init__(self, theta: Optional[Dict[str, Dict[str, float]]] = None):
        # theta: {name: {"lo":..,"hi":..,"conf":..,"n":..}}
        self._t: Dict[str, Dict[str, float]] = {}
        for name, d in (theta or {}).items():
            self._t[name] = dict(d)

    # ---- 割（半空间交，软更新按置信度）----
    def cut(self, name: str, *, lo: Optional[float] = None,
            hi: Optional[float] = None, conf: float = 1.0,
            prior: Optional[tuple] = None) -> Dict[str, float]:
        """加入一条不等式 Θ_name ∈ [lo, hi]（单边可省略）。

        软更新：只在置信度高于已有记录时收紧，防误分类切错方向
        （§15.6 头号风险）。返回收紧后的域。
        """
        p_lo, p_hi = prior or _DEFAULT_PRIORS.get(name, (0.0, math.inf))
        cur = self._t.setdefault(name, {"lo": p_lo, "hi": p_hi,
                                        "conf": 0.0, "n": 0})
        old_lo, old_hi, old_conf = cur["lo"], cur["hi"], cur["conf"]
        new_lo, new_hi = old_lo, old_hi
        if lo is not None:
            cand = max(old_lo, float(lo))
            if conf >= old_conf:
                new_lo = cand
        if hi is not None:
            cand = min(old_hi, float(hi))
            if conf >= old_conf:
                new_hi = cand
        if new_hi < new_lo:  # 矛盾：不交，保宽域并降置信（升级路径 §15.6）
            new_lo, new_hi = old_lo, old_hi
            cur["conf"] = old_conf * 0.5
            cur["contradiction"] = 1
        else:
            cur["lo"], cur["hi"] = new_lo, new_hi
            cur["conf"] = max(old_conf, conf)
            cur["n"] = int(cur.get("n", 0)) + 1
        return dict(cur)

    # ---- 查询 ----
    def domain(self, name: str) -> tuple:
        cur = self._t.get(name)
        if not cur:
            p = _DEFAULT_PRIORS.get(name, (0.0, math.inf))
            return p
        return (cur["lo"], cur["hi"])

    def mid(self, name: str) -> float:
        lo, hi = self.domain(name)
        if math.isinf(hi):
            return lo
        return 0.5 * (lo + hi)

    def has(self, name: str) -> bool:
        return name in self._t

    # ---- 测量直写（探针结果：不是割，是直接观测）----
    def measure(self, name: str, value: float, *,
                err: float = 0.2, conf: float = 0.9,
                prior: Optional[tuple] = None) -> Dict[str, float]:
        """探针测量值带误差带写入：Θ ∈ [v/(1+err), v/(1-err)]。"""
        lo = value * (1.0 - err)
        hi = value * (1.0 + err) if err < 1.0 else value * 2.0
        return self.cut(name, lo=lo, hi=hi, conf=conf, prior=prior)

    # ---- 持久化 ----
    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {k: dict(v) for k, v in self._t.items()}

    @staticmethod
    def load(cfg: Dict[str, Any]) -> "ThetaPosterior":
        """从 per-env cfg（SkillConfigStore deep_merge 后的 dict）读。"""
        return ThetaPosterior(cfg.get("theta") if isinstance(cfg, dict) else None)


def body_mass(env, body: str) -> Optional[float]:
    """物体质量（kg）——摩擦锥割 μ·F ≥ m·g 的输入。读不到返回 None。"""
    try:
        bid = env.mj_model.body_name2id(body)
        return float(env.mj_model.body_mass[bid])
    except Exception:
        return None
