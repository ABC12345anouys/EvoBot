"""臂可达性预检（policy 侧轻量包络，env 侧 cuRobo IK 是最终裁决）。

设计：柱面包络近似（锚点 = episode 开始实测 home ee xy，免标定基座）。
标定依据（/tmp/darwin_obs_dump.pkl 实测）：
- left 到 bowl2(x=0.249) rho=0.632（IK 必 stall，须拒）；right 到 bowl2 rho=0.318
- left home(-0.30) 到 bowl1(x=-0.274) rho≈0.29；home 到 bowl0(x=0.043) rho=0.403
  是历史成功抓取参照 → reach_r_max=0.48 留少量余量。
只做初筛：误杀由 stall 日志校 YAML；漏放行由 stall 兜底弹下一候选。
"""
from __future__ import annotations

import numpy as np

_DEFAULTS = dict(
    reach_r_min=0.12,
    reach_r_max=0.48,
    reach_rho_comfort=0.38,
    reach_z_min=0.55,
    reach_z_max=1.00,
)


class ReachabilityModel:
    def __init__(self, cfg: dict | None = None):
        self.cfg = dict(_DEFAULTS)
        if cfg:
            self.cfg.update({k: v for k, v in cfg.items() if k in _DEFAULTS})
        self._anchors: dict[str, np.ndarray] = {}

    def configure_arm(self, arm: str, home_ee_xy) -> None:
        """episode 开始时用实测 home ee xy 设锚点（免标定基座）。"""
        self._anchors[arm] = np.asarray(home_ee_xy, float)[:2]

    def _rho(self, arm: str, tip_xy) -> float | None:
        anchor = self._anchors.get(arm)
        if anchor is None:
            return None
        return float(np.linalg.norm(np.asarray(tip_xy, float)[:2] - anchor))

    def ok(self, arm: str, tip_xyz, margin: float = 0.0) -> bool:
        """tip（指尖目标，world 系）是否在该臂柱面包络内。"""
        c = self.cfg
        rho = self._rho(arm, tip_xyz)
        if rho is None:
            return True          # 未配置的臂不预检（保守放行）
        z = float(np.asarray(tip_xyz, float)[2])
        return bool(c["reach_r_min"] - margin <= rho <= c["reach_r_max"] + margin
                    and c["reach_z_min"] - margin <= z <= c["reach_z_max"] + margin)

    def rank_arms(self, tip_xy, arms: list | None = None) -> list:
        """按 |rho - comfort| 升序返回候选臂（未配置的排最后）。"""
        c = self.cfg
        arms = arms if arms is not None else list(self._anchors)
        scored = []
        for a in arms:
            rho = self._rho(a, tip_xy)
            scored.append((float("inf") if rho is None else abs(rho - c["reach_rho_comfort"]), a))
        scored.sort()
        return [a for _, a in scored]

    def stats(self, arm: str, tip_xyz) -> dict:
        rho = self._rho(arm, tip_xyz)
        return {"arm": arm, "rho": None if rho is None else round(rho, 3),
                "r_range": [self.cfg["reach_r_min"], self.cfg["reach_r_max"]]}
