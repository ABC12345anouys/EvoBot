"""证据采集：skill 执行期的物理观测 → Evidence（附录 A 循环的 observe 步）。

设计要点：
- 成功与失败同样产生证据（遥测零浪费）——一次干净成功的 descend
  也测量了接触 z 真值与可达短缩。
- 接触力采样是 O(nbody) 遍历，不逐步做：默认每 sample_force_every 步
  一次 + finish 时补一次终值（判别子只用窗口值与终值）。
- 全部观测量为物体中心坐标系或臂无关物理量（表 3 判据的实现基础）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

# 自由空间接触力阈值（N）：低于此值认为指尖与物体无接触。
# MuJoCo 静止接触在 OSC 持续推压下通常 >>1N；0.5 给足噪声裕度。
F_FREE_N = 0.5

# z 进展停滞阈值（m/步）：低于此速率视为"停"。
RATE_EPS = 0.0002  # 0.2mm/步；spatial:4 慢型 0.3mm/步 > 此值不误判


@dataclass
class Evidence:
    """一次 skill 执行的物理证据。trace 按采样序 append。"""

    step: str = ""
    body: Optional[str] = None
    z_trace: List[float] = field(default_factory=list)
    xy_trace: List[List[float]] = field(default_factory=list)
    f_trace: List[float] = field(default_factory=list)   # 与 f_steps 对齐
    f_steps: List[int] = field(default_factory=list)
    t_trace: List[int] = field(default_factory=list)
    clearance_min: float = 1.0
    coll_pair: str = ""
    goal_z: Optional[float] = None
    # 结果字段（skill 回填）
    success: bool = False
    reason: str = ""

    # ---- 派生量（判别子输入）----
    def z_rate_last(self, window: int = 30) -> float:
        """最近 window 个 z 样本的平均下降速率（m/样本，正值=下降）。"""
        if len(self.z_trace) < 2:
            return 0.0
        zs = self.z_trace[-window - 1:]
        if len(zs) < 2:
            return 0.0
        return float(zs[0] - zs[-1]) / (len(zs) - 1)

    def z_rate_median(self, window: int = 30) -> float:
        """窗口内逐段速率的中位数（抗单步噪声；uniform 慢速的充分统计）。"""
        if len(self.z_trace) < 4:
            return self.z_rate_last(window)
        zs = np.asarray(self.z_trace[-window - 1:], float)
        d = zs[:-1] - zs[1:]
        return float(np.median(d)) if len(d) else 0.0

    @property
    def f_at_end(self) -> float:
        return self.f_trace[-1] if self.f_trace else 0.0

    def xy_err_at_end(self, pt_xy) -> float:
        if not self.xy_trace:
            return float("inf")
        e = self.xy_trace[-1]
        return float(np.hypot(e[0] - pt_xy[0], e[1] - pt_xy[1]))

    def to_dict(self) -> Dict[str, Any]:
        """落 jsonl 的紧凑形式（trace 降采样，防 episode 文件膨胀）。"""
        n = len(self.z_trace)
        stride = max(1, n // 50)  # 至多 50 点
        zs = self.z_trace[::stride]
        if zs and zs[-1] != self.z_trace[-1]:
            zs = zs + [self.z_trace[-1]]  # 终点必须保留（reach/cut 用）
        return {
            "step": self.step, "body": self.body,
            "n": n, "goal_z": self.goal_z,
            "z_trace": zs,
            "z_rate_median": round(self.z_rate_median(), 6),
            "f_at_end": round(self.f_at_end, 3),
            "f_max": round(max(self.f_trace), 3) if self.f_trace else 0.0,
            "xy_end": self.xy_trace[-1] if self.xy_trace else None,
            "xy_err_end": None,  # skill 层可回填
            "clearance_min": round(self.clearance_min, 4),
            "coll_pair": self.coll_pair,
            "success": self.success, "reason": self.reason,
        }


class Collector:
    """skill 执行循环内的采样器。用法：

        col = Collector(env, site, step="descend", body=body)
        for t in range(timeout):
            eef = env.get_site_pos(site)
            col.sample(t, eef)
            ...
        ev = col.finish(success, reason)
    """

    def __init__(self, env, site: str, *, step: str = "",
                 body: Optional[str] = None,
                 sample_force_every: int = 10):
        self.env = env
        self.site = site
        self.ev = Evidence(step=step, body=body)
        self._every = max(1, int(sample_force_every))
        self._last_f: Optional[float] = None

    def _force(self) -> Optional[float]:
        if not self.ev.body:
            return None
        try:
            return float(self.env.contact_force_on_body(self.ev.body))
        except Exception:
            return None

    def sample(self, t: int, eef) -> None:
        e = np.asarray(eef, float)
        self.ev.t_trace.append(int(t))
        self.ev.z_trace.append(float(e[2]))
        self.ev.xy_trace.append([float(e[0]), float(e[1])])
        if self.ev.body and (t % self._every == 0):
            f = self._force()
            if f is not None:
                self.ev.f_trace.append(f)
                self.ev.f_steps.append(int(t))

    def sample_force(self, t: int) -> Optional[float]:
        """窗口判定点（如 stall 判定时）强制采一次力。"""
        if not self.ev.body:
            return None
        f = self._force()
        if f is not None:
            self.ev.f_trace.append(f)
            self.ev.f_steps.append(int(t))
            self._last_f = f
        return f

    def finish(self, success: bool, reason: str = "",
               clearance_min: float = 1.0, coll_pair: str = "",
               goal_z: Optional[float] = None) -> Evidence:
        # 终态力补采：判别子 f_at_end 必须有值
        if self.ev.body and (not self.ev.f_trace
                             or self.ev.f_steps[-1] != (self.ev.t_trace[-1] if self.ev.t_trace else -1)):
            self.sample_force(self.ev.t_trace[-1] if self.ev.t_trace else 0)
        self.ev.success = bool(success)
        self.ev.reason = reason
        self.ev.clearance_min = float(clearance_min)
        self.ev.coll_pair = coll_pair
        self.ev.goal_z = goal_z
        return self.ev
