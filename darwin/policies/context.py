"""StepContext：policy 的输入——"当时的情况"。

见 docs/architecture_step_planner.md §2。随 Phase A/B 推进逐步充实；
当前字段已覆盖 grasp_pose 规则与步级重试的需要。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StepContext:
    step: str                      # "grasp_pose" | "descend" | "place" | ...
    env: Any = None                # 执行环境（libero_adapter / robopal）
    body: str = ""                 # 目标物体 body 名
    entry: Dict[str, Any] = field(default_factory=dict)
    cfg: Dict[str, Any] = field(default_factory=dict)   # per-env 底参（anchor）
    attempt: int = 0               # 本 episode 第几次尝试
    retry: int = 0                 # 本步第几次重试（0=首次）
    features: Dict[str, Any] = field(default_factory=dict)
    fail_history: List[Dict[str, Any]] = field(default_factory=list)
