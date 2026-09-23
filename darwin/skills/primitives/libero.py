"""LIBERO demo 回放动作原语（仅保留 demo 逐帧回放）。

IK 末端闭环伺服已下沉到通用 `ik_servo.py`（IkServoSkill/CarrySkill，
跨 env 共享，调 `primitives.servo_step`），不再在此放 libero 专用伺服链。
本模块只保留 `libero_action`：把任务 demo_0 的 robosuite 7 维动作逐帧
回放（ReplayDemoMethod 用），由 reset 时回到 demo 录制初始状态保证确定性。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..base import Skill, SkillSpec, SkillKind, Confidence


class LiberoActionSkill(Skill):
    """执行单条 robosuite 7 维动作（demo 回放兜底用）。"""

    spec = SkillSpec(
        name="libero_action",
        description="Apply one robosuite OSC 7-DoF action [dx,dy,dz,dax,day,daz,gripper]",
        kind=SkillKind.PRIMITIVE, confidence=Confidence.VERIFIED,
        applies_when="LIBERO demo 回放链",
    )

    def execute(self, env, action: List[float],
                actor: Optional[str] = None,
                grip_site: Optional[str] = None) -> Dict[str, Any]:
        act = np.asarray(action, dtype=np.float64).reshape(-1)
        obs, reward, done, info = env.step(act)
        return {"success": True, "steps": 1, "done": bool(done)}


__all__ = ["LiberoActionSkill"]
