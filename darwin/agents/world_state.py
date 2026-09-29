"""WorldState：规划器与谓词共同使用的统一世界状态。

任务无关：实体几何全部来自感知跟踪（Instance），被夹持实体的位置由
ee + held_off 预测。to_json() 的输出就是 LLM 看到的世界。

新任务（stack_blocks / pour / hang…）不改本模块：换的只是指令和
LLM 产出的谓词/技能序列。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Entity:
    """一个可操作对象（标签是策略侧虚拟身份，跨感知刷新保持）。"""
    label: str
    x: float
    y: float
    z_top: float
    z_bottom: float
    half: List[float]              # [hx, hy, hz]
    container: bool = False
    held: bool = False

    @property
    def z_center(self) -> float:
        return (self.z_top + self.z_bottom) / 2.0

    @property
    def pos3(self) -> List[float]:
        return [self.x, self.y, self.z_center]

    def to_json(self) -> Dict[str, Any]:
        return {"label": self.label, "x": round(self.x, 3),
                "y": round(self.y, 3),
                "z_top": round(self.z_top, 3),
                "z_bottom": round(self.z_bottom, 3),
                "half": [round(float(h), 3) for h in self.half],
                "container": self.container,
                **({"held": True} if self.held else {})}


@dataclass
class WorldState:
    table_z: float
    entities: List[Entity] = field(default_factory=list)
    homes: Dict[str, List[float]] = field(default_factory=dict)

    def get(self, label: str) -> Optional[Entity]:
        return next((e for e in self.entities if e.label == label), None)

    def labels(self) -> List[str]:
        return [e.label for e in self.entities]

    def to_json(self) -> Dict[str, Any]:
        return {
            "table_z": round(float(self.table_z), 3),
            "arms": {a: {"home": [round(float(v), 3) for v in xyz]}
                     for a, xyz in sorted(self.homes.items())},
            "entities": [e.to_json() for e in self.entities],
        }


def build_world_state(table_z: float,
                      track: Dict[str, Any],
                      predicted: Dict[str, np.ndarray],
                      held: Optional[str],
                      homes: Dict[str, Any]) -> WorldState:
    """从跟踪表装配 WorldState（model 侧 composition root 调用）。

    track: label → Instance；predicted: label → 质心先验（held 实体
    用 ee+held_off 预测，由调用方写好）；homes: arm → (ee_pose, quat)。
    """
    ents: List[Entity] = []
    for label, it in track.items():
        c = np.asarray(predicted.get(label, it.centroid), float)
        half = np.asarray(it.aabb_half, float)
        ents.append(Entity(
            label=label, x=float(c[0]), y=float(c[1]),
            # c 是实体中心（held 实体的中心为 ee+held_off 预测值）
            z_top=float(c[2]) + float(half[2]),
            z_bottom=float(c[2]) - float(half[2]),
            half=[float(half[0]), float(half[1]), float(half[2])],
            container=bool(it.is_container), held=(label == held)))
    home_xyz = {a: np.asarray(pose, float)[:3].tolist()
                for a, (pose, _q) in homes.items()}
    return WorldState(table_z=float(table_z), entities=ents, homes=home_xyz)
