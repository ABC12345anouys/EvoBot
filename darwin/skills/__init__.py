"""darwin.skills：技能集合（agent 直接持有实例，无 registry 分发）。

启动时加载所有原子技能 + 感知技能 + forged 技能，返回 Dict[str, Skill]。
agent 直接 self.skills[name].execute(env, **params) 调用，不经过分发层。
"""
from .base import Skill, SkillSpec, SkillKind, Confidence, Evidence, make_spec


def load_skills() -> dict:
    """构建默认技能表（感知 + 原语 + forged），返回 {name: Skill}。"""
    skills: dict = {}

    # 感知技能：YOLO + SAM + GraspNet
    from .perception.detect import DetectObjectsSkill, EstimateDepthSkill
    from .perception.segment import SegmentObjectSkill
    from .perception.grasp import GraspPoseSkill
    for s in (DetectObjectsSkill(), EstimateDepthSkill(), SegmentObjectSkill(), GraspPoseSkill()):
        skills[s.spec.name] = s

    # 原子操作技能（primitives 目录下的手写技能，按需导入）
    try:
        from . import primitives
        for attr in dir(primitives):
            obj = getattr(primitives, attr)
            if isinstance(obj, type) and issubclass(obj, Skill) and obj is not Skill:
                try:
                    inst = obj()
                    skills[inst.spec.name] = inst
                except Exception:
                    pass
    except ImportError:
        pass

    # 加载 forged 技能
    from .forged import load_forged_skills
    skills.update(load_forged_skills())
    return skills


__all__ = [
    "Skill", "SkillSpec", "SkillKind", "Confidence", "Evidence", "make_spec",
    "load_skills",
]
