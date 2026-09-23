"""技能基类与元数据（融合 AgentFactory SKILL.md + RPent TOOLS_SPEC + RPent memory 元数据）。

技能 = 可执行 Python 代码 + SKILL.md 元数据。成功解法直接沉淀为代码技能（AgentFactory 核心思想），
而非文本经验。所有技能通过 registry 暴露为 OpenAI function calling 格式供 planner 调用。
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class SkillKind(str, Enum):
    """记忆/技能类型（来自 RPent KINDS）。"""
    PRIMITIVE = "primitive"      # 原子操作技能
    PERCEPTION = "perception"    # 感知技能（SAM/YOLO 等）
    STRATEGY = "strategy"        # 策略技能（多步组合）
    FAILURE = "failure"          # 避坑条目
    INFRA = "infra"              # 基础设施


class Confidence(str, Enum):
    """置信度（来自 RPent CONFIDENCE）。evidence.cells≥3 且 tasks≥2 自动升级 verified。"""
    SINGLE_SHOT = "single-shot"
    PROBABLE = "probable"
    VERIFIED = "verified"


@dataclass
class Evidence:
    """证据（来自 RPent evidence 结构）。"""
    cells: List[str] = field(default_factory=list)        # 验证单元（唯一标识，如 task_seed）
    tasks: List[str] = field(default_factory=list)        # 已验证过的任务（显式维度，跨任务才升 verified）
    attempts: int = 0                                     # 总尝试次数
    solved_seeds: List[str] = field(default_factory=list)
    failed_seeds: List[str] = field(default_factory=list)
    contradicted_by: List[str] = field(default_factory=list)  # 矛盾记忆 ID（失败归因）

    def merge(self, other: "Evidence") -> "Evidence":
        cells = sorted({*self.cells, *other.cells})
        tasks = sorted({*self.tasks, *other.tasks})
        confidence = (
            Confidence.VERIFIED.value if len(cells) >= 3 and len(tasks) >= 2
            else Confidence.PROBABLE.value if len(cells) >= 2
            else Confidence.SINGLE_SHOT.value
        )
        return Evidence(
            cells=cells,
            tasks=tasks,
            attempts=self.attempts + other.attempts,
            solved_seeds=sorted({*self.solved_seeds, *other.solved_seeds}),
            failed_seeds=sorted({*self.failed_seeds, *other.failed_seeds}),
            contradicted_by=sorted({*self.contradicted_by, *other.contradicted_by}),
        ), confidence


@dataclass
class SkillSpec:
    """技能元数据（AgentFactory SKILL.md frontmatter + RPent TOOLS_SPEC）。"""
    name: str
    description: str
    kind: SkillKind = SkillKind.PRIMITIVE
    confidence: Confidence = Confidence.SINGLE_SHOT
    applies_when: str = ""          # 触发条件（自然语言 + 结构化标签）
    entry_file: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)
    evidence: Evidence = field(default_factory=Evidence)
    supersedes: Optional[str] = None  # 替代的旧技能名（AgentFactory supersedes 机制）

    def to_tool_spec(self) -> Dict[str, Any]:
        """导出为 OpenAI function calling 格式（RPent TOOLS_SPEC）。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": self.parameters,
                    "required": self.required,
                },
            },
        }


class Skill:
    """技能基类。子类实现 execute，注册时自动生成 spec。"""
    spec: SkillSpec

    def execute(self, env, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def get_skill_description(self) -> str:
        """AgentFactory 前置描述检查：planner 调用前必须读取。"""
        ev = self.spec.evidence
        return (
            f"技能: {self.spec.name}\n"
            f"类型: {self.spec.kind.value}\n"
            f"置信度: {self.spec.confidence.value}\n"
            f"适用条件: {self.spec.applies_when}\n"
            f"描述: {self.spec.description}\n"
            f"证据: {len(ev.cells)} cells, {ev.attempts} attempts, "
            f"{len(ev.solved_seeds)} solved"
        )


def make_spec(name: str, description: str, fn: Callable, **kwargs) -> SkillSpec:
    """从函数签名自动生成 SkillSpec（参数/必填项）。"""
    sig = inspect.signature(fn)
    parameters = {}
    required = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "env"):
            continue
        ptype = param.annotation if param.annotation is not inspect.Parameter.empty else "any"
        parameters[pname] = {"type": str(ptype).__name__ if not isinstance(ptype, str) else ptype}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    return SkillSpec(name=name, description=description, parameters=parameters, required=required, **kwargs)
