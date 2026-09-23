"""darwin.agents：自进化操作智能体。

核心：ManipulationAgent（持有 skills + rag + llm，闭环 perceive→retrieve→decide→execute→reflect）。
"""
from .agent import ManipulationAgent, EpisodeResult

__all__ = ["ManipulationAgent", "EpisodeResult"]
