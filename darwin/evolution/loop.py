"""进化主循环（融合 MAP-Elites + skill_forge + replay 防退化）。

流程：
1. 从 benchmark 任务分布采样 episode
2. agent 执行（带记忆增强），verifier 逐步校验
3. 成功 → skill_forge 生成代码技能，replay 验证后入库
4. 失败 → failure_miner 写避坑记忆
5. MAP-Elites 维护多样化技能库
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .map_elites import MapElites
from .skill_forge import forge_skill_from_trajectory


class EvolutionLoop:
    def __init__(
        self,
        benchmark: Dict[str, Any],
        *,
        search_space: Dict[str, List[Any]],
        feature_dimensions: List[str],
        feature_ranges: Dict[str, tuple],
        n_iterations: int = 50,
        memory_store=None,
    ) -> None:
        self.memory = memory_store
        self.benchmark = benchmark
        self.map_elites = MapElites(
            search_space=search_space,
            evaluator=self._evaluate,
            feature_dimensions=feature_dimensions,
            feature_ranges=feature_ranges,
            n_iterations=n_iterations,
        )
        self.results: List[Dict] = []

    def run(self, agent_runner: Callable) -> Dict[str, Any]:
        """agent_runner(task_cfg) -> {success, trajectory, info, metrics}"""
        # MapElites 契约：evaluator(cfg) 执行 cfg 并返回 metrics（program 返回值会被丢弃）
        self.map_elites.evaluator = lambda cfg: self._run_episode(cfg, agent_runner)
        best_cfg, info = self.map_elites.run()
        return {"best_cfg": best_cfg, "archive_size": len(info["archive"]),
                "history": info["history"]}

    def _run_episode(self, cfg: Dict, agent_runner: Callable) -> Dict[str, Any]:
        result = agent_runner(cfg)
        success = result.get("success", False)
        trajectory = result.get("trajectory", [])
        # cfg 只含 MAP-Elites 搜索超参（hover/k_descend/...），任务名取自 benchmark
        task_name = cfg.get("task_name") or self.benchmark.get("task_name", "unknown")

        if success:
            # 失败记忆已由 agent.reflect 写入 RAG；成功轨迹沉淀为 forged skill
            forge_skill_from_trajectory(trajectory, task_name=task_name,
                                         feat=result.get("info", {}).get("feat"))

        self.results.append({"cfg": cfg, "success": success, **result.get("metrics", {})})
        return result.get("metrics", {"score": 0.0})

    def _evaluate(self, metrics: Dict) -> Dict[str, Any]:
        return metrics
