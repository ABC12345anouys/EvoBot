"""MAP-Elites 质量多样性进化器（来自 EvoAgentX map_elites_optimizer.py，适配 darwin 技能空间）。

核心思想：不只找最优解，而是按特征维度分箱，每个 cell 保留最优技能，维护多样化技能库。
- feature_dimensions: 任务族 / 操作类型 / 复杂度
- 每 cell 只存最优 cfg（fitness = 成功率 × (1 - 归一化步数)）
- exploration_ratio: 0.2 → 80% 变异利用，20% 随机探索
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ArchiveEntry:
    cfg: Dict[str, Any]
    fitness: float
    metrics: Dict[str, Any]
    cell: Tuple[int, ...]


class MapElites:
    def __init__(
        self,
        search_space: Mapping[str, List[Any]],
        evaluator: Callable[[Dict[str, Any]], Dict[str, Any]],
        *,
        feature_dimensions: List[str],
        feature_ranges: Mapping[str, Tuple[float, float]],
        feature_bins: int = 10,
        fitness_key: str = "score",
        n_iterations: int = 100,
        exploration_ratio: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        self.search_space = dict(search_space)
        self.evaluator = evaluator
        self.feature_dimensions = list(feature_dimensions)
        self.feature_ranges = dict(feature_ranges)
        self.feature_bins = feature_bins
        self.fitness_key = fitness_key
        self.n_iterations = n_iterations
        self.exploration_ratio = exploration_ratio
        if seed is not None:
            random.seed(seed)

        missing = [d for d in self.feature_dimensions if d not in self.feature_ranges]
        if missing:
            raise ValueError(f"feature_ranges 缺失维度: {missing}")

    def run(self, program: Optional[Callable[[], Dict[str, Any]]] = None):
        """主循环。program() 执行一个 cfg 并返回 metrics；evaluator(metrics) 给 fitness。"""
        archive: Dict[Tuple[int, ...], ArchiveEntry] = {}
        history: List[Dict[str, Any]] = []
        best: Optional[ArchiveEntry] = None

        for step in range(self.n_iterations):
            if archive and random.random() > self.exploration_ratio:
                parent = random.choice(list(archive.values()))
                cfg = self._mutate(parent.cfg)
                source = "mutate"
            else:
                cfg = self._random_cfg()
                source = "random"

            if program is not None:
                program(cfg)
            metrics = self.evaluator(cfg)
            fitness = metrics[self.fitness_key]
            cell = self._cell(metrics)

            accepted = False
            existing = archive.get(cell)
            if existing is None or fitness > existing.fitness:
                entry = ArchiveEntry(cfg=cfg, fitness=fitness, metrics=metrics, cell=cell)
                archive[cell] = entry
                accepted = True
                if best is None or entry.fitness > best.fitness:
                    best = entry

            history.append({"step": step, "source": source, "cfg": cfg,
                            "fitness": fitness, "cell": cell, "accepted": accepted})

        return (best.cfg if best else None), {"archive": archive, "history": history, "best": best}

    # ---- 内部 ----

    def _random_cfg(self) -> Dict[str, Any]:
        return {k: copy.deepcopy(random.choice(v)) for k, v in self.search_space.items()}

    def _mutate(self, parent: Dict[str, Any]) -> Dict[str, Any]:
        if not self.search_space:
            return copy.deepcopy(parent)
        cfg = copy.deepcopy(parent)
        key = random.choice(list(self.search_space.keys()))
        choices = self.search_space[key]
        if len(choices) <= 1:
            return cfg
        current = cfg.get(key)
        alternatives = [v for v in choices if v != current]
        cfg[key] = copy.deepcopy(random.choice(alternatives)) if alternatives else current
        return cfg

    def _cell(self, metrics: Dict[str, Any]) -> Tuple[int, ...]:
        coords = []
        for dim in self.feature_dimensions:
            value = float(metrics[dim])
            lo, hi = self.feature_ranges[dim]
            coords.append(self._bin(value, lo, hi, self.feature_bins))
        return tuple(coords)

    @staticmethod
    def _bin(value: float, lo: float, hi: float, bins: int) -> int:
        if hi <= lo:
            return 0
        t = max(0.0, min(1.0, (value - lo) / (hi - lo)))
        idx = int(t * bins)
        return min(idx, bins - 1)
