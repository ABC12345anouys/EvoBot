"""泛化抓取经验库 RAGMemory v3：统一到 MemoryStore 的经验记忆。

v2（scripts/evolution_loop.py 内嵌 JSON）→ v3 的升级点：
1. 统一存储：经验条目 = darwin/memory/experience/ 下的 YAML frontmatter .md，
   与 darwin/memory/store.py 的 MemoryStore / failure 记忆共用一套元数据协议
   （scope / kind / confidence / evidence / applies_when），不再是两套互不相通的记忆系统。
2. TTL 过期：失败记录带时间戳 ts（episode 计数），超过 FAILURE_TTL 自动失效并被
   decay() 清理，替代 v2 中"卡死时全删该 shape 失败记录"的粗暴做法。
3. 多维条件：匹配维度从 shape 扩展为 (shape, task, size_band)，
   size_band = 物体 half-size 最大维的分级（tiny/small/mid/large）。
4. 语义检索：best_success / retrieve 用 TF-IDF 余弦相似度（可选 sentence-transformers
   embedding，设 DARWIN_RAG_EMBED=1 启用，默认不下载任何模型）+ offset 数值接近度
   综合排序，替代纯线性扫描。

兼容 v2 的对外 API：add_failure / add_success / is_failed / best_success /
rank_candidates / success_rate / n_failure / n_success / OFFSET_TOL。
旧 data/rag_memory_v2.json 在首次初始化时自动迁移进 MemoryStore。
"""
from __future__ import annotations

import os
import re
import math
import numpy as np

from pathlib import Path
from typing import Any, Dict, List, Optional

from .store import MemoryStore

# 经验库默认根目录：darwin/memory/experience
DEFAULT_ROOT = Path(__file__).resolve().parent / "experience"
# 旧版 JSON 记忆（一次性迁移来源）
LEGACY_JSON = Path(__file__).resolve().parent.parent.parent / "data" / "rag_memory_v2.json"


# ============================================================
# 语义检索（轻量 TF-IDF，可选 embedding 后端）
# ============================================================

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", str(text).lower())


class _SemanticIndex:
    """TF-IDF 余弦相似度；DARWIN_RAG_EMBED=1 时尝试 sentence-transformers（失败自动回退）。"""

    def __init__(self) -> None:
        self._model = None
        self._embed_tried = False

    def _get_embedder(self):
        if not os.environ.get("DARWIN_RAG_EMBED", "") == "1":
            return None
        if not self._embed_tried:
            self._embed_tried = True
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._model = None
        return self._model

    def scores(self, query: str, docs: List[str]) -> np.ndarray:
        """返回 query 与每条 doc 的相似度（0~1）。"""
        if not docs:
            return np.zeros(0)
        embedder = self._get_embedder()
        if embedder is not None:
            try:
                q = embedder.encode([query], normalize_embeddings=True)[0]
                D = embedder.encode(docs, normalize_embeddings=True)
                return np.clip((D @ q + 1.0) / 2.0, 0.0, 1.0)
            except Exception:
                pass
        # TF-IDF 余弦
        docs_tok = [_tokenize(d) for d in docs]
        q_tok = _tokenize(query)
        vocab = sorted({t for toks in docs_tok for t in toks} | set(q_tok))
        vidx = {t: i for i, t in enumerate(vocab)}
        n_docs, V = len(docs_tok), len(vocab)
        df = np.zeros(V)
        for toks in docs_tok:
            for t in set(toks):
                df[vidx[t]] += 1
        idf = np.log((n_docs + 1) / (df + 1)) + 1.0

        def vec(toks):
            v = np.zeros(V)
            for t in toks:
                v[vidx[t]] += 1.0
            return v * idf

        qv = vec(q_tok)
        qn = np.linalg.norm(qv)
        D = np.stack([vec(toks) for toks in docs_tok])
        Dn = np.linalg.norm(D, axis=1) + 1e-9
        if qn < 1e-9:
            return np.zeros(n_docs)
        cos = (D @ qv) / (Dn * qn)
        return np.clip((cos + 1.0) / 2.0, 0.0, 1.0)


_SEM = _SemanticIndex()


def size_band_of(size: List[float]) -> str:
    """物体尺寸分级（half-size 最大维，米）：tiny<0.02≤small<0.04≤mid<0.09≤large。"""
    m = max(size) if size else 0.0
    if m < 0.02:
        return "tiny"
    if m < 0.04:
        return "small"
    if m < 0.09:
        return "mid"
    return "large"


# ============================================================
# RAGMemory v3
# ============================================================

class RAGMemory:
    """泛化抓取经验库（MemoryStore 后端）。

    条目结构（frontmatter 自定义字段）：
    - success 记录: kind=strategy, {task, shape, size, rel_offset, score_band, attempts, ts, exp}
    - failure 记录: kind=failure,  {task, shape, size, rel_offset, score_band, fail_phase, ts, exp}
    - stats 表:     kind=infra,   name=grasp_stats, {table: {f"{shape}|{task}|{band}": {succ, fail}}}
    """

    OFFSET_TOL = 0.5      # 归一化 offset 匹配半径（half-size 单位）
    FAILURE_TTL = 20      # 失败记录存活 episode 数（过期自动失效/清理）
    OFFSET_SANITY = 1.5   # |rel_offset| 超过此值视为边界异常，应过滤

    def __init__(self, root=None, fresh: bool = False, path: Optional[str] = None) -> None:
        if path is not None:  # 兼容 v2 签名：path 指定 legacy json 时仅用于迁移来源
            root = Path(path).parent.parent / "memory" / "experience"
        self.store = MemoryStore(root or DEFAULT_ROOT)
        self.ts = 0                      # episode 计数（tick() 递增）
        self._stats: Dict[str, Dict[str, int]] = {}
        self._cache: Dict[str, Dict[str, Any]] = {}  # name -> frontmatter meta
        if fresh:
            self._clear_store()
        self._load_all()
        self._migrate_legacy_if_empty()

    # ---- 存储层 ----

    def _clear_store(self) -> None:
        for f in self.store.root.glob("*.md"):
            f.unlink()
        self._cache.clear()
        self._stats.clear()

    def _load_all(self) -> None:
        self._cache = {it["name"]: it["metadata"] for it in self.store.list_all()}
        stats_meta = self._cache.get("grasp_stats")
        if stats_meta:
            self._stats = stats_meta.get("table") or {}

    def _migrate_legacy_if_empty(self) -> None:
        """默认经验库为空且存在 v2 JSON 时自动迁移（成功后把 json 改名为 .imported）。

        仅对默认根目录生效：临时/隔离记忆库（无记忆基线、单测）不做迁移。
        """
        if self._cache or not LEGACY_JSON.exists():
            return
        if self.store.root.resolve() != DEFAULT_ROOT.resolve():
            return
        try:
            import json
            data = json.loads(LEGACY_JSON.read_text())
            for rec in data.get("records", []):
                meta = self._rec_meta(rec)
                self.store.add(f"exp_migrated_{len(self._cache):05d}", meta,
                               f"{rec['type']} 经验（v2 JSON 迁移）")
                self._cache[f"exp_migrated_{len(self._cache) - 1:05d}"] = meta
            self._stats = data.get("stats", {}) or self._stats
            self._save_stats()
            LEGACY_JSON.rename(LEGACY_JSON.with_suffix(".json.imported"))
        except Exception:
            pass  # 迁移失败不影响使用

    def _rec_meta(self, rec: Dict[str, Any]) -> Dict[str, Any]:
        kind = "failure" if rec["type"] == "failure" else "strategy"
        band = rec.get("score_band", "mid")
        size = rec.get("size", [0.02, 0.02, 0.02])
        meta = {
            "scope": "global",
            "kind": kind,
            "confidence": "single-shot",
            "evidence": {"cells": [f"{rec.get('task', 'any')}_{rec.get('shape', '?')}_s{rec.get('ts', 0)}"],
                         "attempts": int(rec.get("attempts", 1))},
            "applies_when": f"{rec.get('shape')} 物体 / {rec.get('task')} 任务",
            "task": rec.get("task", "any"),
            "shape": rec.get("shape", "unknown"),
            "size": size,
            "rel_offset": rec.get("rel_offset", [0.0, 0.0, 0.0]),
            "score_band": band,
            "size_band": size_band_of(size),
            "fail_phase": rec.get("fail_phase", ""),
            "attempts": int(rec.get("attempts", 0)),
            "ts": int(rec.get("ts", self.ts)),
            "exp": rec.get("exp", ""),
        }
        # 自动序列化调用方传入的任意额外字段（control_mode/control_hints 等）
        reserved = set(meta.keys()) | {"type"}
        for k, v in rec.items():
            if k not in reserved and v is not None:
                meta[k] = v
        return meta

    def _write_record(self, rec: Dict[str, Any]) -> None:
        name = f"exp_{self.ts:05d}_{rec['type']}_{len(self._cache):05d}"
        self.store.add(name, self._rec_meta(rec),
                       f"{rec['type']}: {rec.get('shape')} @ {rec.get('task')}, offset={rec.get('rel_offset')}")
        self._cache[name] = self._rec_meta(rec)

    def _save_stats(self) -> None:
        """stats 表直接读写（绕过 MemoryStore.add 的 evidence 合并，避免旧值覆盖新值）。"""
        from .store import _render
        meta = {
            "scope": "global", "kind": "infra", "confidence": "single-shot",
            "evidence": {"cells": ["grasp_stats"], "attempts": 1},
            "applies_when": "统计成功率表 (shape|task|band)",
            "table": self._stats,
        }
        path = self.store.root / "grasp_stats.md"
        path.write_text(_render(meta, "(shape, task, score_band) → 成功/失败计数"))

    # ---- 沉淀 ----

    def tick(self) -> None:
        """每 episode 调一次，推进 TTL 时钟。"""
        self.ts += 1

    def _task_ok(self, rec_task: str, task: str) -> bool:
        return rec_task == "any" or rec_task == task

    def _bump(self, shape: str, task: str, band: str, ok: bool) -> None:
        key = f"{shape}|{task}|{band}"
        s = self._stats.setdefault(key, {"succ": 0, "fail": 0})
        s["succ" if ok else "fail"] += 1
        self._save_stats()

    def add_failure(self, task, feat, rel_offset, score_band, fail_phase, exp: str = "",
                    **extra) -> None:
        """记录失败经验。extra 任意字段自动序列化（control_mode/control_hints 等）。"""
        self._write_record({
            "type": "failure", "task": task,
            "shape": feat["shape"], "size": feat.get("size", [0.02, 0.02, 0.02]),
            "rel_offset": [round(float(x), 3) for x in rel_offset],
            "score_band": score_band, "fail_phase": fail_phase, "ts": self.ts, "exp": exp,
            **extra,
        })
        self._bump(feat["shape"], task, score_band, ok=False)

    def add_success(self, task, feat, rel_offset, score_band, attempts, exp: str = "",
                    **extra) -> None:
        """记录成功经验。extra 任意字段自动序列化。"""
        self._write_record({
            "type": "success", "task": task,
            "shape": feat["shape"], "size": feat.get("size", [0.02, 0.02, 0.02]),
            "rel_offset": [round(float(x), 3) for x in rel_offset],
            "score_band": score_band, "attempts": int(attempts), "ts": self.ts, "exp": exp,
            **extra,
        })
        self._bump(feat["shape"], task, score_band, ok=True)

    # ---- 检索 ----

    def _records(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        out = []
        for name, meta in self._cache.items():
            if name == "grasp_stats":
                continue
            if kind and meta.get("kind") != kind:
                continue
            out.append(meta)
        return out

    @property
    def records(self) -> List[Dict[str, Any]]:
        """兼容 v2：以 dict 列表形式暴露（kind 映射回 type 字段）。"""
        recs = []
        for meta in self._records():
            r = dict(meta)
            r["type"] = "failure" if meta.get("kind") == "failure" else "success"
            recs.append(r)
        return recs

    def is_failed(self, task, feat, rel_offset) -> Optional[Dict[str, Any]]:
        """该归一化偏移（同形状+同任务）近期是否失败过（TTL 内）。"""
        off = np.asarray(rel_offset, float)
        band = size_band_of(feat.get("size", [0.02, 0.02, 0.02]))
        for rec in self._records("failure"):
            if rec.get("shape") != feat["shape"] or not self._task_ok(rec.get("task", "any"), task):
                continue
            if rec.get("size_band") != band:
                continue  # 尺寸差异大时 offset 语义不同，不匹配
            if self.ts - int(rec.get("ts", 0)) > self.FAILURE_TTL:
                continue  # TTL 过期：可能因物体/位置变化已失效
            if np.linalg.norm(off - np.array(rec["rel_offset"], float)) < self.OFFSET_TOL:
                return rec
        return None

    def best_success(self, task, feat) -> Optional[Dict[str, Any]]:
        """最匹配的成功经验（语义相似度 + 中心度综合排序），重建为绝对抓取点用。"""
        succs = [r for r in self._records("strategy")
                 if r.get("shape") == feat["shape"] and self._task_ok(r.get("task", "any"), task)]
        if not succs:
            return None
        query = f"{task} {feat['shape']} grasp pick place"
        docs = [f"{r.get('task')} {r.get('shape')} {r.get('score_band')} {r.get('exp', '')} {r.get('applies_when', '')}"
                for r in succs]
        sem = _SEM.scores(query, docs)
        best, best_score = None, -1.0
        for r, s in zip(succs, sem):
            center_closeness = 1.0 - min(np.linalg.norm(np.array(r["rel_offset"], float)), 2.0) / 2.0
            score = 0.5 * float(s) + 0.5 * center_closeness
            if score > best_score:
                best, best_score = r, score
        return best

    def retrieve(self, task_desc: str, feat, k: int = 5,
                 control_mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """语义检索：给定自然语言任务描述 + 物体特征，返回最相关的 k 条经验。

        control_mode 不为 None 时，同方案经验加权 +0.2（通用经验保留）。
        """
        recs = self._records()
        if not recs:
            return []
        docs = [f"{r.get('task')} {r.get('shape')} {r.get('score_band')} {r.get('exp', '')} {r.get('applies_when', '')} {r.get('control_mode', '')}"
                for r in recs]
        sem = _SEM.scores(task_desc, docs)
        band = size_band_of(feat.get("size", [0.02, 0.02, 0.02]))
        ranked = sorted(zip(recs, sem), key=lambda x: -x[1])
        def boost(r):
            b = 0.1 if r.get("size_band") == band else 0.0
            if control_mode and r.get("control_mode") == control_mode:
                b += 0.2
            return b
        ranked.sort(key=lambda x: -(x[1] + boost(x[0])))
        return [r for r, _ in ranked[:k]]

    def success_rate(self, shape, task, band):
        s = self._stats.get(f"{shape}|{task}|{band}") or self._stats.get(f"{shape}|any|{band}")
        if not s:
            return 0.5, 0  # 无数据：中性先验
        n = s["succ"] + s["fail"]
        return (s["succ"] + 0.5) / (n + 1), n  # 拉普拉斯平滑

    def rank_candidates(self, task, feat, candidates):
        """按泛化统计成功率重排候选（UCB 式探索加成：没试过的先试）。"""
        scored = []
        for c in candidates:
            rate, n = self.success_rate(feat["shape"], task, c["score_band"])
            prio = rate + 0.15 / math.sqrt(1 + n)
            scored.append((prio, c))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored]

    # ---- 过期清理 ----

    def decay(self) -> int:
        """删除 TTL 过期的失败记录文件，返回删除条数。替代 v2 的全删策略。"""
        removed = 0
        for name, meta in list(self._cache.items()):
            if meta.get("kind") != "failure":
                continue
            if self.ts - int(meta.get("ts", 0)) > self.FAILURE_TTL:
                (self.store.root / f"{name}.md").unlink(missing_ok=True)
                del self._cache[name]
                removed += 1
        return removed

    # ---- 汇总 ----

    @property
    def stats(self) -> Dict[str, Dict[str, int]]:
        """统计成功率表（兼容 v2 访问方式）。"""
        return self._stats

    @property
    def n_failure(self):
        return sum(1 for m in self._cache.values()
                   if m.get("kind") == "failure" and self.ts - int(m.get("ts", 0)) <= self.FAILURE_TTL)

    @property
    def n_success(self):
        return sum(1 for m in self._cache.values() if m.get("kind") == "strategy")

    def save(self) -> None:  # 兼容 v2：写入即时落盘，此函数保留为空操作
        pass

    def load(self) -> None:  # 兼容 v2
        self._load_all()
