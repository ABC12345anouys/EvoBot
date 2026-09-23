"""ExperienceStore：episode 级经验库（参考 EvoAgentX AFlow 的 experience.json）。

解决"只跑了一两个任务，技能不完全"时怎么总结经验：
- 每次执行一个目标条件用了某个 method，记录 {method, cond, params_hint,
  before_score, after_score, succeed, fail_phase, log_path}；
- 按 method 聚合 success/failure，format 成给 SkillCreator 的提示
  （"绝对禁止重复这些失败的方法/参数组合"）；
- check_repeated 防止 SkillCreator 生成与已知失败完全相同的 method；
- 经验持久化为 logs/experience/<task>.jsonl，可跨进程检索。

与 RAGMemory 的分工：
- RAGMemory 记"抓取点 rel_offset"级别的微观经验（哪类物体哪个位置抓得起来）；
- ExperienceStore 记"method/skill 链"级别的宏观经验（哪种方法组合对哪类条件有效/无效）。
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .objectives import GoalCond


class ExperienceStore:
    """按 method 聚合的成功/失败经验库。"""

    def __init__(self, root: Optional[str] = None, *, load_history: bool = True) -> None:
        root = Path(root or Path("logs") / "experience")
        root.mkdir(parents=True, exist_ok=True)
        self.root = root
        self._records: Dict[str, Dict[str, Any]] = {}  # task -> {method: {success:[], failure:[]}}
        if load_history:
            self._load_history()

    def _load_history(self) -> None:
        """启动时加载 root 下所有 *.jsonl，实现跨进程经验累积。"""
        for jf in sorted(self.root.glob("*.jsonl")):
            try:
                for line in jf.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    task = r.get("task", "unknown")
                    mname = r.get("method", "unknown")
                    bucket = "success" if r.get("success") else "failure"
                    self._records.setdefault(task, {}) \
                        .setdefault(mname, {"success": [], "failure": []})[bucket].append(r)
            except Exception as e:
                print(f"[experience_store] 加载 {jf.name} 失败: {e}")

    # ---------- 写 ----------

    def add(self, task: str, cond: GoalCond, method_name: str, *,
            success: bool, before_score: float = 0.0, after_score: float = 0.0,
            fail_phase: str = "", params_hint: Optional[Dict] = None,
            log_path: str = "") -> Dict[str, Any]:
        """记录一次方法对某个条件的执行结果。

        before_score：执行前该任务/条件的历史最优分（没有则 0）；
        after_score：本次实际分（成功>0，失败=0 或负）。
        """
        rec = {
            "task": task,
            "cond": cond.describe(),
            "cond_kind": cond.kind,
            "method": method_name,
            "success": bool(success),
            "before": round(float(before_score), 4),
            "after": round(float(after_score), 4),
            "fail_phase": fail_phase,
            "params": params_hint or {},
            "log": log_path,
            "ts": time.time(),
        }
        self._records.setdefault(task, {}) \
            .setdefault(method_name, {"success": [], "failure": []})
        bucket = "success" if success else "failure"
        self._records[task][method_name][bucket].append(rec)
        # 持久化
        path = self.root / f"{task}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    # ---------- 查 ----------

    def best_score(self, task: str, method_name: str) -> float:
        """该方法在该任务上的历史最优 after_score。"""
        bucket = self._records.get(task, {}).get(method_name, {})
        succ = bucket.get("success", [])
        return max((r["after"] for r in succ), default=0.0)

    def failures(self, task: str, method_name: str) -> List[Dict[str, Any]]:
        return self._records.get(task, {}).get(method_name, {}).get("failure", [])

    def successes(self, task: str, method_name: str) -> List[Dict[str, Any]]:
        return self._records.get(task, {}).get(method_name, {}).get("success", [])

    def known_methods(self, task: str) -> List[str]:
        return list(self._records.get(task, {}).keys())

    # ---------- 给 LLM 的经验提示 ----------

    def format_for_creator(self, task: str, cond: GoalCond,
                           max_fail: int = 5) -> str:
        """生成给 SkillCreator 的经验文本：列出该条件上已失败的方法及其原因。

        参考 AFlow ExperienceUtils.format_experience：明确告知"禁止重复这些失败"。
        额外加入 fail_cat 分布统计，引导 LLM 针对主要失败模式改进。
        """
        lines = [f"# 经验：针对条件 `{cond.describe()}` 在任务 `{task}` 上的历史记录"]
        total_fail = 0
        cat_count: Dict[str, int] = {}
        for mname, bucket in self._records.get(task, {}).items():
            fails = bucket.get("failure", [])[-max_fail:]
            succs = bucket.get("success", [])
            if fails:
                total_fail += len(fails)
                lines.append(f"\n## 方法 {mname}（{len(succs)} 成功 / {len(bucket['failure'])} 失败）")
                for r in fails:
                    cat = r.get("fail_phase") or "?"
                    cat_count[cat] = cat_count.get(cat, 0) + 1
                    lines.append(f"- 失败 @ {cat}  "
                                 f"after={r['after']}  params={r['params']}")
                    if r.get("log"):
                        lines.append(f"  log: {r['log']}")
        if total_fail == 0:
            lines.append("（暂无失败记录）")
        else:
            # 失败模式分布，引导改进方向
            sorted_cats = sorted(cat_count.items(), key=lambda x: -x[1])
            top_cat = sorted_cats[0][0]
            lines.append(f"\n# 失败模式分布: {dict(sorted_cats)}")
            hints = {
                "grip_failed": "主要失败是夹爪没抓住 → 增大 descend 深度(k_descend)、"
                               "减小 hover、检查 close_gripper 闭合参数",
                "timeout": "超时 → 增大超时阈值或放宽 stop 条件",
                "collision": "碰撞 → 抬高 hover、增大安全距离",
                "force_exceed": "力超限 → 减小 stiffness、增大 damping",
            }
            if top_cat in hints:
                lines.append(f"# 改进建议（针对 {top_cat}）: {hints[top_cat]}")
        lines.append("\n注意：不要重复已失败的方法/参数组合；需要换思路而非微调。")
        return "\n".join(lines)

    def is_repeated_failure(self, task: str, method_name: str,
                            params_hint: Optional[Dict] = None) -> bool:
        """该方法+参数是否与已知失败完全重复（防止 SkillCreator 重踩坑）。"""
        params_hint = params_hint or {}
        for r in self.failures(task, method_name):
            if r.get("params") == params_hint:
                return True
        return False

    # ---------- 诊断 ----------

    def summary(self, task: Optional[str] = None) -> Dict[str, Any]:
        tasks = [task] if task else list(self._records)
        out = {}
        for t in tasks:
            out[t] = {}
            for mname, bucket in self._records.get(t, {}).items():
                fails = bucket.get("failure", [])
                out[t][mname] = {
                    "success": len(bucket.get("success", [])),
                    "failure": len(fails),
                    "best": self.best_score(t, mname),
                    # 最近一次失败分类：goal_not_reached 说明链基本走通只差
                    # 终态，比 skill 级失败（如 grip_failed）更接近成功
                    "last_fail": fails[-1]["fail_phase"] if fails else "",
                }
        return out


__all__ = ["ExperienceStore"]
