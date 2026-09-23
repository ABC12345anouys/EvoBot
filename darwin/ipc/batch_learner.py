"""batch_learner：LIBERO 全任务批量反思学习（任务队列 + 三重预算 + 台账）。

设计（见 .trae/documents/physics-taskspec-batch-learn_plan.md 步骤7）：
- 每个任务独立 sim 子进程跑 scripts/ipc_learn.py（任务间状态完全隔离）；
- 三重预算：墙钟 --max-hours、每任务 attempt 上限、LLM token 累计上限
  （解析子进程日志里的 LLM_TOKENS= 机器行）；
- 台账 logs/libero_progress.json：pending/learned/failed/unsupported/
  unavailable 状态机，原子写，断点续跑默认跳过 learned；
- 每 N 个任务跑 libero_spatial:0 哨兵回归；失败回滚到上一个哨兵通过时的
  YAML/forged 快照并记 regression 事件。

本模块不 import mujoco：仿真只活在子进程里。
"""
from __future__ import annotations

import argparse
import filecmp
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
# libero_100 的 task map 损坏，队列不含（需要时 --suites 显式给也会在
# preflight 被标 unavailable）
DEFAULT_SUITES = ["libero_spatial", "libero_object", "libero_goal",
                  "libero_10", "libero_90"]
SUITE_SIZES = {"libero_spatial": 10, "libero_object": 10, "libero_goal": 10,
               "libero_10": 10, "libero_90": 90, "libero_100": 10}

# 哨兵回归任务（最初打通的空间放置，attempt1 应稳定成功）
SENTINEL_TASK = "libero_spatial:0"

# ---- 回滚快照监视的文件（相对 REPO_ROOT）----
_WATCHED_DIRS = [
    Path("darwin/skills/configs"),
    Path("darwin/agents/methods"),
    Path("darwin/skills/forged"),
]
_EXCLUDE_PARTS = {"__pycache__", "examples"}


def _watched_files() -> List[Path]:
    out: List[Path] = []
    for d in _WATCHED_DIRS:
        root = REPO_ROOT / d
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            if any(part in _EXCLUDE_PARTS for part in p.relative_to(root).parts):
                continue
            if p.suffix in {".pyc"}:
                continue
            if p.name == "__init__.py":
                continue
            out.append(p.relative_to(REPO_ROOT))
    return sorted(out)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def make_snapshot(dst: Path) -> List[str]:
    """把监视文件复制到 dst，返回相对路径清单。"""
    if dst.exists():
        shutil.rmtree(dst)
    rels = [str(p) for p in _watched_files()]
    for rel in rels:
        src = REPO_ROOT / rel
        tgt = dst / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tgt)
    return rels


def restore_snapshot(src: Path) -> None:
    """回滚：快照内文件覆盖回去；快照后新增的监视 yaml/forged 文件删除。"""
    old_rels = {Path(p) for p in src.rglob("*") if p.is_file()}
    old_rel_str = {str(p.relative_to(src)) for p in old_rels}
    # 删除快照后新增的受监视文件
    for rel in _watched_files():
        if str(rel) not in old_rel_str and rel.suffix in {".yaml", ".md"}:
            try:
                (REPO_ROOT / rel).unlink()
            except OSError:
                pass
    # 覆盖恢复
    for rel in old_rels:
        tgt = REPO_ROOT / rel.relative_to(src)
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rel, tgt)


class BatchLearner:
    def __init__(self, ledger: Path, max_hours: float,
                 per_task_attempts: int = 8,
                 llm: bool = False, llm_token_budget: int = 0,
                 sentinel_every: int = 10, force: bool = False) -> None:
        self.ledger_path = ledger
        self.max_hours = float(max_hours)
        self.per_task_attempts = int(per_task_attempts)
        self.llm = bool(llm)
        self.llm_token_budget = int(llm_token_budget)
        self.sentinel_every = int(sentinel_every)
        self.force = bool(force)
        self.log_dir = REPO_ROOT / "logs" / "batch_logs"
        self.snap_root = REPO_ROOT / "logs" / "snapshots"
        self.deadline = time.time() + self.max_hours * 3600.0
        self.ledger = self._load_ledger()

    # ---- 台账 ----
    def _load_ledger(self) -> Dict[str, Any]:
        if self.ledger_path.exists():
            try:
                return json.loads(self.ledger_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"version": 1, "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "updated_at": "", "tokens_total": 0,
                "tasks": {}, "events": []}

    def _save(self) -> None:
        self.ledger["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _atomic_write_json(self.ledger_path, self.ledger)

    def _event(self, etype: str, **kw) -> None:
        self.ledger["events"].append(
            {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "type": etype, **kw})

    def _set_task(self, task: str, status: str, **kw) -> None:
        rec = self.ledger["tasks"].setdefault(task, {})
        rec["status"] = status
        rec["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec.update(kw)
        self._save()

    # ---- 队列 ----
    def build_queue(self, suites: List[str]) -> List[str]:
        sys.path.insert(0, str(REPO_ROOT))
        from darwin.benchmarks import get_libero_benchmark
        queue: List[str] = []
        for suite in suites:
            for idx in range(SUITE_SIZES.get(suite, 10)):
                task = f"{suite}:{idx}"
                rec = self.ledger["tasks"].get(task)
                if rec and rec.get("status") == "learned" and not self.force:
                    continue
                # preflight：entry/spec 构造失败 → unavailable/unsupported
                try:
                    entry = get_libero_benchmark(suite, idx)
                except Exception as e:  # noqa: BLE001
                    self._set_task(task, "unavailable",
                                   note=f"preflight: {type(e).__name__}: {e}")
                    continue
                spec = entry.get("libero_spec") or {}
                if spec.get("unsupported"):
                    self._set_task(task, "unsupported",
                                   note=f"未知谓词 {spec.get('unsupported')}")
                    continue
                queue.append(task)
        return queue

    # ---- 子进程 ----
    def _run_subprocess(self, task: str, max_attempts: int,
                        tag: str) -> subprocess.CompletedProcess:
        sock = f"/tmp/darwin_batch_{tag}_{os.getpid()}.sock"
        for p in (sock,):
            try:
                os.unlink(p)
            except OSError:
                pass
        log_path = self.log_dir / f"{task.replace(':', '_')}_{tag}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(REPO_ROOT / "scripts" / "ipc_learn.py"),
               task, "--sock", sock, "--max-attempts", str(max_attempts)]
        if self.llm and tag == "learn":
            cmd.append("--llm")
        env = dict(os.environ)
        env.setdefault("MUJOCO_GL", "egl")
        env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        if env.get("MUJOCO_GL") == "egl":
            env.pop("DISPLAY", None)
        with open(log_path, "w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env,
                                  stdout=lf, stderr=subprocess.STDOUT)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        return proc, log_path, text

    @staticmethod
    def _parse_tokens(text: str) -> int:
        n = 0
        for line in text.splitlines():
            if "LLM_TOKENS=" in line:
                try:
                    n = max(n, int(line.rsplit("LLM_TOKENS=", 1)[1].strip()))
                except ValueError:
                    pass
        return n

    # ---- 哨兵回归 ----
    def _sentinel(self) -> bool:
        print(f"[batch] 哨兵回归 {SENTINEL_TASK}（最多 2 attempt）…", flush=True)
        proc, log_path, text = self._run_subprocess(
            SENTINEL_TASK, 2, tag=f"sentinel{int(time.time())}")
        ok = proc.returncode == 0
        self._event("sentinel", task=SENTINEL_TASK, ok=ok,
                    rc=proc.returncode, log=str(log_path))
        print(f"[batch] 哨兵 {'通过' if ok else '失败'}（rc={proc.returncode}，"
              f"日志 {log_path}）", flush=True)
        return ok

    def _refresh_good_snapshot(self, label: str) -> None:
        dst = self.snap_root / "good"
        make_snapshot(dst)
        (self.snap_root / "good_label.txt").write_text(
            f"{label} {time.strftime('%Y-%m-%dT%H:%M:%S')}\n",
            encoding="utf-8")

    # ---- 主循环 ----
    def run(self, suites: List[str]) -> int:
        return self.run_queue(self.build_queue(suites))

    def run_queue(self, queue: List[str]) -> int:
        print(f"[batch] 待学队列 {len(queue)} 个任务: "
              f"{queue[:5]}{' …' if len(queue) > 5 else ''}", flush=True)
        print(f"[batch] 预算: {self.max_hours:g}h / 每任务 "
              f"{self.per_task_attempts} attempts / "
              f"LLM token={'无限' if self.llm_token_budget <= 0 else self.llm_token_budget}"
              f"（llm={'on' if self.llm else 'off'}）", flush=True)
        if not queue:
            print("[batch] 没有待学任务（台账中均已 learned），直接结束。",
                  flush=True)
            return 0
        self._event("batch_start", queue_len=len(queue),
                    max_hours=self.max_hours,
                    per_task_attempts=self.per_task_attempts, llm=self.llm)
        self._save()

        # 基线：先存当前快照，再跑一次哨兵；基线不过则不开批（避免污染判断）
        self._refresh_good_snapshot("baseline")
        if not self._sentinel():
            print("[batch] 基线哨兵未通过，终止批学（请先恢复 spatial:0）。",
                  flush=True)
            return 3

        processed = 0
        for task in queue:
            if time.time() >= self.deadline:
                print(f"[batch] 墙钟预算用尽，剩余任务保持 pending，退出。",
                      flush=True)
                self._event("budget_stop", reason="wallclock")
                break
            if (self.llm_token_budget > 0
                    and int(self.ledger["tokens_total"]) >= self.llm_token_budget):
                print("[batch] LLM token 预算用尽，退出。", flush=True)
                self._event("budget_stop", reason="llm_tokens")
                break

            t0 = time.time()
            print(f"[batch] === {task} 开始 "
                  f"({time.strftime('%H:%M:%S')}) ===", flush=True)
            proc, log_path, text = self._run_subprocess(
                task, self.per_task_attempts, tag="learn")
            dt = time.time() - t0
            tokens = self._parse_tokens(text)
            self.ledger["tokens_total"] = int(
                self.ledger.get("tokens_total", 0)) + tokens
            status = "learned" if proc.returncode == 0 else "failed"
            self._set_task(
                task, status, rc=proc.returncode, seconds=round(dt, 1),
                tokens=tokens, log=str(log_path),
                note="" if proc.returncode in (0, 2)
                else f"异常退出 rc={proc.returncode}")
            print(f"[batch] === {task} {status}（{dt:.0f}s，"
                  f"tokens={tokens}，rc={proc.returncode}）===", flush=True)
            processed += 1

            if processed % self.sentinel_every == 0:
                if self._sentinel():
                    self._refresh_good_snapshot(f"after {processed} tasks")
                else:
                    # 回滚到上个通过点，记 regression，继续后续任务
                    good = self.snap_root / "good"
                    if good.exists():
                        restore_snapshot(good)
                    self._event("regression_rollback", after_task=task)
                    print("[batch] 已回滚到上一哨兵通过点的 YAML/forged 快照。",
                          flush=True)

        self._event("batch_end", processed=processed)
        self._save()
        learned = sum(1 for r in self.ledger["tasks"].values()
                      if r.get("status") == "learned")
        print(f"[batch] 结束。本轮处理 {processed}，累计 learned={learned}，"
              f"台账 {self.ledger_path}", flush=True)
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="LIBERO 全任务批量反思学习")
    ap.add_argument("--suites", default=",".join(DEFAULT_SUITES),
                    help="逗号分隔，默认 spatial,object,goal,libero_10,libero_90")
    ap.add_argument("--limit", default="",
                    help="逗号分隔的精确任务（如 libero_spatial:0,libero_object:0），"
                         "覆盖队列顺序用于冒烟")
    ap.add_argument("--max-hours", type=float, default=1.0)
    ap.add_argument("--per-task-attempts", type=int, default=8)
    ap.add_argument("--llm", action="store_true",
                    help="启用 LLM 语义反思（默认关，纯规则反思）")
    ap.add_argument("--llm-token-budget", type=int, default=0,
                    help="LLM token 累计上限，0=不限")
    ap.add_argument("--sentinel-every", type=int, default=10)
    ap.add_argument("--ledger",
                    default=str(REPO_ROOT / "logs" / "libero_progress.json"))
    ap.add_argument("--force", action="store_true",
                    help="已 learned 的任务也重学")
    args = ap.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    bl = BatchLearner(Path(args.ledger), max_hours=args.max_hours,
                      per_task_attempts=args.per_task_attempts,
                      llm=args.llm, llm_token_budget=args.llm_token_budget,
                      sentinel_every=args.sentinel_every, force=args.force)
    if args.limit:
        # 冒烟模式：先跑全量 preflight（顺带标记 unavailable/unsupported），
        # 再只学给定任务并保持给定顺序
        wanted = [t.strip() for t in args.limit.split(",") if t.strip()]
        full = set(bl.build_queue(suites))
        missing = [t for t in wanted if t not in full]
        if missing:
            print(f"[batch] --limit 任务不可用（已跳过）: {missing}", flush=True)
        return bl.run_queue([t for t in wanted if t in full])
    return bl.run(suites)


if __name__ == "__main__":
    raise SystemExit(main())
