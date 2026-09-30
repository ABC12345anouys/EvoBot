"""LIBERO 串行 agent runner：符号规划 + 几何求解 + 闭环重试。

架构（替代旧 per-task YAML 阈值 + 固定技能链）：
    BDDL goal ──task_spec──> 有序子目标（拓扑排序，确定性）
    子目标 ──libero_planner──> 技能步骤（谓词级映射）
    技能 ──libero_skills──> adapter GT 几何求解 + OSC 伺服
    失败 ──mechanism──> 重规划（换抓取候选/重试），不碰任务参数

执行语义（用户要求）：
- 任务串行：一个任务通过（BDDL check_success=True）才启动下一个；
- 每个任务只跑一条轨迹，失败就重试，直到成功或 --max-attempts；
- 台账 logs/libero_agent_progress.json：重跑自动跳过已通过任务。

用法：
    python -m darwin.agents.libero_runner --tasks libero_spatial:0
    python -m darwin.agents.libero_runner \
        --suites libero_spatial,libero_object,libero_goal
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from darwin.agents import libero_planner as PL
from darwin.agents.task_spec import load_or_parse
from darwin.envs.libero_adapter import (LiberoEnvAdapter, AttemptStepLimit,
                                        _ensure_libero_path,
                                        _task_info, parse_env_id)

_ensure_libero_path()  # _task_info 直接 import libero，须先于其调用

LEDGER = _REPO / "logs" / "libero_agent_progress.json"
VIDEO_DIR = _REPO / "videos" / "libero_agent"
SUITE_SIZES = {"libero_spatial": 10, "libero_object": 10,
               "libero_goal": 10, "libero_10": 10, "libero_90": 90}


class _AttemptTimeout(Exception):
    """单条轨迹硬超时（用户规则：一个任务 >5min 即默认失败）。"""


def _on_alarm(signum, frame):
    raise _AttemptTimeout()


# ---------------- 台账 ----------------

def _load_ledger() -> Dict[str, Any]:
    if LEDGER.exists():
        try:
            return json.loads(LEDGER.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"tasks": {}}


def _save_ledger(ledger: Dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    tmp.replace(LEDGER)


# ---------------- 视频 ----------------

class _Recorder:
    """挂在 adapter.step 上抓 agentview 帧（libero 图像上下颠倒，翻转）。"""

    def __init__(self, adapter):
        self.frames: List = []
        self._orig = adapter.step

        def rec(action):
            obs, r, d, i = self._orig(action)
            if isinstance(obs, dict):
                img = obs.get("agentview_image")
                if img is not None:
                    self.frames.append(img[::-1].copy())
            return obs, r, d, i
        adapter.step = rec

    def save(self, path: Path, fps: int = 20) -> bool:
        if not self.frames:
            return False
        try:
            import imageio.v2 as imageio
            path.parent.mkdir(parents=True, exist_ok=True)
            imageio.mimsave(str(path), self.frames, fps=fps)
            return True
        except Exception as e:
            print(f"[runner] 视频保存失败: {e}", flush=True)
            return False


# ---------------- episode ----------------

def run_episode(adapter, spec: Dict[str, Any],
                max_rounds: int = 4,
                mem: Optional[Dict[str, dict]] = None) -> Dict[str, Any]:
    """单条轨迹：按序推进子目标，失败按机制重规划（同 episode 内）。

    每轮从世界现状重新评估哪些子目标已满足（已满足的跳过），
    对未满足的执行计划；子目标内失败计数驱动候选轮换。

    mem 可跨 attempt 传入：__failed__ 候选键（世界系 site+R，任务
    reset 位姿确定）是机制级失败记忆——attempt 超时被杀的 grasp 已
    试过的坏候选不应在下一 attempt 原样重试（object:6 实证：同序列
    每 attempt 从头重复，300s 只够试 2-3 个候选，永远到不了可行方位）。
    """
    subs = [s for s in spec.get("subgoals", []) if s.get("kind") != "unsupported"]
    skipped = [s for s in spec.get("subgoals", []) if s.get("kind") == "unsupported"]
    for s in skipped:
        print(f"[runner] 跳过 unsupported 谓词: {s.get('predicate')}", flush=True)
    if mem is None:
        mem = {}
    history: List[Dict[str, Any]] = []
    last_mech: Dict[tuple, Optional[str]] = {}  # 子目标 → 上次失败机制

    for round_ in range(max_rounds + 1):
        remaining = []
        for s in subs:
            try:
                if not PL.subgoal_satisfied(adapter, s):
                    remaining.append(s)
            except Exception as e:
                print(f"[runner] 子目标判定异常（按未满足）: {e}", flush=True)
                remaining.append(s)
        if not remaining:
            return {"success": True, "history": history,
                    "note": f"全部子目标满足（round {round_}）"}
        if round_ == max_rounds:
            # 最后一轮只做验证不再执行：否则最后一个执行轮达成的子目标
            # 永远等不到下一轮复核，明明已全部满足仍报"轮次耗尽"
            # （goal:4 实证：round 3 达成 On 后直接落入耗尽分支）。
            break

        s = remaining[0]
        steps = PL.plan_subgoal(s)
        used_alt = False
        if s.get("kind") == "place" and any(
                h.get("subgoal") is s and h.get("step") == "grasp"
                and not h.get("ok")
                and h.get("mechanism") == "no_candidate" for h in history):
            # 抓取候选空间已实测耗尽（no_candidate 是几何事实）→ On/In
            # 换用替代物理实现（推/拨）。析取规划，与任务无关。
            alt = PL.plan_subgoal_alternative(s)
            if alt:
                steps = alt
                used_alt = True
        if not steps:
            history.append({"subgoal": s, "ok": False, "reason": "no_plan"})
            return {"success": False, "history": history,
                    "fail": f"子目标无计划: {s}"}
        label = (f"{s.get('predicate')}({s.get('object') or ''}"
                 f"{',' + s['target'] if s.get('target') else ''})")
        skey = (s.get("predicate"), s.get("object"), s.get("target"))
        failures = sum(1 for h in history
                       if h.get("subgoal") is s and not h.get("ok"))
        hint = PL.replan_hint(failures, last_mech.get(skey))
        ok_sub = True
        for st in steps:
            res = PL.execute_step(adapter, st, mem=mem, **hint)
            history.append({"subgoal": s, "step": st["skill"],
                            "ok": bool(res.get("success")),
                            "reason": res.get("reason"),
                            "mechanism": res.get("mechanism"),
                            "measures": res.get("measures")})
            tag = "✓" if res.get("success") else "✗"
            print(f"[runner]   {st['skill']} {tag} "
                  f"{res.get('reason') or ''} {res.get('measures') or ''}",
                  flush=True)
            if not res.get("success"):
                last_mech[skey] = res.get("mechanism")
                hint = PL.replan_hint(failures + 1, res.get("mechanism"))
                ok_sub = False
                break
        if ok_sub and not PL.subgoal_satisfied(adapter, s):
            history.append({"subgoal": s, "ok": False,
                            "reason": "goal_not_reached",
                            "mechanism": "goal_not_reached"})
            print(f"[runner]   {label} 执行完成但谓词未满足", flush=True)
        elif ok_sub:
            print(f"[runner] 子目标达成: {label}", flush=True)
        if used_alt and not ok_sub:
            # 替代物理实现（推/拨）失败后不再消耗剩余轮次：替代分支只有一条
            # push_to，而 push_to 不接受 replan_hint 的 cand/strategy，后面每轮
            # 在参数上是逐字重复（实测 goal:5 与 object:1/6/9 都是同一原因连挂
            # 4 轮）。直接收尾并上报真实机制，省掉 3/4 的执行时间，也避免把
            # no_candidate（抓取几何耗尽）伪装成"轮次耗尽"。
            return {"success": False, "history": history,
                    "fail": f"替代方案失败: {last_mech.get(skey) or '未知机制'}"}
    return {"success": False, "history": history,
            "fail": f"重规划轮次耗尽（{max_rounds}）"
                    f": 机制 {sorted({m for m in last_mech.values() if m})}"}


def run_task(env_id: str, max_attempts: int = 8,
             record: bool = True,
             attempt_timeout: int = 300,
             attempt_steps: int = 7000) -> Dict[str, Any]:
    """单任务重试闭环：每 attempt 全 reset 跑一条轨迹，直到 BDDL 成功。

    attempt_steps：单条轨迹的**仿真步数**上限（确定性截断，主判据）。
    attempt_timeout：挂钟兜底（秒），只在进程异常慢/卡住时才可能先触发，
    默认 300=5min。两者都触发时，先到的生效。
    """
    suite, idx = parse_env_id(env_id)
    info = _task_info(suite, idx)
    spec = load_or_parse(suite, idx, info["bddl"],
                         language=getattr(info["task"], "language", ""))
    print(f"[runner] === {env_id}: {getattr(info['task'], 'language', '')}",
          flush=True)
    print(f"[runner] 子目标: "
          f"{[(s.get('predicate'), s.get('object'), s.get('target')) for s in spec.get('subgoals', [])]}",
          flush=True)

    adapter = LiberoEnvAdapter(suite, idx)
    attempts: List[Dict[str, Any]] = []
    mem: Dict[str, dict] = {}          # 跨 attempt 的机制级记忆（失败候选键）
    try:
        for attempt in range(1, max_attempts + 1):
            t0 = time.time()
            # 确定性：perception 里的点云下采样用的是全局 np.random（见
            # skills/perception/grasp.py 的 np.random.choice），而全局 RNG
            # 从未播种 —— 每次跑到这里的子采样都不同，喂给 GraspNet 的点
            # 随之不同，候选抓取时有时无（实测同一任务同一初始状态，
            # 一次 GraspNet 给出 0 个候选、另一次给出多个，整条路径分叉）。
            # 按 attempt 序号播种：同一次 attempt 可复现，不同 attempt 仍
            # 有不同子采样（保留重试多样性）。
            np.random.seed(attempt)
            adapter.reset()
            adapter.step_budget = int(attempt_steps) or None
            rec = _Recorder(adapter) if record else None
            signal.signal(signal.SIGALRM, _on_alarm)
            signal.alarm(attempt_timeout)
            try:
                result = run_episode(adapter, spec, mem=mem)
            except AttemptStepLimit as e:
                # 确定性截断：只看步数，与机器负载无关，保证同初始状态同结果
                from darwin.agents import libero_skills as _S
                _S.snap_dump_last(f"step_budget>{e.budget}steps")
                result = {"success": False,
                          "fail": f"step_budget>{e.budget}steps",
                          "history": []}
            except _AttemptTimeout:
                from darwin.agents import libero_skills as _S
                _S.snap_dump_last(f"attempt_timeout>{attempt_timeout}s")
                result = {"success": False,
                          "fail": f"attempt_timeout>{attempt_timeout}s",
                          "history": []}
            except Exception as e:  # 物理层异常（episode_terminated 等）
                from darwin.agents import libero_skills as _S
                _S.snap_dump_last(f"exception: {e}")
                result = {"success": False, "fail": f"exception: {e}",
                          "history": []}
            finally:
                signal.alarm(0)
            ok = bool(result.get("success")) and adapter.check_success()
            if result.get("success") and not ok:
                result["note"] = (result.get("note", "")
                                  + "；子目标满足但 check_success=False")
            rec_out = {"attempt": attempt, "success": ok,
                       "fail": result.get("fail"),
                       "steps": int(adapter.steps),
                       "secs": round(time.time() - t0, 1)}
            # 保留每步执行痕迹：原来 history（skill/reason/mechanism/
            # measures）跑完即弃，失败后无法定位是哪一步、哪种机制，
            # 只能再整任务重跑。这里压缩落台账：机制计数 + 末尾若干步。
            hist = result.get("history") or []
            if hist:
                mech_count: Dict[str, int] = {}
                for h in hist:
                    if not h.get("ok"):
                        m = h.get("mechanism") or h.get("reason") or "?"
                        mech_count[m] = mech_count.get(m, 0) + 1
                trace = [{"skill": h.get("step"), "ok": bool(h.get("ok")),
                          "reason": h.get("reason"),
                          "mech": h.get("mechanism")}
                         for h in hist[-12:]]
                rec_out["fail_mech"] = mech_count
                rec_out["trace"] = trace
            attempts.append(rec_out)
            print(f"[runner] attempt {attempt}/{max_attempts} "
                  f"{'SUCCESS' if ok else 'FAIL'} "
                  f"({rec_out['secs']}s, {rec_out['steps']}步) "
                  f"{result.get('fail') or ''}",
                  flush=True)
            if rec is not None:
                tag = "ok" if ok else "fail"
                rec.save(VIDEO_DIR / f"{suite}_{idx:02d}_a{attempt}_{tag}.mp4")
            if ok:
                return {"env_id": env_id, "success": True,
                        "attempts": attempts}
    finally:
        adapter.close()
    return {"env_id": env_id, "success": False, "attempts": attempts}


# ---------------- 队列 ----------------

def build_queue(suites: List[str]) -> List[str]:
    out = []
    for s in suites:
        for i in range(SUITE_SIZES[s]):
            out.append(f"libero:{s}:{i}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="LIBERO 串行 agent runner")
    ap.add_argument("--tasks", default="",
                    help="逗号分隔，如 libero_spatial:0,libero_object:2")
    ap.add_argument("--suites", default="",
                    help="逗号分隔 suite 名（与 --tasks 二选一）")
    ap.add_argument("--max-attempts", type=int, default=8)
    # 主判据：仿真步数（确定性）。标定依据：spatial:4 一次完整失败尝试
    # = 1605 步 / 88.2s（伺服段约 32 步/s，前 ~30s 是环境构建+GraspNet
    # 感知，不走步）。取 7000 ≈ 4.4× 已观测最大尝试，正常负载下约 220s
    # 会先于挂钟兜底触发。
    ap.add_argument("--attempt-steps", type=int, default=7000,
                    help="单条轨迹仿真步数上限（确定性截断，主判据）；0=不限")
    # 兜底：挂钟截断，只在"步数没走够但卡住"时才触发。300s 是取舍后的取值：
    # 对"不走步但卡住"的轨迹（如 object:7，600s 只走 ~2100 步），600s 会让
    # 每条失败尝试的代价翻倍（8×600s≈80min/轮）；代价是"走步但慢"的轨迹上
    # 挂钟可能先于步数预算触发。
    ap.add_argument("--attempt-timeout", type=int, default=300,
                    help="单条轨迹挂钟兜底秒数（默认 300；主判据是 --attempt-steps）")
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="忽略台账已通过记录，强制重跑")
    args = ap.parse_args()

    if args.tasks:
        queue = [t.strip() for t in args.tasks.split(",") if t.strip()]
        queue = [t if t.startswith("libero:") else f"libero:{t}"
                 for t in queue]
    else:
        suites = [s.strip() for s in args.suites.split(",") if s.strip()] \
            or ["libero_spatial", "libero_object", "libero_goal"]
        queue = build_queue(suites)

    ledger = _load_ledger()
    if not args.force:
        done = {t for t, r in ledger["tasks"].items() if r.get("success")}
        queue = [t for t in queue if t not in done]
    print(f"[runner] 队列 {len(queue)} 个任务: "
          f"{queue[:6]}{' …' if len(queue) > 6 else ''}", flush=True)

    n_ok = 0
    for env_id in queue:
        res = run_task(env_id, max_attempts=args.max_attempts,
                       record=not args.no_video,
                       attempt_timeout=args.attempt_timeout,
                       attempt_steps=args.attempt_steps)
        ledger["tasks"][env_id] = {
            "success": res["success"],
            "attempts": res["attempts"],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _save_ledger(ledger)
        n_ok += int(res["success"])
        if not res["success"]:
            print(f"[runner] {env_id} 达到最大重试次数，继续下一任务",
                  flush=True)
    total = len(ledger["tasks"])
    passed = sum(1 for r in ledger["tasks"].values() if r.get("success"))
    print(f"[runner] 本轮 {n_ok}/{len(queue)} 通过；台账累计 {passed}/{total}",
          flush=True)
    return 0 if n_ok == len(queue) else 2


if __name__ == "__main__":
    raise SystemExit(main())
