"""EpisodeLogger：episode 级结构化执行日志（JSONL），用于事后排查"哪里出了问题"。

设计目标：
- 记录每个任务相关物体的位置 + 机械臂状态（TCP / 关节角 / 抽屉关节 / 碰撞间隙），
  失败时能还原"在哪个 skill、物体在哪里、臂在哪里、间隙多少"。
- 不能每个 sim step 都写盘（一次 episode 上千控制步，数据量大）：
  * sim step 采样默认每 10 步写一条 sample（约 0.5s@20Hz）；
  * skill 边界 / 规划 / 验证等关键事件总是强制快照（一条都不丢）。
- 纯追加 JSONL：每行一个 JSON 事件，grep/jq/pandas 均可直接分析。

事件类型（ev 字段）：
  episode_start / episode_end / attempt_start / attempt_end
  plan            动态规划器为某个未满足目标条件生成一个 method 的 skill 序列
  skill_start / skill_end / sample / condition / verify / note

用法：
    log = EpisodeLogger("drawer_place", bodies=["green_block"],
                        grip_site="0_grip_site", actor="agent0")
    log.attach(env)                 # hook env.step 开始低频采样
    log.episode_start(entry)
    log.plan(cond_desc, method_name, steps)
    log.skill_start(name, params)
    ... skill.execute(...)          # 内部 env.step 自动每 10 步落一条 sample
    log.skill_end(name, result)
    log.verify({...})
    log.episode_end(success=..., **summary)
    log.detach(env); log.close()
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class EpisodeLogger:
    """低频采样的 episode JSONL 日志器。"""

    def __init__(self, task: str, log_dir: Optional[str] = None, *,
                 every: int = 10, actor: str = "agent0",
                 grip_site: Optional[str] = None,
                 bodies: Optional[Iterable[str]] = None,
                 named_joints: Optional[Iterable[str]] = None,
                 verbose: bool = True) -> None:
        """
        every: 每 N 个 sim step 写一条 sample；skill 边界不受此限、总是快照。
        bodies: 始终记录位置的 body 名（物体）；env.obstacle_bodies 会自动并入。
        named_joints: 额外按名记录的关节 qpos（如 "drawer:joint"）。
        """
        self.task = task
        self.every = max(1, int(every))
        self.actor = actor
        self.grip_site = grip_site
        self.bodies = list(bodies or [])
        self.named_joints = list(named_joints or [])
        self.verbose = verbose

        log_dir = Path(log_dir or Path("logs") / "episodes")
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"{task}_{stamp}_{id(self) % 100000:05d}.jsonl"
        self._fh = self.path.open("w", encoding="utf-8")

        self._env = None
        self._orig_step = None
        self.sim_step = 0
        self._t0 = time.time()
        # 关节名 → qpos 地址（惰性构建一次）
        self._arm_joint_addrs: Dict[str, int] = {}
        self._joints_resolved = False
        self._monitor = None
        self._monitor_tried = False

    # ============================================================
    # env.step hook（低频采样）
    # ============================================================

    def attach(self, env) -> None:
        """包装 env.step：每个控制步计数，每 every 步落一条 sample。"""
        assert self._env is None, "logger 已 attach，请勿重复挂载"
        self._env = env
        self._orig_step = env.step
        logger = self

        def _wrapped_step(*a, **kw):
            r = logger._orig_step(*a, **kw)
            logger.sim_step += 1
            if logger.sim_step % logger.every == 0:
                logger._write("sample", logger._snapshot())
            return r

        env.step = _wrapped_step

    def detach(self, env) -> None:
        if self._orig_step is not None and env is self._env:
            env.step = self._orig_step
        self._env = None
        self._orig_step = None

    # ============================================================
    # 状态快照
    # ============================================================

    def _resolve_joints(self, env) -> None:
        """收集机械臂关节（actor 前缀 + finger）的 qpos 地址。"""
        if self._joints_resolved:
            return
        m = env.mj_model
        prefix = self.actor[-1] + "_" if self.actor[-1].isdigit() else ""
        for i in range(m.njnt):
            name = m.joint(i).name
            if not name:
                continue
            if (prefix and name.startswith(prefix)) or "finger" in name:
                self._arm_joint_addrs[name] = int(m.jnt_qposadr[i])
        self._joints_resolved = True

    def _get_monitor(self, env):
        """惰性构建一次碰撞监测器（env.obstacle_bodies 可能在 attach 后才注册）。"""
        if self._monitor_tried:
            return self._monitor
        self._monitor_tried = True
        try:
            from ..skills.primitives.collision import get_monitor
            self._monitor = get_monitor(env, self.actor)
        except Exception:
            self._monitor = None
        return self._monitor

    def _snapshot(self) -> Dict[str, Any]:
        """物体位置 + 机械臂状态的完整快照（sample / skill 边界共用）。"""
        env = self._env
        snap: Dict[str, Any] = {"sim_step": self.sim_step,
                                "t": round(time.time() - self._t0, 3)}
        if env is None:
            return snap
        try:
            # 1) TCP
            if self.grip_site:
                tcp = env.get_site_pos(self.grip_site)
                snap["tcp"] = [round(float(v), 4) for v in np.asarray(tcp, float)]
            # 2) 机械臂 + 手指关节角
            self._resolve_joints(env)
            qp = env.mj_data.qpos
            if self._arm_joint_addrs:
                snap["arm_qpos"] = {n: round(float(qp[a]), 4)
                                    for n, a in self._arm_joint_addrs.items()}
            # 3) 任务物体 + 障碍 body 位置
            bodies = list(self.bodies)
            bodies += [b for b in (getattr(env, "obstacle_bodies", None) or ())
                       if b not in bodies]
            bpos: Dict[str, List[float]] = {}
            for b in bodies:
                try:
                    bpos[b] = [round(float(v), 4)
                               for v in np.asarray(env.get_body_pos(b), float)]
                except Exception:
                    continue
            if bpos:
                snap["bodies"] = bpos
            # 4) 命名关节（抽屉等）
            jvals: Dict[str, float] = {}
            for jn in self.named_joints:
                try:
                    jvals[jn] = round(float(env.mj_data.joint(jn).qpos[0]), 4)
                except Exception:
                    continue
            if jvals:
                snap["joints"] = jvals
            # 5) 碰撞间隙
            mon = self._get_monitor(env)
            if mon is not None:
                d, pair = mon.clearance()
                snap["clear_mm"] = round(d * 1000.0, 1)
                snap["coll_pair"] = pair
        except Exception as e:  # 快照失败不能影响执行
            snap["snap_error"] = str(e)
        return snap

    # ============================================================
    # 事件 API
    # ============================================================

    def _write(self, ev: str, payload: Optional[Dict[str, Any]] = None) -> None:
        rec = {"ev": ev, **(payload or {})}
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
        self._fh.flush()

    def event(self, ev: str, **kw) -> None:
        self._write(ev, kw)

    def episode_start(self, entry: Dict[str, Any], cfg: Optional[Dict] = None) -> None:
        self._write("episode_start", {
            "task": entry.get("task_name"), "env_id": entry.get("env_id"),
            "robot": entry.get("robot"), "body": entry.get("body"),
            "actor": entry.get("actor"), "grip_site": entry.get("grip_site"),
            "mode": entry.get("mode"), "cfg": cfg or {},
            "log": str(self.path), "sample_every": self.every,
        })
        if self.verbose:
            print(f"[epilog] {self.task} -> {self.path} (sample every {self.every} steps)")

    def attempt_start(self, attempt: int, cand: Dict[str, Any]) -> None:
        self._write("attempt_start", {
            "attempt": attempt,
            "grasp_pt": cand.get("position"),
            "score": cand.get("score"),
            "score_band": cand.get("score_band"),
            "rel_offset": cand.get("rel_offset"),
            "from_memory": bool(cand.get("from_memory")),
            **self._snapshot(),
        })

    def plan(self, attempt: int, goal: str, method: str,
             steps: List[Dict[str, Any]], measures: Optional[Dict] = None) -> None:
        """动态规划器产出一个 method 的 skill 序列。"""
        self._write("plan", {
            "attempt": attempt, "goal": goal, "method": method,
            "measures": measures or {},
            "steps": [{"skill": s["action"], "params": self._brief(s["params"])}
                      for s in steps],
        })
        if self.verbose:
            chain = " -> ".join(s["action"] for s in steps)
            print(f"[epilog] plan[{goal}] via {method}: {chain}")

    def condition(self, attempt: int, name: str, satisfied: bool,
                  measures: Dict[str, Any]) -> None:
        """目标条件观测结果（observe 阶段）。"""
        self._write("condition", {"attempt": attempt, "name": name,
                                  "satisfied": bool(satisfied),
                                  "measures": measures})

    def skill_start(self, name: str, params: Dict[str, Any]) -> None:
        self._write("skill_start", {"skill": name, "params": self._brief(params),
                                    **self._snapshot()})

    def skill_end(self, name: str, result: Dict[str, Any]) -> None:
        self._write("skill_end", {"skill": name,
                                  "ok": bool(result.get("success")),
                                  "result": self._brief(result),
                                  **self._snapshot()})
        if self.verbose:
            tag = "ok " if result.get("success") else "FAIL"
            reason = result.get("reason") or result.get("error") or ""
            print(f"[epilog]   {name:16s} {tag} {reason}")

    def verify(self, measures: Dict[str, Any], ok: bool) -> None:
        self._write("verify", {"ok": bool(ok), "measures": measures,
                               **self._snapshot()})

    def attempt_end(self, attempt: int, ok: bool, fail_phase: str = "",
                    note: str = "") -> None:
        self._write("attempt_end", {"attempt": attempt, "ok": bool(ok),
                                    "fail_phase": fail_phase, "note": note,
                                    **self._snapshot()})

    def episode_end(self, success: bool, attempts: int, steps: int,
                    fail_phases: Optional[List] = None, **extra) -> None:
        self._write("episode_end", {"success": bool(success), "attempts": attempts,
                                    "sim_steps": self.sim_step, "exec_steps": steps,
                                    "fail_phases": fail_phases or [], **extra})

    def note(self, msg: str, **kw) -> None:
        self._write("note", {"msg": msg, **kw})

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    # ============================================================

    @staticmethod
    def _brief(params: Dict[str, Any]) -> Dict[str, Any]:
        """裁剪大字段（plan/result 只留标量与短数组，避免日志膨胀）。"""
        out: Dict[str, Any] = {}
        for k, v in params.items():
            if isinstance(v, (list, tuple)) and len(v) > 6:
                out[k] = f"<{type(v).__name__} len={len(v)}>"
            elif isinstance(v, dict) and len(v) > 8:
                out[k] = f"<dict keys={list(v)[:8]}>"
            else:
                out[k] = v
        return out
