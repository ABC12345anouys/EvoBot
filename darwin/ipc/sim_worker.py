"""sim_worker：常驻仿真进程。

env（LIBERO/robopal，含 GraspNet）只在启动时创建一次；之后每个 IPC run
消息执行一次 attempt（reset → GraspNet 观察 → 现场规划 skill 链 → 执行 →
终态核验），skill 级事件实时推给 agent_learner。agent 可随时重启重连，
仿真不重新初始化。

失败回退续跑：每个物理步自动存状态快照到环形缓冲（经 servo_step 的
_darwin_post_step 钩子）；run 消息带 resume={rollback_steps:N} 时，恢复
失败前 N 步的物理状态，从失败 skill 用新参数原位续跑，不 reset。

用法（单独启动）：
    MUJOCO_GL=egl python -m darwin.ipc.sim_worker \\
        --task libero_spatial:0 --sock /tmp/darwin_sim.sock

任务名：LIBERO 用 "<suite>:<idx>"（如 libero_spatial:0）；robopal 直接用
benchmark 名（如 pickplace / drawer_place）。
"""
from __future__ import annotations

import argparse
import os
from collections import deque
from pathlib import Path

# 无头渲染引导（先于 mujoco/robopal 导入）
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

import traceback
from typing import Any, Dict, List, Optional

from ..agents.chain_registry import default_registry
from ..agents.experience_store import ExperienceStore
from ..agents.runner_dynamic import DynamicEpisodeRunner
from ..benchmarks import get_benchmark, get_libero_benchmark
from ..envs.physics_checkpoint import restore_physics, save_physics
from ..skills.physics_profile import PhysicsProfile
from ..skills.skill_config import SkillConfigStore, public_spec
from .protocol import (DEFAULT_SOCK_PATH, PROTOCOL_VERSION, JsonLineConnection,
                       msg_attempt_result, msg_bye, msg_error, msg_ready,
                       msg_skill_event)

# 快照环形缓冲：LIBERO 20Hz 下 300 步 ≈ 15 秒；token 几百维 float，<1MB
_CHECKPOINT_MAXLEN = 300
# 可原位续跑的失败类别（物体仍在夹爪中、世界未被不可逆破坏）
_RESUMABLE_PHASES = {"place_failed", "goal_not_reached", "timeout"}
# 可从中段续跑的 skill（失败发生在这些动作上，回退后重做该动作有意义）
_RESUMABLE_ACTIONS = {"place", "carry", "lift", "open_gripper"}


def parse_task(task: str) -> Dict[str, Any]:
    """'libero_spatial:0' → LIBERO entry；其余按 robopal benchmark 名解析。"""
    if ":" in task:
        suite, idx = task.split(":", 1)
        return get_libero_benchmark(suite.strip(), int(idx))
    return get_benchmark(task.strip())


def config_env_of(entry: Dict[str, Any]) -> str:
    """大 skill YAML 的 env 标签：libero 用 mode，其余共用 default。"""
    return "libero" if entry.get("mode") == "libero" else "default"


def task_id_of(entry: Dict[str, Any]) -> Optional[str]:
    """LIBERO per-task YAML ID（如 libero_spatial_0）。"""
    if entry.get("libero_suite"):
        return f"{entry['libero_suite']}_{entry['libero_task_idx']}"
    return None


class SimWorker:
    def __init__(self, task: str, sock_path: str,
                 log_dir: Optional[str] = None,
                 record_dir: Optional[str] = None,
                 verbose: bool = True) -> None:
        self.task_name = task
        self.sock_path = sock_path
        self.verbose = verbose
        self.entry = parse_task(task)
        self.registry = default_registry()
        # rag=None：反思/记忆在 agent 进程；sim 只负责执行与上报。
        # 候选黑名单（directive 执行器）：sim 侧只读（写权限在 agent），
        # 文件在 sock 目录下=会话作用域；blacklist_read 每次调用重读文件，
        # agent 每 attempt 写入后 sim 自然看到最新黑名单。
        self.blacklist_path = str(Path(sock_path).parent / "candidate_blacklist.json")
        # 会话启动清空：黑名单是本次运行的会话知识，不跨运行残留（r14
        # 实证 /tmp 残留会把上一轮的误杀带进下一轮）。
        try:
            if os.path.exists(self.blacklist_path):
                os.remove(self.blacklist_path)
        except Exception:
            pass
        self.runner = DynamicEpisodeRunner(
            self.entry, rag=None, max_attempts=1, verbose=verbose,
            log_dir=log_dir, sample_every=50,
            registry=self.registry, experience=ExperienceStore(),
            blacklist_path=self.blacklist_path, blacklist_write=False)
        self.store = SkillConfigStore.load(
            "ik_servo", config_env_of(self.entry),
            task=task_id_of(self.entry))
        # 物理世界 profile（sim 只读；agent 观测后落盘，下一条 run 消息读到新值）
        self.physics = PhysicsProfile.load(config_env_of(self.entry))
        self.handle = self.runner.prepare_episode(
            record_dir=record_dir, initial_cfg=dict(self.store.params))

        # 物理状态快照环形缓冲（resume 回退用），挂在 env 的 post-step 钩子上
        self._checkpoints: deque = deque(maxlen=_CHECKPOINT_MAXLEN)
        self.handle.env._darwin_post_step = self._snapshot
        self.handle.env._darwin_physics = self.physics
        self.handle.env._darwin_physics_obs = {}
        self._stop = False
        self.attempts = 0
        self.total_steps = 0
        self.success = False
        self.fail_phases = []
        # 会话状态（供 resume 续跑）：上个全 attempt 的候选/方法/续跑计数
        self._last_cand = None
        self._last_method_name: Optional[str] = None
        self._last_body0 = None
        self._n_resume = 0

    def log(self, msg: str) -> None:
        print(f"[sim_worker] {msg}", flush=True)

    def _snapshot(self) -> None:
        try:
            token = save_physics(self.handle.env)
        except Exception:
            return
        action = getattr(self.handle.env, "_darwin_action", None)
        self._checkpoints.append((token, action))

    def _pick_rollback_index(self, resume_action: str,
                             rollback: int) -> int:
        """定位回退快照索引。

        关键：回退点必须在**续跑 skill 自己的执行段内**（碗仍在闭合夹爪中），
        而不是相对 attempt 终态——goal_not_reached 时终态在 open_gripper
        之后，相对终态回退会落到碗已释放的状态，重跑 place 没有意义。
        取 resume_action 最后一个执行段的结尾前 rollback 步（不足则退到
        该段起点，绝不跨到前一个 skill）。
        """
        n = len(self._checkpoints)
        last = n - 1
        while last >= 0 and self._checkpoints[last][1] != resume_action:
            last -= 1
        if last < 0:
            # 快照里没有该 action（缓冲滚掉等）：退化为终态前 rollback 步
            return max(0, n - (rollback + 1))
        first = last
        while first - 1 >= 0 and self._checkpoints[first - 1][1] == resume_action:
            first -= 1
        return max(first, last - rollback)

    # ---- resumable 判定 ----

    def _resume_action(self, ar: Dict[str, Any]) -> Optional[str]:
        """建议续跑起点 skill 名；不可续跑返回 None。"""
        if ar.get("fail_phase") not in _RESUMABLE_PHASES:
            return None
        if ar["fail_phase"] == "goal_not_reached":
            # 链全成功但终态不达标：从 place 重新下放/释放；没有 place 则取末动作
            for t in reversed(ar.get("traj", [])):
                if t["action"] == "place":
                    return "place"
            return ar["traj"][-1]["action"] if ar.get("traj") else None
        # skill 级失败：从失败动作续跑（限于后半链动作）
        fa = ar.get("failed_action")
        return fa if fa in _RESUMABLE_ACTIONS else None

    def _is_resumable(self, ar: Dict[str, Any], rollback_steps: int) -> Optional[str]:
        """返回续跑起点 action；不可续跑/快照不足返回 None。"""
        if len(self.handle.objectives) != 1:
            return None
        action = self._resume_action(ar)
        if action is None:
            return None
        # 至少要有 1 个可恢复的快照（不足 N 步就退到最早可用点，由调用方处理）
        if len(self._checkpoints) == 0:
            return None
        return action

    # ---- run 消息处理 ----

    def _handle_run(self, conn: JsonLineConnection, msg: Dict[str, Any]) -> None:
        resume_req = msg.get("resume")
        if resume_req:
            self._handle_resume(conn, msg, resume_req)
        else:
            self._handle_attempt(conn, msg)

    def _handle_attempt(self, conn: JsonLineConnection,
                        msg: Dict[str, Any]) -> None:
        attempt = int(msg.get("attempt", self.attempts + 1))
        cfg = dict(msg.get("cfg") or self.store.params)
        jit_scale = float(cfg.get("jit", 0.0))
        self._checkpoints.clear()
        self._n_resume = 0
        self.handle.env._darwin_physics_obs = {}
        self.log(f"--- attempt {attempt} 开始（{len(cfg)} 个参数）---")
        self.attempts = attempt
        try:
            ar = self.runner._run_single_attempt(
                self.handle.env, self.handle.objectives,
                attempt=attempt - 1, local_cfg=cfg,
                logger=self.handle.logger,
                fresh_candidates=self.handle.fresh_candidates,
                jit_scale=jit_scale,
                event_cb=lambda ev: self._safe_send(
                    conn, msg_skill_event(attempt, ev)))
        except Exception as e:  # noqa: BLE001 — sim 不能因单次 attempt 异常死掉
            self._send_exception(conn, attempt, e)
            return

        # 缓存会话状态供 resume 使用
        self._last_cand = ar.get("cand")
        self._last_method_name = ar.get("method_name")
        self._last_body0 = ar.get("body0")
        self._emit_result(conn, attempt, ar, kind="attempt")

    def _handle_resume(self, conn: JsonLineConnection,
                       msg: Dict[str, Any], resume_req: Dict[str, Any]) -> None:
        attempt = int(msg.get("attempt", self.attempts))
        rollback = max(1, int(resume_req.get("rollback_steps", 10)))
        cfg = dict(msg.get("cfg") or self.store.params)
        action = self._resume_action_from_cache()
        if (action is None or self._last_cand is None
                or self._last_method_name is None or self._last_body0 is None):
            conn.send(msg_attempt_result(attempt, {
                "success": False, "ok_all": False,
                "fail_phase": "resume_unavailable", "failed_action": None,
                "resumable": False, "n_resume": self._n_resume,
                "measures": {}, "telemetry": {}, "steps": 0,
                "methods_used": [], "events": [],
                "log_path": str(self.handle.logger.path)}))
            return

        # 恢复点：续跑 skill 执行段内、结尾前 rollback 步（碗仍在夹爪中）
        idx = self._pick_rollback_index(action, rollback)
        token = self._checkpoints[idx][0]
        seg_end = idx
        while (seg_end + 1 < len(self._checkpoints)
               and self._checkpoints[seg_end + 1][1] == action):
            seg_end += 1
        actual_back = seg_end - idx
        try:
            restore_physics(self.handle.env, token)
        except Exception as e:  # noqa: BLE001
            self.log(f"状态恢复失败: {e}")
            conn.send(msg_attempt_result(attempt, {
                "success": False, "ok_all": False,
                "fail_phase": "rollback_failed", "failed_action": action,
                "resumable": False, "n_resume": self._n_resume,
                "measures": {}, "telemetry": {}, "steps": 0,
                "methods_used": [], "events": [],
                "log_path": str(self.handle.logger.path)}))
            return

        self.handle.env._darwin_physics_obs = {}
        self._n_resume += 1
        self.log(f"--- attempt {attempt} resume#{self._n_resume}: "
                 f"回退 {actual_back} 步，从 {action} 用新参数续跑 ---")
        try:
            ar = self.runner._resume_from_action(
                self.handle.env, self.handle.objectives,
                method_name=self._last_method_name, failed_action=action,
                local_cfg=cfg, logger=self.handle.logger, cand=self._last_cand,
                body0=self._last_body0,
                event_cb=lambda ev: self._safe_send(
                    conn, msg_skill_event(attempt, ev)),
                resume_seq=self._n_resume)
        except Exception as e:  # noqa: BLE001
            self._send_exception(conn, attempt, e)
            return
        self._emit_result(conn, attempt, ar, kind="resume",
                          rollback_steps=actual_back)

    def _resume_action_from_cache(self) -> Optional[str]:
        """从最近一次结果缓存推断续跑点（_handle_attempt 已判定过 resumable）。"""
        # agent 只会在收到 resumable=True 后发 resume；此处信任上次的失败动作
        return getattr(self, "_last_failed_action", None)

    def _send_exception(self, conn: JsonLineConnection, attempt: int,
                        e: Exception) -> None:
        tb = traceback.format_exc(limit=4)
        self.log(f"attempt {attempt} 异常: {e}\n{tb}")
        conn.send(msg_attempt_result(attempt, {
            "success": False, "ok_all": False,
            "fail_phase": "sim_exception", "error": str(e),
            "failed_action": None, "resumable": False,
            "n_resume": self._n_resume,
            "measures": {}, "telemetry": {}, "steps": 0,
            "methods_used": [], "events": [],
            "log_path": str(self.handle.logger.path)}))

    def _emit_result(self, conn: JsonLineConnection, attempt: int,
                     ar: Dict[str, Any], *, kind: str,
                     rollback_steps: int = 0) -> None:
        self.total_steps += int(ar["n_steps"])
        self.success = bool(ar["verified"])
        self.fail_phases.append({"attempt": attempt, "kind": kind,
                                 "fail_cat": ar["fail_phase"]})
        tel = ar.get("telemetry") or {}
        resume_action = self._is_resumable(ar, rollback_steps)
        resumable = resume_action is not None
        # 缓存建议续跑点，供后续 resume 消息使用
        self._last_failed_action = resume_action if resumable else None
        tag = f"resume#{ar.get('resume_seq', 0)}" if kind == "resume" else "attempt"
        if tel:
            self.log(f"{tag} {attempt} 结果: verified={ar['verified']} "
                     f"fail={ar['fail_phase']} xy_dist={tel.get('xy_dist')} "
                     f"z_delta={tel.get('z_delta')} bddl={tel.get('bddl_success')} "
                     f"resumable={resumable}")
        else:
            self.log(f"{tag} {attempt} 结果: verified={ar['verified']} "
                     f"fail={ar['fail_phase']} resumable={resumable}")
        conn.send(msg_attempt_result(attempt, {
            "success": bool(ar["verified"]),
            "ok_all": bool(ar["ok_all"]),
            "fail_phase": ar["fail_phase"],
            "failed_action": ar.get("failed_action"),
            "resume_action": resume_action,
            "resumable": resumable,
            "n_resume": self._n_resume,
            "kind": kind,
            "rollback_steps": rollback_steps,
            "mechanism": ar.get("mechanism"),
            "cand_xy": self._cand_xy_of(ar),
            "measures": ar["measures"],
            "telemetry": tel,
            "steps": int(ar["n_steps"]),
            "methods_used": ar["methods_used"],
            "events": ar["skill_events"],
            "traj": ar.get("traj", []),
            "physics": dict(getattr(self.handle.env,
                                    "_darwin_physics_obs", {}) or {}),
            "log_path": str(self.handle.logger.path),
        }))

    def _cand_xy_of(self, ar: Dict[str, Any]) -> Optional[List[float]]:
        xy = DynamicEpisodeRunner._cand_xy(ar.get("cand"))
        return [float(v) for v in xy] if xy is not None else None

    def _safe_send(self, conn: JsonLineConnection, message: Dict[str, Any]) -> None:
        try:
            conn.send(message)
        except OSError:
            pass

    # ---- 连接主循环：agent 断线后回 accept 等下一个，env 不销毁 ----

    def serve(self) -> None:
        listener = JsonLineConnection.listen(self.sock_path)
        self.log(f"监听 {self.sock_path}（task={self.task_name}, "
                 f"body={self.entry['body']}），等待 agent 连接…")
        try:
            while not self._stop:
                conn = JsonLineConnection.accept(listener)
                self.log("agent 已连接")
                try:
                    self._serve_conn(conn)
                except (OSError, ConnectionError) as e:
                    self.log(f"连接中断: {e}（env 保留，等待重连）")
                finally:
                    conn.close()
        finally:
            listener.close()
            try:
                os.unlink(self.sock_path)
            except OSError:
                pass
            self.handle.env._darwin_post_step = None
            self.runner.teardown_episode(
                self.handle, success=self.success, attempts=self.attempts,
                total_steps=self.total_steps, fail_phases=self.fail_phases)
            self.log("已退出")

    def _serve_conn(self, conn: JsonLineConnection) -> None:
        hello = conn.recv()
        if hello is None:
            return
        if hello.get("type") != "hello":
            conn.send(msg_error("首条消息必须是 hello"))
            return
        if int(hello.get("protocol", 0)) != PROTOCOL_VERSION:
            conn.send(msg_error(
                f"协议版本不匹配: client={hello.get('protocol')} "
                f"server={PROTOCOL_VERSION}"))
            return
        conn.send(msg_ready(self.task_name, public_spec(),
                            dict(self.store.params)))
        while True:
            msg = conn.recv()
            if msg is None:
                self.log("agent 断开")
                return
            mtype = msg.get("type")
            if mtype == "stop":
                conn.send(msg_bye())
                self._stop = True
                self.log("收到 stop，准备退出")
                return
            if mtype == "run":
                self._handle_run(conn, msg)
            else:
                conn.send(msg_error(f"未知消息类型: {mtype}"))


def main() -> None:
    ap = argparse.ArgumentParser(description="darwin 常驻仿真进程")
    ap.add_argument("--task", required=True,
                    help="LIBERO: '<suite>:<idx>'（如 libero_spatial:0）；"
                         "robopal: benchmark 名（如 pickplace）")
    ap.add_argument("--sock", default=DEFAULT_SOCK_PATH, help="unix socket 路径")
    ap.add_argument("--log-dir", default=None, help="EpisodeLogger JSONL 目录")
    ap.add_argument("--record-dir", default=None, help="mp4 录屏目录")
    ap.add_argument("--quiet", action="store_true", help="减少 stdout 输出")
    args = ap.parse_args()

    worker = SimWorker(args.task, args.sock, log_dir=args.log_dir,
                       record_dir=args.record_dir, verbose=not args.quiet)
    worker.serve()


if __name__ == "__main__":
    main()
