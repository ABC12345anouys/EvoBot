"""agent_learner：反思学习进程（IPC client）。

每轮：从大 skill YAML（darwin/skills/configs/<skill>.<env>.yaml）取当前参数
下发给 sim_worker 跑一次 attempt → 收 skill 事件流与终态遥测 → 失败时用
agents/reflection.py 的规则（+ 可选 LLM）分析并改参数 → 参数变更与反思
结论立即写回 YAML history。循环到 success 或 max-attempts。

失败后的两种继续方式（agent 决策）：
- 可恢复失败（place/goal_not_reached 等，物体仍在夹爪中）：发带 resume
  的 run——sim 回退 N 个物理步（resume_rollback_steps，YAML 可学），从
  失败 skill 用新参数原位续跑，不 reset、不浪费已成功的抓取/抬升；
- 不可逆失败（grip_failed/collision）或同点续跑超过 --max-resume 次：
  发普通 run，sim 全 reset 开始新 attempt。

sim 与 agent 完全解耦：本进程可随时 Ctrl-C 改代码重启，重连后用 YAML 里
的最新参数继续，仿真进程不重新初始化。

交互模式（--interactive）：失败后不自动反思改参，停下打印遥测与 YAML
路径，等你手动改完 YAML 回车 → 重读 YAML 发 resume（resumable=true 时）
或全 reset（resumable=false 时）。resumable=false 时也可输入 q 退出或
强制 attempt。比自动反思快一个数量级，省掉每轮 2-3 分钟抓取/抬升时间。

用法：
    python -m darwin.ipc.agent_learner --task libero_spatial:0 \
        --sock /tmp/darwin_sim.sock --max-attempts 15
    # 交互模式（边跑边改 YAML）：
    python -m darwin.ipc.agent_learner --task libero_spatial:0 --interactive
"""
from __future__ import annotations

import argparse
import datetime
import socket
from typing import Any, Dict, Optional

from ..agents.reflection import adapt_cfg
from ..skills.skill_config import SkillConfigStore
from ..skills.physics_profile import PhysicsProfile
from .protocol import (DEFAULT_SOCK_PATH, PROTOCOL_VERSION, JsonLineConnection,
                       msg_hello, msg_run, msg_stop)

# 同一失败点最多原位续跑次数（再失败 → 全 reset，避免原地打转）
DEFAULT_MAX_RESUME = 2
# 总迭代（attempt + resume）硬上限，防止不可学参数导致无限循环
_ITER_CAP_MULTIPLIER = 3


def _env_label(task: str) -> str:
    return "libero" if ":" in task else "default"


def _task_id_of(task: str) -> Optional[str]:
    """per-task YAML ID（libero_spatial:0 → libero_spatial_0）。"""
    if ":" in task:
        return task.replace(":", "_")
    return None


class AgentLearner:
    def __init__(self, task: str, sock_path: str, max_attempts: int = 15,
                 skill_name: str = "ik_servo", use_llm: bool = False,
                 max_resume: int = DEFAULT_MAX_RESUME,
                 connect_retries: int = 180,
                 interactive: bool = False) -> None:
        self.task = task
        self.sock_path = sock_path
        self.max_attempts = max_attempts
        self.max_resume = max_resume
        self.use_llm = use_llm
        # 交互模式：失败后停下让你改 YAML，回车后续跑/重跑，不自动调参
        self.interactive = interactive
        self._skill_name = skill_name
        self.store = SkillConfigStore.load(
            skill_name, _env_label(task), task=_task_id_of(task))
        self.physics = PhysicsProfile.load(_env_label(task))
        self.conn = JsonLineConnection.connect(
            sock_path, retries=connect_retries, retry_interval=1.0)
        # attempt 可能跑数分钟，给 15 分钟读超时（卡死时不永久挂起）
        self.conn.sock.settimeout(900.0)
        self.fail_streak: Dict[str, int] = {}
        self._llm_agent = None
        # 参数振荡保护：history 里最近一次 success=True 的完整 cfg 快照；
        # 同一失败连续多次时回退到它，终止"调好一个弄坏一个"的发散。
        self._success_cfg = self._load_last_success_cfg()
        if self._success_cfg:
            self.log(f"已载入末次成功参数快照（{len(self._success_cfg)} 项）")

    def _load_last_success_cfg(self) -> Optional[Dict[str, Any]]:
        """从 YAML history 倒序找第一条 result.success=True 的 cfg 快照。"""
        for rec in reversed(getattr(self.store, "history", []) or []):
            if (rec.get("result") or {}).get("success"):
                cfg = rec.get("cfg")
                if isinstance(cfg, dict) and cfg:
                    return dict(cfg)
        return None

    def _last_place_dir(self) -> Optional[str]:
        """history 倒序找最近一次 place 侧参数调整的方向（滞环状态）。

        place_k 升高 / release_offset 降低 = "big"（按 xy 偏大处理）；
        反之 = "small"。无记录返回 None（硬阈值行为）。
        """
        for rec in reversed(getattr(self.store, "history", []) or []):
            deltas = rec.get("deltas") or {}
            if "place_k" in deltas:
                d = deltas["place_k"]
                try:
                    old, new = (d if isinstance(d, list) else (None, d))
                    if old is not None and float(new) != float(old):
                        return "big" if float(new) > float(old) else "small"
                except (TypeError, ValueError):
                    return None
            if "release_offset" in deltas:
                d = deltas["release_offset"]
                try:
                    old, new = (d if isinstance(d, list) else (None, d))
                    if old is not None and float(new) != float(old):
                        return "big" if float(new) < float(old) else "small"
                except (TypeError, ValueError):
                    return None
        return None

    def log(self, msg: str) -> None:
        print(f"[agent_learner] {msg}", flush=True)

    # ---- 物理 profile 更新（带标签观测；只增可信度不改控制链）----

    def _update_physics(self, result: Dict[str, Any]) -> Dict[str, Any]:
        summary = result.get("physics") or {}
        if not isinstance(summary, dict) or not summary:
            return {}
        changed = self.physics.observe(
            summary, success=bool(result.get("success")),
            reason=str(result.get("fail_phase", "")), source=self.task)
        # 观测桶也随 yaml 持久化：agent 进程每任务重启，桶必须跨任务
        # 子进程积累，参数激活样本数才攒得起来。
        self.physics.save()
        if changed:
            self.log(f"物理 profile 更新: {changed}")
        return changed

    # ---- 可选 LLM 反思（默认关；无 key/失败一律静默跳过，不阻塞学习循环）----

    def _maybe_llm_reflect(self, attempt: int, result: Dict[str, Any]) -> None:
        if not self.use_llm:
            return
        try:
            if self._llm_agent is None:
                from ..agents.agent import ManipulationAgent
                self._llm_agent = ManipulationAgent(
                    rag=None, task_name=self.task, seed=f"ipc{attempt}")
            traj = [{"action": e.get("action"),
                     "result": {"success": e.get("success"),
                                "reason": e.get("reason")}}
                    for e in result.get("events", [])]
            feat = {"shape": "libero_object", "size": [0.02, 0.02, 0.02]}
            self._llm_agent.reflect(
                self.task, feat, traj, success=bool(result.get("success")),
                fail_cat=result.get("fail_phase", ""))
        except Exception as e:  # noqa: BLE001
            self.log(f"LLM 反思跳过: {e}")

    # ---- 主循环 ----

    def run(self) -> bool:
        self.conn.send(msg_hello(self.task))
        ready = self.conn.recv()
        if ready is None or ready.get("type") != "ready":
            raise ConnectionError(f"握手失败，sim 回了: {ready}")
        if int(ready.get("protocol", 0)) != PROTOCOL_VERSION:
            raise ConnectionError(
                f"协议版本不匹配: server={ready.get('protocol')} "
                f"client={PROTOCOL_VERSION}")
        self.log(f"已连接 sim（task={ready.get('task')}），起始参数: "
                 f"{ {k: self.store.params[k] for k in list(self.store.params)[:6]} }"
                 f" …共 {len(self.store.params)} 个")

        success = False
        attempt = 1            # 全 reset 次数编号（sim 侧同一编号）
        total_iters = 0        # attempt + resume 总迭代（硬上限保护）
        iter_cap = self.max_attempts * _ITER_CAP_MULTIPLIER
        mode = "attempt"       # "attempt"=全 reset 新跑；"resume"=回退后续跑
        while attempt <= self.max_attempts and total_iters < iter_cap:
            total_iters += 1
            cfg_snapshot = dict(self.store.params)
            resume_req = None
            if mode == "resume":
                resume_req = {
                    "rollback_steps": int(
                        self.store.params.get("resume_rollback_steps", 10))}
            self.conn.send(msg_run(attempt, cfg_snapshot, resume=resume_req))
            result = self._collect_attempt_result(attempt)
            if result is None:
                return False  # sim 退出/致命错误

            self._maybe_llm_reflect(attempt, result)
            self._update_physics(result)
            ok = bool(result.get("success"))
            fail_phase = result.get("fail_phase", "unknown")
            kind = result.get("kind", mode)
            tel = result.get("telemetry") or {}

            if ok:
                note = ("回退续跑成功，参数定型" if kind == "resume"
                        else "attempt 成功，参数定型")
                self.store.append_history(self._history_record(
                    attempt, cfg_snapshot, result, deltas={}, note=note))
                self.store.mark_learned(True)
                self.store.save()
                self._success_cfg = dict(cfg_snapshot)
                tag = "回退续跑" if kind == "resume" else "attempt"
                self.log(f"✔ attempt {attempt} {tag}成功！参数已写回 "
                         f"{self.store.path.name}（learned=true）")
                success = True
                break

            # 失败：连续同种失败加速；不同 fail 出现时重置其他 streak
            self.fail_streak[fail_phase] = self.fail_streak.get(fail_phase, 0) + 1
            for k in list(self.fail_streak.keys()):
                if k != fail_phase:
                    self.fail_streak[k] = 0

            if self.interactive:
                # 交互模式：不自动改参，停下让你改 YAML，回车后续跑/重跑。
                # 仍记录一条 history 便于事后复盘（deltas 留空，改参由用户在 YAML
                # 侧完成，下轮 send 时自然重读到）。
                self.store.append_history(self._history_record(
                    attempt, cfg_snapshot, result, deltas={},
                    note=f"interactive: {fail_phase}，待用户改 YAML"))
                self.store.save()
                next_mode = self._interactive_prompt(attempt, result)
                if next_mode == "quit":
                    self.log("用户选择退出，保存 YAML")
                    break
                elif next_mode == "resume":
                    mode = "resume"
                else:  # "attempt"
                    mode = "attempt"
                    attempt += 1
                continue

            # 自动模式：反思规则改参 + 决策 resume/reset
            streak = self.fail_streak[fail_phase]
            if (streak >= 4 and self._success_cfg
                    and any(self.store.params.get(k) != v
                            for k, v in self._success_cfg.items())):
                # 同一失败连续 4+ 次：规则调参已在发散（来回改同一参数），
                # 回退到末次成功快照，改由 resume/换候选探索，不再污染参数。
                # contact_stop_band 例外：它是 calibrate_from_logs 遥测标定的
                # 外部写入，不属于反思漂移，不被本回退冲掉。
                deltas = {k: [self.store.params.get(k), v]
                          for k, v in self._success_cfg.items()
                          if k != "contact_stop_band"
                          and self.store.params.get(k) != v}
                applied = self.store.update_params(deltas)
                note = (f"连续 {streak} 次同类失败：回退到末次成功参数快照 "
                        f"（{len(applied)} 项），停止规则调参")
            else:
                _, deltas, note = adapt_cfg(cfg_snapshot, fail_phase, streak, tel,
                                            result.get("failed_action"),
                                            anchor=self._success_cfg,
                                            place_dir=self._last_place_dir())
                applied = self.store.update_params(deltas)
            self.store.append_history(self._history_record(
                attempt, cfg_snapshot, result, deltas=applied, note=note))
            self.store.save()

            # 决策下一步：可恢复且续跑未超限 → resume（attempt 编号不变）；
            # 否则全 reset 进入下一个 attempt
            n_resume = int(result.get("n_resume", 0))
            if bool(result.get("resumable")) and n_resume < self.max_resume:
                rb = int(self.store.params.get("resume_rollback_steps", 10))
                self.log(f"✘ {kind} {attempt} 失败（{fail_phase}，"
                         f"at {result.get('resume_action')}）→ "
                         f"反思调整: {applied or '无可学参数'}；"
                         f"回退 {rb} 步原位续跑（{n_resume + 1}/{self.max_resume}）")
                mode = "resume"
            else:
                why = ("sim 判定不可续跑（物体可能已滑脱/碰撞）"
                       if not result.get("resumable")
                       else f"已达续跑上限 {self.max_resume} 次")
                self.log(f"✘ {kind} {attempt} 失败（{fail_phase}）→ "
                         f"反思调整: {applied or '无可学参数'}；{why}，下轮全 reset")
                mode = "attempt"
                attempt += 1
        if not success:
            # 全部 attempt 用完仍失败：清掉可能残留的 learned 标记并落盘，
            # 保证 per-task YAML 与台账状态一致（下轮 build_queue 仍会重跑）。
            self.store.mark_learned(False)
            self.store.save()
        # 批学器读取的机器行：LLM token 累计用量（无 LLM 时为 0）
        try:
            from ..llm.client import total_tokens_used
            self.log(f"LLM_TOKENS={total_tokens_used()}")
        except Exception:
            self.log("LLM_TOKENS=0")
        return success

    def _collect_attempt_result(self, attempt: int) -> Optional[Dict[str, Any]]:
        """收 skill_event 流直到 attempt_result（或 error/bye）。"""
        while True:
            try:
                msg = self.conn.recv()
            except socket.timeout:
                self.log("等待 sim 结果超时（15 分钟），终止学习")
                return None
            if msg is None:
                self.log("sim 连接关闭")
                return None
            mtype = msg.get("type")
            if mtype == "skill_event":
                mark = "✓" if msg.get("success") else "✗"
                reason = f" {msg.get('reason')}" if msg.get("reason") else ""
                self.log(f"  [{msg.get('idx')}] {msg.get('action')} {mark}"
                         f"({msg.get('steps')} 步){reason}")
            elif mtype == "attempt_result":
                if int(msg.get("attempt", -1)) != attempt:
                    self.log(f"警告：收到非本轮 attempt 结果 "
                             f"{msg.get('attempt')} != {attempt}，继续等")
                    continue
                return msg
            elif mtype == "error":
                self.log(f"sim 报错: {msg.get('message')}")
                return None
            elif mtype == "bye":
                self.log("sim 已退出（bye）")
                return None
            else:
                self.log(f"忽略未知消息: {mtype}")

    # ---- 交互模式：失败后让用户改 YAML 决定下一步 ----

    def _interactive_prompt(self, attempt: int,
                            result: Dict[str, Any]) -> str:
        """失败后停下，打印遥测+YAML 路径，等用户改 YAML 后回车。

        返回 "resume" | "attempt" | "quit"。
        - resumable=true：默认回车 → resume（回退续跑，省掉前面已成功的步骤）
        - resumable=false：默认回车 → attempt（全 reset 下一个 attempt）
        - 任何情况下都可输入：r=全 reset；q=保存退出；
          a=强制 attempt（resumable=false 仍想重跑时用）
        用户改完 YAML 后回车，会重读 YAML 拿到最新参数。
        """
        tel = result.get("telemetry") or {}
        resumable = bool(result.get("resumable"))
        fail_phase = result.get("fail_phase", "unknown")
        kind = result.get("kind", "attempt")
        n_resume = int(result.get("n_resume", 0))

        # 失败摘要：一行核心遥测 + YAML 路径 + 当前参数快照
        sep = "=" * 64
        print(f"\n{sep}", flush=True)
        print(f"✘ {kind} {attempt} 失败：{fail_phase}", flush=True)
        print(f"  failed_action={result.get('failed_action')} "
              f"resume_action={result.get('resume_action')} "
              f"resumable={resumable} n_resume={n_resume}", flush=True)
        print(f"  xy_dist={tel.get('xy_dist')} z_delta={tel.get('z_delta')} "
              f"bddl={tel.get('bddl_success')} steps={result.get('steps')}",
              flush=True)
        # 关键调参相关字段（便于一眼定位改哪个）
        keys_hint = ["place_k", "hover", "k_descend", "lift_height",
                     "timeout_scale", "resume_rollback_steps",
                     "stop_above", "carry_z_cap_m", "place_release_comp_m"]
        shown = {k: self.store.params.get(k) for k in keys_hint
                 if k in self.store.params}
        print(f"  YAML: {self.store.path}", flush=True)
        print(f"  关键参数: {shown}", flush=True)

        if resumable:
            hint = ("回车 = 重读 YAML 发 resume（回退续跑，省 2-3 分钟）"
                    " | r = 全 reset 下个 attempt"
                    " | q = 保存退出")
        else:
            hint = ("resumable=false（不可回退）"
                    " | 回车 = 全 reset 下个 attempt"
                    " | q = 保存退出"
                    " | a = 强制 attempt（同回车，显式语义）")

        while True:
            try:
                line = input(f"[interactive] {hint}\n> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return "quit"

            # 重读 YAML：用户在编辑器里改完，回车时拿到最新参数
            try:
                self.store = SkillConfigStore.load(
                    self._skill_name, _env_label(self.task),
                    task=_task_id_of(self.task))
                self.log(f"已重读 YAML，最新参数（前 6 个）："
                         f"{ {k: self.store.params[k] for k in list(self.store.params)[:6]} }")
            except Exception as e:  # noqa: BLE001 — 重读失败不致命，沿用旧参数
                self.log(f"重读 YAML 失败（沿用旧参数）：{e}")

            if line == "q":
                return "quit"
            if line == "r":
                return "attempt"
            if line == "a":
                return "attempt"
            # 空行 / 任何其他输入：按 resumable 决定默认动作
            return "resume" if resumable else "attempt"

    @staticmethod
    def _history_record(attempt: int, cfg: Dict[str, Any],
                        result: Dict[str, Any], deltas: Dict[str, Any],
                        note: str) -> Dict[str, Any]:
        tel = result.get("telemetry") or {}
        return {
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
            "attempt": attempt,
            "kind": result.get("kind", "attempt"),
            "failed_action": result.get("failed_action"),
            "rollback_steps": int(result.get("rollback_steps", 0) or 0),
            "cfg": dict(sorted(cfg.items())),
            "result": {
                "success": bool(result.get("success")),
                "fail_phase": result.get("fail_phase", ""),
                "steps": int(result.get("steps", 0)),
                "methods_used": result.get("methods_used", []),
                "xy_dist": tel.get("xy_dist"),
                "z_delta": tel.get("z_delta"),
                "bddl_success": tel.get("bddl_success"),
            },
            "deltas": deltas,
            "note": note,
        }

    def shutdown(self) -> None:
        try:
            self.conn.send(msg_stop())
            self.conn.recv()  # bye
        except OSError:
            pass
        self.conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="darwin 反思学习进程（IPC client）")
    ap.add_argument("--task", required=True,
                    help="与 sim_worker 相同的任务名（如 libero_spatial:0）")
    ap.add_argument("--sock", default=DEFAULT_SOCK_PATH)
    ap.add_argument("--max-attempts", type=int, default=15)
    ap.add_argument("--max-resume", type=int, default=DEFAULT_MAX_RESUME,
                    help="同一失败点最多回退续跑次数（再失败则全 reset）")
    ap.add_argument("--skill", default="ik_servo", help="大 skill 名（对应 configs/<skill>.*.yaml）")
    ap.add_argument("--llm", action="store_true",
                    help="启用 LLM 语义反思（默认关；无 key 自动跳过）")
    ap.add_argument("--interactive", action="store_true",
                    help="交互模式：失败后停下让你改 YAML 回车续跑，不自动反思改参")
    args = ap.parse_args()

    learner = AgentLearner(args.task, args.sock,
                           max_attempts=args.max_attempts,
                           max_resume=args.max_resume,
                           skill_name=args.skill, use_llm=args.llm,
                           interactive=args.interactive)
    try:
        ok = learner.run()
    finally:
        learner.shutdown()
    print(f"\n学习结果: {'SUCCESS' if ok else 'REACHED_MAX_ATTEMPTS'}；"
          f"配置文件: {learner.store.path}")
    raise SystemExit(0 if ok else 2)


if __name__ == "__main__":
    main()
