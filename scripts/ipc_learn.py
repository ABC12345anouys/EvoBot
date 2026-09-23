#!/usr/bin/env python
"""一键启动：拉起常驻 sim_worker 子进程 + 在本进程跑 agent_learner。

两个进程也可以分别在两个终端单独启动（便于只重启 agent 调反思逻辑）：
    终端1: MUJOCO_GL=egl python -m darwin.ipc.sim_worker --task libero_spatial:0
    终端2: python -m darwin.ipc.agent_learner --task libero_spatial:0

用法:
    python scripts/ipc_learn.py libero_spatial:0 --max-attempts 15
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="LIBERO: '<suite>:<idx>'；robopal: benchmark 名")
    ap.add_argument("--sock", default="/tmp/darwin_sim.sock")
    ap.add_argument("--max-attempts", type=int, default=15)
    ap.add_argument("--record-dir", default=None, help="sim 侧 mp4 录屏目录")
    ap.add_argument("--log-dir", default=None, help="sim 侧 EpisodeLogger 目录")
    ap.add_argument("--llm", action="store_true", help="启用 LLM 语义反思")
    ap.add_argument("--interactive", action="store_true",
                    help="交互模式：失败后停下让你改 YAML 回车续跑，不自动反思改参")
    ap.add_argument("--max-resume", type=int, default=None,
                    help="同一失败点最多回退续跑次数（仅自动模式生效，默认 2）")
    args = ap.parse_args()

    env = dict(os.environ)
    env.setdefault("MUJOCO_GL", "egl")
    env["PYTHONPATH"] = (str(REPO_ROOT) + os.pathsep
                         + env.get("PYTHONPATH", ""))
    if env.get("MUJOCO_GL") == "egl":
        env.pop("DISPLAY", None)

    sim_cmd = [sys.executable, "-m", "darwin.ipc.sim_worker",
               "--task", args.task, "--sock", args.sock]
    if args.record_dir:
        sim_cmd += ["--record-dir", args.record_dir]
    if args.log_dir:
        sim_cmd += ["--log-dir", args.log_dir]

    print(f"[launcher] 启动 sim_worker: {' '.join(sim_cmd)}", flush=True)
    sim_proc = subprocess.Popen(sim_cmd, cwd=str(REPO_ROOT), env=env)

    # 延迟导入：本进程不碰 mujoco（env 只活在 sim 子进程里）
    sys.path.insert(0, str(REPO_ROOT))
    from darwin.ipc.agent_learner import AgentLearner

    learner = None
    ok = False
    try:
        kw = dict(max_attempts=args.max_attempts, use_llm=args.llm,
                  interactive=args.interactive)
        if args.max_resume is not None:
            kw["max_resume"] = args.max_resume
        learner = AgentLearner(args.task, args.sock, **kw)
        ok = learner.run()
    except KeyboardInterrupt:
        print("\n[launcher] 收到 Ctrl-C，停止…", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[launcher] agent 异常: {e}", flush=True)
    finally:
        if learner is not None:
            learner.shutdown()
        # agent 已发 stop；等 sim 优雅退出，兜底 kill
        try:
            sim_proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            sim_proc.kill()
            sim_proc.wait(timeout=5)
    print(f"[launcher] sim 退出码={sim_proc.returncode}；"
          f"结果={'SUCCESS' if ok else '未成功（见 YAML history）'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
