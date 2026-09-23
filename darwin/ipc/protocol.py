"""IPC 消息协议 + AF_UNIX 换行 JSON 连接。

拓扑：sim_worker 是 server（env 昂贵、长期常驻），agent_learner 是 client
（可随时改代码重启、断线重连，仿真不重新初始化）。所有消息都是单行 JSON +
"\\n"，字段只用 JSON 原生类型，禁止 numpy 标量（发送前转 float/int/list）。

协议版本：PROTOCOL_VERSION。握手时 client 发 hello，server 回 ready。

消息表：
  agent → sim
    hello  {"type":"hello","protocol":1,"task":"libero_spatial:0"}
    run    {"type":"run","attempt":1,"cfg":{...全量参数...},
             "resume":{"rollback_steps":10}  ← 可选；带它=回退物理状态后续跑，
             sim 不 reset，从失败 skill 用新 cfg 重新执行；缺省=全新 attempt}
    stop   {"type":"stop"}
  sim → agent
    ready          {"type":"ready","protocol":1,"task":...,
                     "param_spec":{...},"initial_cfg":{...}}
    skill_event    {"type":"skill_event","attempt":1,"idx":0,"action":...,
                     "success":bool,"reason":...,"steps":int,"tcp":[x,y,z],
                     "body":[x,y,z]}
    attempt_result {"type":"attempt_result","attempt":1,"success":bool,
                     "fail_phase":...,"failed_action":...,"resume_action":...,
                     "resumable":bool,"n_resume":int,"kind":"attempt|resume",
                     "rollback_steps":int,"measures":{...},"telemetry":{...},
                     "steps":int,"methods_used":[...],"events":[...],
                     "log_path":...}
    error          {"type":"error","message":...}
    bye            {"type":"bye"}
"""
from __future__ import annotations

import json
import os
import socket
import time
from typing import Any, Dict, Optional

PROTOCOL_VERSION = 1
DEFAULT_SOCK_PATH = "/tmp/darwin_sim.sock"


# ============================================================
# 消息构造（集中 schema，两边共用，禁止手写 type 字符串散落各处）
# ============================================================

def msg_hello(task: str) -> Dict[str, Any]:
    return {"type": "hello", "protocol": PROTOCOL_VERSION, "task": task}


def msg_run(attempt: int, cfg: Dict[str, Any],
            resume: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    out = {"type": "run", "attempt": int(attempt), "cfg": dict(cfg)}
    if resume is not None:
        out["resume"] = dict(resume)
    return out


def msg_stop() -> Dict[str, Any]:
    return {"type": "stop"}


def msg_ready(task: str, param_spec: Dict[str, Any],
              initial_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {"type": "ready", "protocol": PROTOCOL_VERSION, "task": task,
            "param_spec": param_spec, "initial_cfg": initial_cfg}


def msg_skill_event(attempt: int, event: Dict[str, Any]) -> Dict[str, Any]:
    out = {"type": "skill_event", "attempt": int(attempt)}
    out.update(event)
    return out


def msg_attempt_result(attempt: int, result: Dict[str, Any]) -> Dict[str, Any]:
    out = {"type": "attempt_result", "attempt": int(attempt)}
    out.update(result)
    return out


def msg_error(message: str) -> Dict[str, Any]:
    return {"type": "error", "message": str(message)}


def msg_bye() -> Dict[str, Any]:
    return {"type": "bye"}


# ============================================================
# 连接：AF_UNIX + 换行 JSON
# ============================================================

class ProtocolError(RuntimeError):
    pass


class JsonLineConnection:
    """一条已接受/已连接的 unix socket，按行收发 JSON。"""

    def __init__(self, sock: socket.socket, timeout: Optional[float] = None) -> None:
        self.sock = sock
        self._buf = b""
        if timeout is not None:
            self.sock.settimeout(timeout)

    # ---- 收发 ----

    def send(self, msg: Dict[str, Any]) -> None:
        # 禁止 numpy 类型混入：ensure_ascii=False 对中文 note 更友好
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        self.sock.sendall(data)

    def recv(self) -> Optional[Dict[str, Any]]:
        """阻塞读一条消息；对端干净关闭返回 None。"""
        while b"\n" not in self._buf:
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                raise
            except OSError:
                return None
            if not chunk:
                if self._buf.strip():
                    raise ProtocolError(
                        f"连接关闭但有残余数据: {self._buf!r}")
                return None
            self._buf += chunk
        line, self._buf = self._buf.split(b"\n", 1)
        line = line.strip()
        if not line:
            return self.recv()
        try:
            msg = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as e:
            raise ProtocolError(f"非法 JSON 行: {e}") from e
        if not isinstance(msg, dict) or "type" not in msg:
            raise ProtocolError(f"消息必须是带 type 的 dict: {msg!r}")
        return msg

    def close(self) -> None:
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        try:
            self.sock.close()
        except OSError:
            pass

    # ---- server / client 建立 ----

    @classmethod
    def listen(cls, path: str) -> socket.socket:
        """创建并 bind+listen 一个 unix socket（幂等覆盖残留 socket 文件）。"""
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(path):
            os.unlink(path)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(path)
        listener.listen(1)
        os.chmod(path, 0o660)
        return listener

    @classmethod
    def accept(cls, listener: socket.socket,
               timeout: Optional[float] = None) -> "JsonLineConnection":
        conn, _ = listener.accept()
        return cls(conn, timeout=timeout)

    @classmethod
    def connect(cls, path: str, *, retries: int = 0,
                retry_interval: float = 1.0,
                timeout: Optional[float] = None) -> "JsonLineConnection":
        """连接 sim server；retries>0 时按固定间隔重试（agent 早于 sim 启动）。"""
        last_err: Optional[Exception] = None
        for i in range(retries + 1):
            try:
                sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                if timeout is not None:
                    sock.settimeout(timeout)
                sock.connect(path)
                return cls(sock, timeout=timeout)
            except OSError as e:
                last_err = e
                if i >= retries:
                    break
                time.sleep(retry_interval)
        raise ConnectionError(f"连接 sim 失败 {path}: {last_err}")
