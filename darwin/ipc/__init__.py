"""darwin.ipc：仿真进程（sim_worker）与学习进程（agent_learner）之间的
进程间通信层。零第三方依赖（标准库 socket + 换行分隔 JSON）。

- protocol.py：消息 schema 与 JsonLineConnection（AF_UNIX）。
- sim_worker.py：常驻仿真进程，env 只创建一次，按 run{cfg} 反复执行 attempt。
- agent_learner.py：反思学习进程，每轮下发参数、分析结果、写回大 skill YAML。
"""
from .protocol import (
    PROTOCOL_VERSION, DEFAULT_SOCK_PATH,
    JsonLineConnection,
    msg_hello, msg_run, msg_stop,
    msg_ready, msg_skill_event, msg_attempt_result, msg_error,
)

__all__ = ["PROTOCOL_VERSION", "DEFAULT_SOCK_PATH", "JsonLineConnection",
           "msg_hello", "msg_run", "msg_stop",
           "msg_ready", "msg_skill_event", "msg_attempt_result", "msg_error"]
