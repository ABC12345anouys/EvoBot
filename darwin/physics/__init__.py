"""物理认知引擎（P0 四合一）：约束 + 推导 + 割 + 判别子 + 探针。

设计依据 docs/architecture_step_planner.md 附录 A/B：
- 表行是知识（可沉淀、可迁移），本模块是这些表的代码化单一来源。
- 一切判据用物体中心坐标系写（禁世界坐标）；唯一与臂有关的量经
  L2 投影（gripper geometry / 可达性）注入。
- 观测基础设施复用 libero_adapter 的 object_bounds /
  contact_force_on_body（真值源，606 行里最值钱的 200 行）。

模块划分：
  observables.py   证据采集（skill 执行期逐样本观测 → Evidence）
  discriminators.py 表3：遥测签名 → 机制（hypothesis 二分）
  cuts.py          表2：机制 + 证据 → Θ 半空间割
  derives.py       参数推导（timeout= f(距离,速率)、band= f(实测短缩)…）
  probes.py        表4：主动实验（接触/可达/摩擦/几何核对）
  posterior.py     Θ 后验（半空间交 + 置信度 + per-task YAML 持久化）
  constraints.py   表1：L1 约束判据（摩擦锥/支撑多边形/力封闭 lite/
                   可达/夹持力/间隙/接近走廊）
  articulation.py  关节约束类：JointDecl 参数化声明（轴/限位/作用点，
                   从模型白拿）+ 流形/限位/力对齐三判据——一种声明覆盖
                   柜门/抽屉/翻盖/旋钮（参数实例≠类实例）
"""
from .observables import Evidence, Collector, F_FREE_N, RATE_EPS
from .discriminators import Mechanism, classify_stall, classify_slip, classify_no_grip
from .cuts import cut_for
from .posterior import ThetaPosterior
from .articulation import JointDecl, extract_joint_decl, read_q, \
    on_manifold_track, within_limits, aligned_wrench

__all__ = [
    "Evidence", "Collector", "F_FREE_N", "RATE_EPS",
    "Mechanism", "classify_stall", "classify_slip", "classify_no_grip",
    "cut_for", "ThetaPosterior",
    "JointDecl", "extract_joint_decl", "read_q",
    "on_manifold_track", "within_limits", "aligned_wrench",
]
