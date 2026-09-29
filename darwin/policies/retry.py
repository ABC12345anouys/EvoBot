"""步级重试的换参策略（Phase B）。

失败后回滚该步物理快照、换一组参数原地重试（预算 5 次），作用在步内、
不消耗 attempt 预算。

换参只有两条来源（P0-6 退役手工序列表后）：
1. 机制驱动（判别子 physics/discriminators 给出 mechanism）——每种机制
   只换对症的参数，同一原因不重复烧预算（§16.1）；
2. 机制未知时的通用假设阶梯——同型失败只有两类病因："被挡"（更软更慢）
   与"慢"（更多时间），阶梯前段探"被挡"、后段给时间，是两假设竞争的
   最便宜判别序，不是 per-action 拍脑袋表。

见 docs/architecture_step_planner.md §3/§13/§16。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# 每步重试预算（§3 约束层：预算约束）
STEP_RETRY_BUDGET = 5

# 允许步内重试的 step（白名单；close_gripper 等无参可换的重试无意义，
# 不在列——失败直接向上传播，由 attempt 级反思处理）。
# ik_servo 覆盖 libero 链的 above+descend 两步（methods.py 同名注册），
# 换参（k/vcap/timeout）对两者都安全。
RETRYABLE = {
    "move_above", "descend", "ik_servo", "lift", "carry", "place",
    "pose_move_above", "pose_descend", "pose_lift",
}


def is_retryable(action: str) -> bool:
    return action in RETRYABLE


# 通用假设阶梯（机制未知时的探索序）：前 3 步"被挡"假设（k/vcap 递减）
# + 递增时间，后 2 步"慢"假设（恢复原速、时间 ×4.5）。"被挡"与"慢"是
# 同型失败的全部两类病因，该阶梯对任何 retryable 步成立（P0-6）。
_GENERIC_SEQ = [(0.8, 0.6, 1.0),
                (0.7, 0.5, 2.0),
                (0.6, 0.4, 3.0),
                (1.0, 1.0, 4.5),
                (1.0, 1.0, 6.0)]


def _apply_seq(params: Dict[str, Any], retry: int) -> Optional[Dict[str, Any]]:
    k_p, v_p, t_p = _GENERIC_SEQ[min(retry, len(_GENERIC_SEQ)) - 1]
    return _mult(params, k=k_p, vcap=v_p, timeout=t_p)


def retry_params(action: str, params: Dict[str, Any],
                 retry: int, mechanism: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """第 retry 次重试（1-based）的参数；None = 该步此轮不重试。

    机制驱动（判别子给出 mechanism）优先——同一原因不重复烧预算（§16.1），
    每种机制只换对症的参数：
      budget_short    时间给足（慢型 stall 只有时间能治，spatial:4 实证）
      contact_blocked 更软更慢（被挡：压得更狠只会更卡）
      friction_slip   更慢 + 增益递增（滑脱：降惯性 + 压稳）
      ik_unreachable  增益上调 + 时间（xy 收敛乏力）
    机制未知 → 通用假设阶梯（见模块 docstring）。
    """
    if mechanism:
        p = _mechanism_params(action, params, retry, mechanism)
        if p is not None:
            return p
    if action in RETRYABLE:
        return _apply_seq(params, retry)
    return None


def _mult(params: Dict[str, Any], *, k=None, vcap=None, timeout=None) -> Dict[str, Any]:
    import copy as _copy
    p = _copy.deepcopy(params)
    changed = False
    if k and isinstance(p.get("k"), (int, float)):
        p["k"] = float(p["k"]) * k
        changed = True
    if vcap and isinstance(p.get("vcap"), (int, float)):
        p["vcap"] = float(p["vcap"]) * vcap
        changed = True
    if timeout and isinstance(p.get("timeout"), (int, float)):
        p["timeout"] = int(float(p["timeout"]) * timeout)
        changed = True
    return p if changed else None


def _mechanism_params(action: str, params: Dict[str, Any],
                      retry: int, mechanism: str) -> Optional[Dict[str, Any]]:
    """机制 → 参数方向。retry 递进强度；超预算返回 None 走原序列。"""
    step = min(retry, 3)
    if mechanism == "budget_short":
        # 时间序列 ×2/×3/×4.5（非累积，对 base）
        return _mult(params, timeout=(2.0, 3.0, 4.5)[step - 1])
    if mechanism == "contact_blocked":
        return _mult(params, k=(0.8, 0.7, 0.6)[step - 1],
                     vcap=(0.6, 0.5, 0.4)[step - 1])
    if mechanism == "friction_slip":
        if action == "place":
            return _mult(params, k=(1.25, 1.56, 1.95)[step - 1],
                         vcap=(0.6, 0.36, 0.25)[step - 1])
        return _mult(params, vcap=(0.8, 0.64, 0.5)[step - 1])
    if mechanism == "ik_unreachable":
        return _mult(params, k=(1.25, 1.5, 1.75)[step - 1],
                     timeout=(1.5, 2.0, 3.0)[step - 1])
    if mechanism == "reach_limit":
        return None  # 已被接受准则处理，不该到这
    return None  # 其余机制：参数无效（几何问题），走原序列或向上传播
