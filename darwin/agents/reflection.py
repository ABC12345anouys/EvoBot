"""adapt_cfg：失败 → 参数修正的纯函数反思（规则单一来源）。

runner_dynamic 进程内循环与 IPC agent_learner 进程共用本模块：
- 入参只读：当前 cfg、失败分类、机制标签（可选）、连续同种失败次数、
  （可选）几何遥测；
- 出参：new_cfg（不改原 dict）、deltas（实际变更项 old→new，供写 history）、
  note（人类/LLM 可读的反思结论）。
持久化（写 per-env YAML / RAG）由调用方负责，本模块不碰 env 与文件。

P0-6 后只剩两类规则（旧盲方向 if-elif 已退役）：
1. 机制驱动（physics/discriminators 的 mechanism）——物理方向的唯一
   来源，与步级重试 retry.py 同一张方向表，attempt 级与 step 级一致；
2. 遥测/约束驱动——goal_not_reached/place_failed 的 xy 滞环分流、stall
   的绕障余量、力控参数，这些是几何/约束层工程，不是物理知识负债。
无机制且无遥测证据的失败：不改参（旧 other/unknown 的"保守加深逼近"
是盲方向，已删——不知道原因时乱调参数只会污染 cfg，交给 resume/换候选）。
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _put(new_cfg: Dict[str, Any], deltas: Dict[str, Any], key: str,
         new_val) -> None:
    """写入一个参数并记录 old→new（仅当真的变化时）。"""
    old = new_cfg.get(key)
    new_cfg[key] = new_val
    if old != new_val:
        def _f(v):
            return round(float(v), 4) if isinstance(v, (int, float)) else v
        deltas[key] = [_f(old), _f(new_val)]


_INFRA_PHASES = frozenset({
    "resume_unsupported", "resume_unavailable", "rollback_failed",
    "sim_exception", "unknown_skill", "no_candidate",
})

# 末次成功快照锚定的漂移下限：anchor=0 或极小时窗口不小于该值
# （= 规则一步的基础步长）。相对窗口取 ±50%·|anchor| 与此下限的较大者。
_DRIFT_FLOOR = {
    "carry_vcap": 0.03, "hover": 0.02, "k": 1.0, "k_descend": 0.5,
    "vcap": 0.1, "jit": 0.005, "stop_above": 0.01, "contact_stop_band": 0.02,
    "place_vcap": 0.01,
    "timeout_scale": 0.2, "release_offset": 0.005, "lift_height": 0.02,
    "place_k": 0.3, "place_timeout": 25, "reach_tol": 0.003,
    "grasp_container_offset_ratio": 0.1, "grasp_container_z_delta": 0.005,
    "grasp_max_width": 0.005, "stiffness": 30.0, "damping": 10.0,
    "spiral_radius": 0.002, "push_force_n": 1.0, "push_depth_m": 0.004,
    "force_limit": 5.0,
}


def _clamp_to_anchor(new_cfg: Dict[str, Any], deltas: Dict[str, Any],
                     anchor: Optional[Dict[str, Any]]) -> list:
    """把本轮变更裁剪进末次成功快照的邻域，返回被裁剪的参数名。

    规则带 streak 加速，连续失败时一步可以把参数推出几倍（k_descend ×2/次），
    这是"k_descend 被反思改乱"的根源。约束：任一参数单次反思后必须落在
    anchor ± max(0.5·|anchor|, 下限) 内；跨档学习（如 stop_above 穿入到
    -0.075）必须经过"成功"逐档 ratchet——anchor 随每次成功更新。
    无 anchor（尚无成功记录）时不裁剪，由 PARAM_SPEC 兜底。
    """
    if not anchor:
        return []
    clamped = []
    for key in list(deltas):
        if key not in anchor or key not in new_cfg:
            continue
        try:
            a = float(anchor[key])
            v = float(new_cfg[key])
        except (TypeError, ValueError):
            continue
        w = max(0.5 * abs(a), _DRIFT_FLOOR.get(key, abs(a) * 0.1 + 0.01))
        c = min(a + w, max(a - w, v))
        if c != v:
            def _f(x):
                return round(float(x), 4)
            new_cfg[key] = _f(c)
            deltas[key] = [_f(deltas[key][0]), _f(c)] if isinstance(
                deltas[key], list) else _f(c)
            clamped.append(key)
    return clamped


# BDDL On 判定的 xy 阈值 0.03m 附近的滞环带宽：落点恰在阈值两侧抖动时
# （xy 0.024~0.034），不加滞环会导致 place_k/release_offset 随 fail_phase
# 交替反向调整、在 resume 循环里无限往返（goal:8/9 实测振荡）。
_XY_ON_THRESHOLD = 0.030
_XY_HYSTERESIS = 0.008


def _xy_side(xy: float, last_dir: Optional[str]) -> str:
    """判定落点属于哪一侧，带滞环。

    last_dir：上一次 place 侧调整把参数往哪边推（"big"=按 xy 偏大处理，
    提高 place_k/降 release_offset；"small"=反之）。要翻转判定，xy 必须越过
    阈值 ±_XY_HYSTERESIS，否则维持原判定。无 last_dir 时按硬阈值分。
    """
    if last_dir == "small" and xy < _XY_ON_THRESHOLD + _XY_HYSTERESIS:
        return "small"
    if last_dir == "big" and xy > _XY_ON_THRESHOLD - _XY_HYSTERESIS:
        return "big"
    return "big" if xy >= _XY_ON_THRESHOLD else "small"


# ---- 机制 → 参数方向（与 policies/retry.py 的步级表同源同向）----
#
# attempt 级调的是 cfg 持久键（k/vcap/place_k/…），step 级调的是步内
# params；两级的"哪个机制往哪个方向调"必须一致，否则同一次失败在步内
# 和 attempt 级被往相反方向推。失效边界：geometry_squeeze 等几何类
# 机制参数无效（参数不是几何的解），只出 note 交给 derive/换候选。
def _mechanism_adapt(new_cfg: Dict[str, Any], deltas: Dict[str, Any],
                     mechanism: str, action: Optional[str],
                     accel: float) -> Optional[str]:
    """机制驱动调参；返回 note。返回 None = 该机制无参数方向。"""
    if mechanism == "budget_short":
        _put(new_cfg, deltas, "timeout_scale",
             min(3.0, float(new_cfg.get("timeout_scale", 1.0))
                 * (1.0 + 0.15 * accel)))
        if action == "place":
            _put(new_cfg, deltas, "place_timeout",
                 int(new_cfg.get("place_timeout", 150)) + int(50 * accel))
        return (f"机制=budget_short（慢型 stall，只有时间能治）："
                f"放宽时间")
    if mechanism == "contact_blocked":
        # 被挡：压得更狠只会更卡——更软更慢（k/vcap 双降）。
        _put(new_cfg, deltas, "k",
             max(0.5, float(new_cfg.get("k", 2.0)) * (0.85 / accel)))
        _put(new_cfg, deltas, "vcap",
             max(0.01, float(new_cfg.get("vcap", 0.05)) * (0.85 / accel)))
        return "机制=contact_blocked（有接触但不接受）：更软更慢"
    if mechanism == "friction_slip":
        if action == "place":
            # 滑脱：降惯性（vcap↓）+ 压稳（k↑）
            _put(new_cfg, deltas, "place_k",
                 min(4.0, float(new_cfg.get("place_k", 1.2))
                     * (1.15 * accel)))
            _put(new_cfg, deltas, "place_vcap",
                 max(0.005, float(new_cfg.get("place_vcap", 0.05))
                     * (0.85 / accel)))
            return "机制=friction_slip@place：降惯性压稳"
        # 夹持期滑脱（lift_no_grip）：压更深（stop_above↓）+ 逼近更稳（k↑）
        _put(new_cfg, deltas, "stop_above",
             max(-0.09, float(new_cfg.get("stop_above", -0.01))
                 - 0.005 * accel))
        _put(new_cfg, deltas, "k",
             min(12.0, float(new_cfg.get("k", 2.0)) * (1.15 * accel)))
        return "机制=friction_slip@grasp：压更深夹更稳"
    if mechanism == "ik_unreachable":
        # xy 收敛乏力：增益上调 + 时间（与 retry 的 ik_unreachable 同向）
        _put(new_cfg, deltas, "k",
             min(12.0, float(new_cfg.get("k", 2.0)) * (1.2 * accel)))
        _put(new_cfg, deltas, "timeout_scale",
             min(3.0, float(new_cfg.get("timeout_scale", 1.0))
                 * (1.0 + 0.10 * accel)))
        return "机制=ik_unreachable（xy 收敛乏力）：增益上调+时间"
    if mechanism == "geometry_squeeze":
        # 几何挤压：参数不是几何的解，交给 derive/换候选，不乱调参
        return ("机制=geometry_squeeze（几何挤压）：参数无效，"
                "待几何推导/换候选，保持 cfg")
    # reach_limit 等：已被接受准则处理或参数无方向
    return None


def adapt_cfg(cfg: Dict[str, Any], fail_phase: str, streak: int = 1,
              telemetry: Optional[Dict[str, Any]] = None,
              failed_action: Optional[str] = None,
              anchor: Optional[Dict[str, Any]] = None,
              place_dir: Optional[str] = None,
              mechanism: Optional[str] = None
              ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """根据失败分类返回修正后的 cfg。

    优先级：基础设施类 → 机制驱动（判别子标签）→ 遥测/约束类规则
    （goal_not_reached/place_failed/stall/force_exceed/carry-timeout）→
    无证据不改参。

    连续同种失败加速（streak 越大步长越大）：accel = 1 + 0.3·(streak-1)，
    但所有变更仍被 anchor 邻域裁剪（见 _clamp_to_anchor）。
    telemetry：sim 上报的几何遥测（_place_telemetry），让 goal_not_reached
    这类"链走通但终态差一点"的失败也有参数证据，而非盲目调参。
    failed_action：失败发生的 skill 名（如 "carry"/"place"），用于在同一
    失败分类内做相位级分流（timeout + carry → 调 carry_vcap）。
    anchor：末次成功 cfg 快照（可选），所有变更被裁剪进 anchor 邻域。
    place_dir：上一次 place 侧调整的判定方向（"big"/"small"，见 _xy_side）。
    mechanism：判别子（physics/discriminators）给出的机制标签；给出时物理
    方向由机制表决定，盲 fail_phase 规则不再介入。
    """
    new_cfg = dict(cfg)
    deltas: Dict[str, Any] = {}
    accel = 1.0 + 0.3 * max(0, streak - 1)
    note = ""

    if fail_phase in _INFRA_PHASES:
        # 基础设施类失败与 skill 参数无关：调参只会污染 cfg
        # （resume 被拒一次就被加深一次 k_descend 的教训）。
        note = f"基础设施类失败({fail_phase})：与参数无关，保持 cfg"
        return new_cfg, deltas, note

    if mechanism:
        mech_note = _mechanism_adapt(new_cfg, deltas, mechanism,
                                     failed_action, accel)
        if mech_note is not None:
            note = mech_note
            if anchor:
                clamped = _clamp_to_anchor(new_cfg, deltas, anchor)
                if clamped:
                    note += (f"；已锚定裁剪: {', '.join(clamped)}"
                             f"（限制在末次成功值 ±50% 邻域内）")
            return new_cfg, deltas, note
        # 机制无参数方向：落到下面的遥测/约束规则或保持 cfg

    if fail_phase == "goal_not_reached":
        # 链全部成功但终态不达标。旧实现此时不调参（只换方法）；现在用
        # telemetry 做几何归因：BDDL On 谓词要求物体-支撑物 xy < 0.03m。
        tel = telemetry or {}
        xy = tel.get("xy_dist")
        side = _xy_side(float(xy), place_dir) if xy is not None else None
        if side == "big":
            # 偏心插指闭合时碗会沿夹爪方向位移，预计算补偿因此失准。
            # 减小偏心比（更靠近居中，位移更小）+ 略降释放高度（落位更准）。
            # 0.62 硬地板：ratio<0.6 时插指进不了容器侧壁，8/8 夹空
            # （spatial:9 死亡螺旋 0.70→0.52 实证）；到地板后改调释放/放置，
            # 不再继续压 ratio。
            r0 = float(new_cfg.get("grasp_container_offset_ratio", 0.70))
            if r0 > 0.62:
                _put(new_cfg, deltas, "grasp_container_offset_ratio",
                     max(0.62, r0 - 0.05 * accel))
            else:
                _put(new_cfg, deltas, "release_offset",
                     max(0.01, float(new_cfg.get("release_offset", 0.02))
                         - 0.004 * accel))
                _put(new_cfg, deltas, "place_k",
                     min(4.0, float(new_cfg.get("place_k", 1.2))
                         * (1.1 * accel)))
            ro0 = float(new_cfg.get("release_offset", 0.04))
            if r0 > 0.62:
                _put(new_cfg, deltas, "release_offset",
                     max(0.02, ro0 - 0.005 * accel))
            note = (f"物体-目标 xy={xy:.3f}m（{('滞环保持big' if xy < 0.030 else '≥0.03阈值')}）："
                    f"减小偏心比与释放高度，降低闭合位移"
                    f"{'（ratio 已到地板，改调释放/放置）' if r0 <= 0.62 else ''}")
        elif side == "small":
            # xy 已达标但仍失败：通常是接触/高度问题（z_delta 为碗-支撑面）
            zd = float(tel.get("z_delta", 0.0))
            lh0 = float(new_cfg.get("lift_height", 0.10))
            _put(new_cfg, deltas, "lift_height",
                 min(0.16, lh0 + 0.01 * accel))
            note = (f"xy={xy:.3f}m 已达标但谓词仍 False（z_delta={zd:.3f}），"
                    f"略增抬升高度重落位")
        else:
            note = "链走通但终态不达标且无几何遥测：保持 cfg，交给换方法/换候选"

    elif fail_phase == "stall":
        # 撞墙磨停（follow_waypoints 30 步 <0.5mm fail-fast）：不是慢，提速/
        # 加预算无效（撞墙型 move_timeout 实证）。加大绕障余量让下次 plan
        # 飞得更高/绕得更外；渐进重规划，两次后余量涨幅加倍。
        if failed_action == "carry":
            _put(new_cfg, deltas, "carry_margin",
                 min(0.30, float(new_cfg.get("carry_margin", 0.0))
                     + (0.05 if streak < 3 else 0.10) * accel))
            note = (f"carry 撞墙磨停（stall）：加大绕障余量 carry_margin 换路径，"
                    f"不提速")
        else:
            # above/waypoint 撞墙：路径已由 avoidance 几何绕障解决，不动
            # k_descend（与逼近无关）；只放宽时间（绕行走廊更长）。
            _put(new_cfg, deltas, "timeout_scale",
                 min(3.0, float(new_cfg.get("timeout_scale", 1.0))
                     * (1.0 + 0.10 * accel)))
            note = f"{failed_action or 'skill'} 撞墙磨停（stall）：绕障走廊已重规划，放宽时间"

    elif fail_phase == "timeout":
        if failed_action == "carry":
            # carry 移动超时是跨距/巡航速度问题（撞墙/滑移另算），
            # 不是逼近卡死：不能降 k_descend，应提速 carry_vcap。
            _put(new_cfg, deltas, "carry_vcap",
                 min(0.30, float(new_cfg.get("carry_vcap", 0.05))
                     + 0.03 * accel))
            _put(new_cfg, deltas, "timeout_scale",
                 min(3.0, float(new_cfg.get("timeout_scale", 1.0))
                     * (1.0 + 0.10 * accel)))
            note = "carry 移动超时：提高巡航速度上限 carry_vcap 并适度放宽时间"
        else:
            # 其余相位超时：无机制标签时不猜方向（盲降 k_descend 已退役，
            # 且 k_descend 对 libero IK 链无效），只放宽时间——时间对"慢"
            # 与"卡死"都无害，方向选择留给判别子的机制标签。
            _put(new_cfg, deltas, "timeout_scale",
                 min(3.0, float(new_cfg.get("timeout_scale", 1.0))
                     * (1.0 + 0.15 * accel)))
            _put(new_cfg, deltas, "place_timeout",
                 int(new_cfg.get("place_timeout", 150)) + int(50 * accel))
            note = "超时（无机制标签）：只放宽时间，方向留给判别子"

    elif fail_phase == "place_failed":
        # 放置失败按几何遥测分两种，方向相反：
        tel = telemetry or {}
        xy = tel.get("xy_dist")
        side = _xy_side(float(xy), place_dir) if xy is not None else None
        if side == "big":
            # 落点横向超 BDDL On 阈值（物体落在支撑物旁，z_delta 常为负）：
            # 一味放软只会更压不下去。提高放置增益收紧 xy 跟踪、降低释放高度
            # 减少松手落座后的侧向漂移。
            _put(new_cfg, deltas, "place_k",
                 min(4.0, float(new_cfg.get("place_k", 1.2)) * (1.15 * accel)))
            _put(new_cfg, deltas, "release_offset",
                 max(0.01, float(new_cfg.get("release_offset", 0.02))
                     - 0.004 * accel))
            note = (f"放置落点 xy={xy:.3f}m（{('滞环保持big' if xy < 0.030 else '≥0.03偏出支撑物')}）："
                    f"提高放置增益收紧跟踪、降低释放高度")
        else:
            # 横向已到位仍不稳（接触/落座）：放慢放置速度、加长时间。
            _put(new_cfg, deltas, "place_k",
                 max(0.3, float(new_cfg.get("place_k", 1.2)) * (0.85 / accel)))
            _put(new_cfg, deltas, "place_timeout",
                 int(new_cfg.get("place_timeout", 150)) + int(40 * accel))
            _put(new_cfg, deltas, "timeout_scale",
                 min(3.0, float(new_cfg.get("timeout_scale", 1.0)) * 1.1))
            note = "放置不稳（横向已到位）：降低放置增益、放宽放置时间"

    elif fail_phase == "force_exceed":
        # 力控插入：降低目标力/深度、增大力上限、降低刚度（更柔顺）。
        _put(new_cfg, deltas, "push_force_n",
             max(0.5, float(new_cfg.get("push_force_n", 3.0)) * (0.7 / accel)))
        _put(new_cfg, deltas, "push_depth_m",
             max(0.003, float(new_cfg.get("push_depth_m", 0.012)) * (0.7 / accel)))
        _put(new_cfg, deltas, "force_limit",
             float(new_cfg.get("force_limit", 15.0)) * (1.15 * accel))
        _put(new_cfg, deltas, "stiffness",
             float(new_cfg.get("stiffness", 100.0)) * (0.7 / accel))
        _put(new_cfg, deltas, "damping",
             float(new_cfg.get("damping", 40.0)) * (1.0 + 0.2 * accel))
        note = "力超限：降低目标力/深度、增大力上限、降低刚度"

    else:
        # 无机制标签、无遥测证据、非约束类失败：不改参。
        # 盲调参（旧 other/unknown 的"保守加深逼近"）只会污染 cfg——
        # 不知道原因时，resume 回退/换候选/让判别子拿到证据才是正解。
        note = (f"失败({fail_phase}) 无机制标签无遥测：保持 cfg，"
                f"交 resume/换候选/证据采集")

    if anchor:
        clamped = _clamp_to_anchor(new_cfg, deltas, anchor)
        if clamped:
            note += (f"；已锚定裁剪: {', '.join(clamped)}"
                     f"（限制在末次成功值 ±50% 邻域内）")

    return new_cfg, deltas, note
