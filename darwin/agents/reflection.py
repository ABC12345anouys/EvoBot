"""adapt_cfg：失败 → 参数修正的纯函数反思（规则单一来源）。

runner_dynamic 进程内循环与 IPC agent_learner 进程共用本模块：
- 入参只读：当前 cfg、失败分类、连续同种失败次数、（可选）几何遥测；
- 出参：new_cfg（不改原 dict）、deltas（实际变更项 old→new，供写 history）、
  note（人类/LLM 可读的反思结论）。
持久化（写 per-env YAML / RAG）由调用方负责，本模块不碰 env 与文件。

设计原则（用户要求）：允许 hard-code 参数，但它们必须是反思学习的产物；
这里的规则只决定"朝哪个方向学、步长多少"，学到的值落在 per-env YAML，
换环境共享同一份代码、各自维护各自的 YAML。
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


def adapt_cfg(cfg: Dict[str, Any], fail_phase: str, streak: int = 1,
              telemetry: Optional[Dict[str, Any]] = None,
              failed_action: Optional[str] = None,
              anchor: Optional[Dict[str, Any]] = None,
              place_dir: Optional[str] = None
              ) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """根据失败分类返回修正后的 cfg。

    连续同种失败加速（streak 越大步长越大）：accel = 1 + 0.3·(streak-1)。
    telemetry：sim 上报的几何遥测（_place_telemetry），让 goal_not_reached
    这类"链走通但终态差一点"的失败也有参数证据，而非盲目调参。
    failed_action：失败发生的 skill 名（如 "carry"/"place"），用于在同一
    失败分类内做相位级分流（timeout + carry → 调 carry_vcap，而非降 k_descend）。
    anchor：末次成功 cfg 快照（可选）。给出时所有变更被裁剪进 anchor 邻域
    （见 _clamp_to_anchor），防止 streak 加速把参数推飞；无 anchor 行为不变。
    place_dir：上一次 place 侧调整的判定方向（"big"/"small"，见 _xy_side）。
    给出时 xy 阈值分流带 ±8mm 滞环，防止落点在 0.03 附近抖动时两条反向
    规则交替调整同一参数；无 place_dir 时按硬阈值分（旧行为）。
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

    if fail_phase == "goal_not_reached":
        # 链全部成功但终态不达标。旧实现此时不调参（只换方法）；现在用
        # telemetry 做几何归因：BDDL On 谓词要求物体-支撑物 xy < 0.03m。
        tel = telemetry or {}
        xy = tel.get("xy_dist")
        side = _xy_side(float(xy), place_dir) if xy is not None else None
        if side == "big":
            # 偏心插指闭合时碗会沿夹爪方向位移，预计算补偿因此失准。
            # 减小偏心比（更靠近居中，位移更小）+ 略降释放高度（落位更准）。
            r0 = float(new_cfg.get("grasp_container_offset_ratio", 0.70))
            _put(new_cfg, deltas, "grasp_container_offset_ratio",
                 max(0.50, r0 - 0.05 * accel))
            ro0 = float(new_cfg.get("release_offset", 0.04))
            _put(new_cfg, deltas, "release_offset",
                 max(0.02, ro0 - 0.005 * accel))
            note = (f"物体-目标 xy={xy:.3f}m（{('滞环保持big' if xy < 0.030 else '≥0.03阈值')}）："
                    f"减小偏心比与释放高度，降低闭合位移")
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

    elif fail_phase == "grip_failed":
        # 没夹住：下得更深、更靠近、闭合更紧；容器再加深插指深度（有上限，
        # 太深 OSC 到不了会变 timeout）。
        _put(new_cfg, deltas, "k_descend",
             float(new_cfg.get("k_descend", 2.0)) * (1.25 * accel))
        _put(new_cfg, deltas, "hover",
             max(0.03, float(new_cfg.get("hover", 0.12)) - 0.02 * accel))
        # stop_above 调整：>0 时调负（下降更深）；到 0 后先开 jit 横向探索；
        # jit 已满仍夹不住→继续调负穿入（细高罐/盒顶部干涉：手指顶在罐顶
        # 把 TCP 顶开致夹空，需指尖深入罐身中上部）。下限 -0.04 防指尖
        # 降到扁平物体底面下方（object_9 教训）。
        _sa = float(new_cfg.get("stop_above", -0.01))
        _jit = float(new_cfg.get("jit", 0.0))
        if _sa > 0:
            _put(new_cfg, deltas, "stop_above", max(0.0, _sa - 0.005 * accel))
        elif _jit < 0.02:
            _put(new_cfg, deltas, "jit", min(0.02, _jit + 0.005 * accel))
        elif _sa > -0.08:
            _put(new_cfg, deltas, "stop_above", max(-0.08, _sa - 0.01 * accel))
        if "grasp_container_z_delta" in new_cfg:
            _put(new_cfg, deltas, "grasp_container_z_delta",
                 min(0.045, float(new_cfg["grasp_container_z_delta"])
                     + 0.004 * accel))
        note = "未夹住：加深下降、减小悬停、容器加深插指"

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
            # 超时分"慢"与"卡死"：加时间只对慢有效；卡死靠柔和逼近（k↓）。
            _put(new_cfg, deltas, "timeout_scale",
                 min(3.0, float(new_cfg.get("timeout_scale", 1.0))
                     * (1.0 + 0.15 * accel)))
            _put(new_cfg, deltas, "k_descend",
                 max(1.0, float(new_cfg.get("k_descend", 2.0)) * (0.85 / accel)))
            _put(new_cfg, deltas, "place_timeout",
                 int(new_cfg.get("place_timeout", 150)) + int(50 * accel))
            note = "超时：放宽时间并降低逼近增益（区分慢/卡死）"

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

    elif fail_phase == "collision":
        _put(new_cfg, deltas, "hover",
             float(new_cfg.get("hover", 0.12)) + 0.03 * accel)
        _put(new_cfg, deltas, "k_descend",
             max(1.0, float(new_cfg.get("k_descend", 2.0)) * (0.8 / accel)))
        note = "碰撞风险：抬高悬停、降低逼近增益"

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
        # other/unknown：保守通用调整——多数失败与抓取深度/逼近速度有关
        _put(new_cfg, deltas, "k_descend",
             float(new_cfg.get("k_descend", 2.0)) * (1.15 * accel))
        _put(new_cfg, deltas, "hover",
             max(0.04, float(new_cfg.get("hover", 0.12)) - 0.01 * accel))
        note = f"未分类失败({fail_phase})：保守加深逼近"

    if anchor:
        clamped = _clamp_to_anchor(new_cfg, deltas, anchor)
        if clamped:
            note += (f"；已锚定裁剪: {', '.join(clamped)}"
                     f"（限制在末次成功值 ±50% 邻域内）")

    return new_cfg, deltas, note
