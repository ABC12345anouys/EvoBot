"""任务无关的操作技能库（RoboDojo adapter）。

每个 skill 是生成器：yield 帧动作，return 结构化结果：
    {"success": bool, "reason": str,
     "mechanism": str(可选，驱动重规划), "measures": {...},
     "world_change": bool}

设计要点（与旧硬编码状态机的区别）：
- 走廊不是任务流程，而是 skill 内部的通用避障：cruise 高度 =
  其他实体 z_top 最大值 + clearance（从实时世界状态算），对碗、方块、
  瓶子都成立。
- 落座高度由抓取时记录的几何关系 rel/half_h 推出（物体底部骑支撑面），
  无任务魔数。
- 失败给出 mechanism（ik_unreachable/stall/grasp_miss…），Agent Loop
  把它喂给 LLM 换臂/换候选/换方向，skill 自己不写重试分支。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from ..llm.client import LLMClient  # noqa: F401  (保证模块轻依赖)
from .world_state import WorldState

# ---- 技能清单（序列化进 planner prompt）----

MANIFEST: List[Dict[str, Any]] = [
    {"skill": "grasp",
     "args": {"target": "实体 label",
              "arm": "left | right | auto（默认 auto，按可达性选）",
              "cand": "抓取候选序号，默认 0",
              "sign": "+1 | -1 | auto（从目标的哪一侧跨沿）"},
     "desc": "抓取桌面上的实体：自动选臂、规划高于障碍的水平走廊、"
             "下降、闭合、抬起并验证"},
    {"skill": "place_on",
     "args": {"target": "被夹持实体 label",
              "support": "支撑实体 label"},
     "desc": "把夹持中的实体放到支撑实体上：自动运输走廊与落座高度、"
             "张开夹爪、垂直退让并验证"},
    {"skill": "move_to",
     "args": {"x": "数字（米）", "y": "数字（米）", "z": "数字（米）",
              "frame": "tip | ee（默认 tip）",
              "gripper": "keep | open | close（默认 keep）"},
     "desc": "移动到指定坐标；用于没有标准抓放模板的步骤"},
    {"skill": "gripper",
     "args": {"value": "open | close | 0~1 的数",
              "frames": "保持帧数，默认 10"},
     "desc": "只控制夹爪开合"},
    {"skill": "lift",
     "args": {"height": "垂直抬升高度（米），默认 0.12"},
     "desc": "将夹持中的实体垂直抬高"},
]

# ---- 注入上下文（model 侧 composition root 提供，skill 不依赖 model）----


@dataclass
class SkillContext:
    adapter: Any
    cfg: dict
    # 回调
    perceive: Callable[[bool], WorldState]
    propose_grasps: Callable[[str], list]
    select_arm: Callable[[list, Optional[str]], Optional[str]]
    # 活状态访问
    get_track: Callable[[], dict]
    get_held: Callable[[], tuple]              # (label, held_off)
    set_held: Callable[[Optional[str], Optional[np.ndarray], Optional[dict]], None]
    get_homes: Callable[[], dict]
    last_matched: Callable[[], set]
    # 抓取几何记忆（label → {rel, half_h, half_xy}），place_on 使用
    grasp_mem: Dict[str, dict] = field(default_factory=dict)


def _ok(**measures) -> dict:
    return {"success": True, "world_change": False, "measures": measures}


def _fail(reason: str, mechanism: Optional[str] = None, **measures) -> dict:
    out = {"success": False, "reason": reason, "measures": measures}
    if mechanism:
        out["mechanism"] = mechanism
    return out


def _gripper_v(v: Any) -> float:
    if isinstance(v, str):
        return {"open": 1.0, "close": 0.0}.get(v.lower(), 1.0)
    return float(np.clip(v, 0.0, 1.0))


def _cruise(state: WorldState, exclude: set, cfg: dict) -> float:
    tops = [e.z_top for e in state.entities if e.label not in exclude]
    base = max(tops) if tops else state.table_z
    return min(float(cfg.get("corridor_z_max", 0.90)),
               base + float(cfg.get("corridor_clear", 0.05)))


def _goto(a, target, *, gripper=1.0, cfg: dict, k=None, vcap=None):
    """统一 goto 参数 + 状态翻译。返回 result 或 None（成功）。"""
    st, info = yield from a.goto(
        target, gripper=gripper,
        k=float(k if k is not None else cfg.get("k", 2.0)),
        vcap=float(vcap if vcap is not None else cfg.get("vcap", 0.05)),
        tip=True,
        tol=float(cfg.get("reach_tol", 0.012)),
        stall_frames=int(cfg.get("stall_frames", 15)),
        timeout=int(cfg.get("leg_timeout", 120)))
    if st != "reached":
        mech = "ik_unreachable" if st == "stall" else "timeout"
        return _fail(f"{st} @ {np.round(np.asarray(target), 3).tolist()}",
                     mech, target=np.asarray(target).tolist(),
                     **(info or {}))
    return None


# ================= skills =================

def gripper(ctx: SkillContext, value="open", frames: int = 10):
    a = ctx.adapter
    gv = _gripper_v(value)
    yield from a.hold(gv, int(frames))
    return _ok(gripper=gv)


def move_to(ctx: SkillContext, x, y, z, frame: str = "tip",
            gripper: str = "keep"):
    a = ctx.adapter
    gv = None if gripper == "keep" else _gripper_v(gripper)
    target = [float(x), float(y), float(z)]
    if frame == "tip":
        st, _info = yield from a.goto(
            target, gripper=gv if gv is not None else 1.0,
            k=float(ctx.cfg.get("k", 2.0)),
            vcap=float(ctx.cfg.get("vcap", 0.05)), tip=True,
            tol=float(ctx.cfg.get("reach_tol", 0.012)),
            stall_frames=int(ctx.cfg.get("stall_frames", 15)),
            timeout=int(ctx.cfg.get("leg_timeout", 120)))
        if st != "reached":
            return _fail(f"{st} @ {target}",
                         "ik_unreachable" if st == "stall" else "timeout",
                         target=target)
        return _ok(target=target)
    # ee 系：单帧步进由调用方负责（此处按一帧目标处理）
    yield a.build_action(target, gripper=gv, tip=False)
    return _ok(target=target)


def lift(ctx: SkillContext, height: float = 0.12):
    a = ctx.adapter
    cur_tip = a._tip_world(a.ee_pose, a.ee_quat)
    target = [float(cur_tip[0]), float(cur_tip[1]),
              float(cur_tip[2]) + float(height)]
    fail_res = yield from _goto(a, target, gripper=0.0, cfg=ctx.cfg)
    if fail_res:
        return fail_res
    return _ok(target=target)


def grasp(ctx: SkillContext, target: str, arm: str = "auto",
          cand: int = 0, sign: str = "auto"):
    a = ctx.adapter
    cfg = ctx.cfg
    cands = ctx.propose_grasps(target)
    if not cands:
        return _fail("无抓取候选", "candidate_exhausted", target=target)
    if sign in ("+1", "-1"):
        want = float(sign)
        filtered = [c for c in cands if float(c.get("sign", 0.0)) == want]
        if filtered:
            cands = filtered
    idx = int(cand) % len(cands)
    plan = cands[idx]
    tip = np.asarray(plan["tip"], float)

    state = ctx.perceive(True)
    cruise = _cruise(state, {target}, cfg)
    preferred = None if arm == "auto" else arm
    sel = ctx.select_arm([[float(tip[0]), float(tip[1]), float(cruise)],
                          [float(tip[0]), float(tip[1]), float(tip[2])]],
                         preferred)
    if sel is None:
        return _fail("臂预检全败：目标与走廊均不可达",
                     "ik_unreachable", target=target, tip=tip.tolist(),
                     cruise=cruise)
    if preferred is not None and sel != preferred:
        return _fail(f"指定臂 {preferred} 不可达（可用 {sel}）",
                     "ik_unreachable", target=target, tip=tip.tolist())
    a.active_arm = sel

    # 水平走廊（从当前位置 goto；adapter 内部按 5cm 分腿保持 IK 分支）
    fail_res = yield from _goto(a, [tip[0], tip[1], cruise],
                                gripper=1.0, cfg=cfg)
    if fail_res:
        return fail_res
    # 垂直下降
    fail_res = yield from _goto(a, tip, gripper=1.0,
                                cfg=cfg, k=float(cfg.get("k_descend", 2.0)))
    if fail_res:
        return fail_res
    # 预合窄缝 → 全闭（吸收伺服瞬态）
    yield from a.hold(0.72, 6)
    yield from a.hold(0.0, 15)

    # 记录抓取几何（place_on 的落座高度依据）
    bowl_c = np.asarray(plan["bowl_c"], float)
    half_h = float(plan["half_z"])
    rel = float(bowl_c[2] - tip[2])       # 实体中心 z − 抓取 tip z
    ctx.grasp_mem[target] = {"rel": rel, "half_h": half_h}
    held_off = np.array([
        float(bowl_c[0] - tip[0]),
        float(bowl_c[1] - tip[1]),
        rel - float(a.ee_to_tip)])

    # 抬起
    cur_tip = a._tip_world(a.ee_pose, a.ee_quat)
    lift_target = [float(cur_tip[0]), float(cur_tip[1]),
                   float(cur_tip[2]) + float(cfg.get("grasp_lift_delta", 0.12))]
    fail_res = yield from _goto(a, lift_target, gripper=0.0, cfg=cfg)
    if fail_res:
        return fail_res

    # 抓取验证：轻量感知，实体仍贴桌 = miss
    light = ctx.perceive(False)
    ent = light.get(target)
    if ent is not None and target in ctx.last_matched() \
            and ent.z_bottom < light.table_z + 0.05:
        ctx.set_held(None, None, plan=plan)
        return _fail("抬起后实体仍在桌面（夹空/滑脱）",
                     "grasp_miss", target=target, tip=tip.tolist())
    ctx.set_held(target, held_off, plan=plan)
    out = _ok(arm=sel, cand=idx, cruise=round(cruise, 3), tip=tip.tolist())
    out["world_change"] = True
    return out


def place_on(ctx: SkillContext, target: str, support: str):
    a = ctx.adapter
    cfg = ctx.cfg
    held, _off = ctx.get_held()
    if held != target:
        return _fail(f"hand 中不是 {target}（held={held}）", "wrong_state",
                     target=target)
    mem = ctx.grasp_mem.get(target)
    if mem is None:
        return _fail("缺少抓取几何记忆", "wrong_state", target=target)

    state = ctx.perceive(True)
    sb = state.get(support)
    if sb is None:
        return _fail(f"支撑实体 {support} 不存在", "wrong_state",
                     support=support)
    cruise = _cruise(state, {target, support}, cfg)
    # 落座 tip z：物体底部（rel − half_h 相对 tip）骑支撑面
    place_z = sb.z_top - float(mem["rel"]) + float(mem["half_h"])
    sx, sy = sb.x, sb.y
    sel = ctx.select_arm([[sx, sy, place_z]], a.active_arm)
    if sel is None:
        return _fail("放置点不可达", "ik_unreachable",
                     place_tip=[sx, sy, place_z], support=support)
    a.active_arm = sel

    cur_tip = a._tip_world(a.ee_pose, a.ee_quat)
    fail_res = yield from _goto(a, [cur_tip[0], cur_tip[1], cruise],
                                gripper=0.0, cfg=cfg)
    if fail_res:
        return fail_res
    fail_res = yield from _goto(a, [sx, sy, cruise], gripper=0.0, cfg=cfg)
    if fail_res:
        return fail_res
    fail_res = yield from _goto(a, [sx, sy, place_z], gripper=0.0,
                                cfg=cfg,
                                k=float(cfg.get("place_k", 1.2)),
                                vcap=float(cfg.get("place_vcap", 0.05)))
    if fail_res:
        return fail_res

    yield from a.hold(1.0, int(cfg.get("open_steps", 10)))
    # 垂直退让（手指向上抽出，不蹭物体）
    fail_res = yield from _goto(a, [sx, sy, place_z + 0.08],
                                gripper=1.0, cfg=cfg)
    if fail_res:
        return fail_res
    ctx.set_held(None, None)
    out = _ok(place_tip=[round(sx, 3), round(sy, 3), round(place_z, 3)],
              support=support, cruise=round(cruise, 3))
    out["world_change"] = True
    return out


SKILLS = {"grasp": grasp, "place_on": place_on, "move_to": move_to,
          "gripper": gripper, "lift": lift}
