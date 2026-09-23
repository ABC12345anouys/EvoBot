"""methods：技能方法基类 + 内置方法（从 runner_dynamic.py 抽出）。
Method 声明"能达成哪类条件"（can_achieve）+ 现场生成 skill 序列（make_steps）。
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
from .objectives import (GoalCond, JointAtLeast, BodyLifted, BodyNearSite,
                          LiberoGoal, LiberoSubGoal)
from ..skills.primitives.collision import SAFE_Z
class PlanContext:
    """单次方法展开的上下文：任务参数 + 实时 env + 当前抓取候选。"""
    def __init__(self, entry: Dict[str, Any], env, cand: Dict[str, Any],
                 cfg: Dict[str, Any], is_first: bool,
                 body0_pos=None) -> None:
        self.entry = entry
        self.env = env
        self.cand = cand
        self.cfg = cfg
        self.is_first = is_first
        # plan 时刻（夹取前）物体的桌面初始位姿。偏心放置偏移 grasp-bowl_center
        # 是两个静止世界坐标之差，必须用夹取前的碗位；resume 回退续跑时碗已被
        # lift/carry 移走，现场重算会得到巨大错误偏移把 place target 带偏。
        self.body0_pos = (None if body0_pos is None
                          else np.asarray(body0_pos, dtype=float).reshape(3))
        self.cmn = {"actor": entry["actor"], "grip_site": entry["grip_site"]}
    @property
    def grasp_pt(self) -> List[float]:
        return [float(x) for x in np.asarray(self.cand["position"], float).reshape(3)]
    @property
    def hover(self) -> float:
        return float(self.cfg.get("hover", 0.12))
    @property
    def k_desc(self) -> float:
        return float(self.cfg.get("k_descend", 2.0))
    @property
    def lift_h(self) -> float:
        return float(self.cfg.get("lift_height", 0.52))
    @property
    def timeout_scale(self) -> float:
        """timeout 缩放系数：timeout 类失败后自适应放大，让 skill 有更多收敛时间。"""
        return float(self.cfg.get("timeout_scale", 1.0))
    
class Method:
    """方法基类：can_achieve 声明适用条件，make_steps 动态产出 skill 链。"""
    name = "method"
    includes_home = False          # 生成的链已自带 home，引擎不再补
    flavor: Optional[str] = None   # None=与 flavor 无关
    next: Optional[str] = None     # 方法编排：执行成功后优先选的下一个方法名
    def can_achieve(self, cond: GoalCond, entry: Dict[str, Any]) -> bool:
        raise NotImplementedError
    def make_steps(self, ctx: PlanContext, cond: GoalCond) -> List[Dict[str, Any]]:
        raise NotImplementedError

def _carry_timeout(target, grasp, carry_vcap: float,
                   timeout_scale: float) -> int:
    """carry 预算随跨距自适应：实测每步有效位移 ≈ vcap/80（P 收敛+避障+
    末段容差损耗，object:9 实证 36cm/285 步@vcap0.10），固定 160 步在
    跨距 >20cm 时必 move_timeout（spatial:09 实证）。1.5× 余量，上限 1500。
    """
    try:
        dist = float(np.linalg.norm(np.asarray(target, float)[:2]
                                    - np.asarray(grasp, float)[:2]))
    except Exception:
        dist = 0.0
    return int(min(1500, max(160.0, 120.0 * dist / max(carry_vcap, 0.01))
                   * timeout_scale))


def _grasp_chain(ctx: PlanContext, body: str, *, stop_above: float = 0.0,
                 pose: bool = False, descend_tol: float = 0.012) -> List[Dict[str, Any]]:
    """抓取子链：above → descend → close → lift（cart / pose 两套原语前缀）。
    timeout 显式传入并乘 timeout_scale，保证 timeout 类失败的自适应能生效。
    descend_tol：到达容差（受限空间放宽）。
    """
    p = "pose_" if pose else ""
    pt = ctx.grasp_pt
    cmn = ctx.cmn
    ts = int(90 * ctx.timeout_scale)
    steps = [
        {"action": f"{p}move_above", "params": {"point": pt, "hover": ctx.hover,
                                                "timeout": ts, **cmn}},
        {"action": f"{p}descend", "params": (
            {"point": pt, "body": body, "k": ctx.k_desc,
             "stop_above": stop_above, "reach_tol": descend_tol,
             "timeout": ts, **cmn})},
        {"action": f"{p}close_gripper", "params": {**cmn}},
        {"action": f"{p}lift", "params": {"height": ctx.lift_h, "body": body,
                                          "timeout": ts, **cmn}},
    ]
    return steps

class OpenDrawerMethod(Method):
    """拉开关节式容器：home → 走到把手侧安全高度 → 驱动抽屉关节（skill 内置避障）。"""
    name = "open_drawer"
    includes_home = True

    def can_achieve(self, cond, entry) -> bool:
        return isinstance(cond, JointAtLeast)

    def make_steps(self, ctx, cond: JointAtLeast) -> List[Dict[str, Any]]:
        env, cmn = ctx.env, ctx.cmn
        handle = cond.handle_site or entry_get(ctx.entry, "drawer_site", "drawer")
        hp = np.asarray(env.get_site_pos(handle), float)
        return [
            {"action": "home", "params": {**cmn}},
            {"action": "move_to", "params": {"target": [float(hp[0]), 0.0,
                                                       float(SAFE_Z)],
                                             "gripper": 0.0, **cmn}},
            {"action": "pull_drawer", "params": {"joint_name": cond.joint,
                                                 "target_qpos": cond.target_qpos,
                                                 "handle_site": handle, **cmn}},
        ]


def entry_get(entry: Dict[str, Any], key: str, default: Any) -> Any:
    return entry.get(key, default)


class GraspLiftMethod(Method):
    """抓起来即成功（grasp-only）。"""
    name = "grasp_lift"

    def can_achieve(self, cond, entry) -> bool:
        return isinstance(cond, BodyLifted)

    def make_steps(self, ctx, cond: BodyLifted) -> List[Dict[str, Any]]:
        return _grasp_chain(ctx, cond.body)


def _needs_keep_hold(env, cond: BodyNearSite, body: str) -> bool:
    """goal 明显高于物体当前所在支撑面（空中递送点）时，放置后不能松爪，
    否则物体坠落、终态核验失败。物体当前 z 近似其支撑面高度。"""
    try:
        goal = np.asarray(env.get_site_pos(cond.site), float)
        body_pos = np.asarray(env.get_body_pos(body), float)
        return bool(goal[2] > body_pos[2] + 0.02)
    except Exception:
        return False


class TransferCartMethod(Method):
    """笛卡尔抓放：抓取 → 抬升 → 移到目标上方 → 下放 → 松开。"""
    name = "transfer_cart"
    flavor = "cart"

    def can_achieve(self, cond, entry) -> bool:
        return isinstance(cond, BodyNearSite) and cond.flavor == "cart"

    def make_steps(self, ctx, cond: BodyNearSite) -> List[Dict[str, Any]]:
        env, cmn = ctx.env, ctx.cmn
        goal = np.asarray(env.get_site_pos(cond.site), float)
        steps = _grasp_chain(ctx, cond.body, stop_above=cond.stop_above,
                             descend_tol=cond.descend_tol)
        steps += [
            {"action": "move_to_xy_top",
             "params": {"target": [float(goal[0]), float(goal[1])],
                        "height": ctx.lift_h,
                        "timeout": int(150 * ctx.timeout_scale), **cmn}},
            {"action": "place",
             "params": {"goal": [float(x) for x in goal], "body": cond.body,
                        "tol": cond.tol,
                        "timeout": int(cond.place_timeout * ctx.timeout_scale),
                        "k": float(ctx.cfg.get("place_k", cond.place_k)), **cmn}},
        ]
        # 空中递送点：松爪物体会坠落，保持夹持直到终态核验通过
        if not _needs_keep_hold(env, cond, cond.body):
            steps.append({"action": "open_gripper", "params": {**cmn}})
        return steps


class TransferPoseMethod(Method):
    """位姿+力控插装链（peg-in-hole）：视觉伺服对准 → guarded_move → 阻抗压入 → 螺旋搜索。"""
    name = "transfer_pose"
    flavor = "pose"

    def can_achieve(self, cond, entry) -> bool:
        return isinstance(cond, BodyNearSite) and cond.flavor == "pose"

    def make_steps(self, ctx, cond: BodyNearSite) -> List[Dict[str, Any]]:
        env, cmn = ctx.env, ctx.cmn
        cfg = ctx.cfg
        hole = np.asarray(env.get_site_pos(cond.site), float)
        steps = _grasp_chain(ctx, cond.body, pose=True)
        steps += [
            {"action": "pose_move_to",
             "params": {"target": [float(hole[0]), float(hole[1]),
                                   float(hole[2]) + 0.05], **cmn}},
            {"action": "servo_align",
             "params": {"target": {"type": "pose", "name": cond.site}, **cmn}},
            {"action": "guarded_move",
             "params": {"direction": [0.0, 0.0, -1.0], "until": {"force_n": 2.0},
                        "max_travel_m": 0.15, "speed_mps": 0.05, "timeout": 200, **cmn}},
            {"action": "impedance_push", "name": "push", "on_fail": "spiral",
             "params": {"axis": "z",
                        "until": {"force_n": float(cfg.get("push_force_n", 3.0)),
                                  "depth_m": float(cfg.get("push_depth_m", 0.012))},
                        "force_limit": float(cfg.get("force_limit", 15.0)),
                        "stiffness": float(cfg.get("stiffness", 100.0)),
                        "damping": float(cfg.get("damping", 40.0)), **cmn}},
            {"action": "spiral_search", "name": "spiral",
             "params": {"radius": float(cfg.get("spiral_radius", 0.004)),
                        "until": "depth_reached", **cmn}},
            {"action": "pose_open_gripper", "params": {**cmn}},
        ]
        return steps


class TransferPoseCartMethod(Method):
    """位姿环境下的纯位置抓放链（cube 放置，无力控）。

    InsertEnv 等 CARTIK 位姿环境不支持 cart 原语（4维动作），
    但 mode=full 时不需要力控插装，用 pose_move_to 直接放到 goal。
    """
    name = "transfer_pose_cart"
    flavor = "pose_cart"

    def can_achieve(self, cond, entry) -> bool:
        return isinstance(cond, BodyNearSite) and cond.flavor == "pose_cart"

    def make_steps(self, ctx, cond: BodyNearSite) -> List[Dict[str, Any]]:
        env, cmn = ctx.env, ctx.cmn
        goal = np.asarray(env.get_site_pos(cond.site), float)
        steps = _grasp_chain(ctx, cond.body, pose=True)
        # pose_place 是闭环放置：内部检查 body 是否到达 goal，自动补偿握持偏移
        steps += [
            {"action": "pose_place",
             "params": {"goal": [float(goal[0]), float(goal[1]),
                                 float(goal[2])], "body": cond.body,
                        "tol": cond.tol, **cmn}},
        ]
        # 空中递送点：松爪物体会坠落，保持夹持直到终态核验通过
        if not _needs_keep_hold(env, cond, cond.body):
            steps.append({"action": "pose_open_gripper", "params": {**cmn}})
        return steps


class IKLiberoTransferMethod(Method):
    """LIBERO IK 笛卡尔调度方法（不依赖任何 demo）。

    从 BDDL goal 结构化目标出发，在 make_steps 当下观察世界：
      抓取点 = ctx.grasp_pt（由 runner 候选注入；Phase C 起 source=graspnet）
      放置点 = 支撑物顶面（On）或容器 region site（In）
    现场绑定一条通用 IK 末端闭环链（与 robopal cart 链同构，跨 env 共享）：
      ik_servo(above) → ik_servo(descend) → close → lift → carry → place → open。
    差异参数（hover/k/reach_tol/lift_height/release_offset/timeout_scale）走
    per-env YAML（$cfg），反思学习写回 YAML，不写死代码 if-else。
    """
    name = "ik_libero_transfer"
    includes_home = True
    flavor = "libero"

    def can_achieve(self, cond, entry) -> bool:
        if entry.get("mode") != "libero":
            return False
        if isinstance(cond, LiberoSubGoal):
            return cond.skind == "place"
        g = entry.get("libero_goal")
        return (isinstance(cond, LiberoGoal) and bool(g)
                and g.get("kind") == "place")

    def make_steps(self, ctx: PlanContext, cond) -> List[Dict[str, Any]]:
        from .libero_tasks import resolve_site_name

        env, entry, cmn = ctx.env, ctx.entry, ctx.cmn
        if isinstance(cond, LiberoSubGoal):
            obj, target_name, predicate = cond.object, cond.target, cond.predicate
        else:
            g = entry["libero_goal"]
            obj, target_name, predicate = g["object"], g["target"], g["predicate"]

        # 1) 抓取点：由 runner 候选注入（ctx.grasp_pt），不再在此几何兜底。
        #    Phase C 起 libero 候选 source=graspnet（grasp_pose skill）。
        grasp = ctx.grasp_pt
        # 2) 放置目标：BDDL 物体（plate 等）取支撑顶面；region 取 site 真值
        if target_name in env.object_names:
            target = [float(x) for x in env.support_point(target_name)]
        else:
            site_name = resolve_site_name(env.mj_model, target_name)
            if site_name is None:
                raise RuntimeError(
                    f"LIBERO 放置目标无法解析: {target_name}（非物体也非已知 region）")
            target = [float(x) for x in np.asarray(env.get_site_pos(site_name), float)]
        # 偏心抓取补偿：容器偏心插指时碗中心 ≠ TCP（TCP 偏移 grasp-bowl_center
        # 方向）。carry/place 的 target 要加这个偏移，让碗落在 plate 中心而非
        # 边缘。实心物体 grasp≈bowl_center，偏移≈0 不影响。
        # 必须用夹取前碗位（ctx.body0_pos）：resume 续跑时碗已在夹爪中被抬走，
        # 用现场位置会把静止偏移算成移动量，place 目标被带偏十几厘米。
        bowl_center = (ctx.body0_pos
                       if ctx.body0_pos is not None
                       else np.asarray(env.get_body_pos(obj), float))
        grasp_offset = np.array(grasp[:2], float) - bowl_center[:2]
        target = [target[0] + float(grasp_offset[0]),
                  target[1] + float(grasp_offset[1]),
                  target[2]]

        cfg = ctx.cfg
        hover = float(cfg.get("hover", 0.08))
        k = float(cfg.get("k", 5.0))
        vcap = float(cfg.get("vcap", 1.0))
        reach_tol = float(cfg.get("reach_tol", 0.006))
        release_offset = float(cfg.get("release_offset", 0.02))
        # lift：libero 目标是 BDDL On（非固定 z），cfg.lift_height 是 delta（抬升量），
        # LiftSkill 用绝对 z 阈值（robopal 同款，匹配 BodyLifted(0.52) 绝对目标），
        # 故在此把 delta 转绝对：body_z0 + lift_height
        lift_delta = float(cfg.get("lift_height", 0.10))
        lift_h = float(env.get_body_pos(obj)[2]) + lift_delta
        ts = int(120 * ctx.timeout_scale)
        # 进容器(In)：实时读 region site 的真实判定盒半尺寸（物理事实，等价于
        # LIBERO SiteObject.in_box 的轴对齐半边长），传给 PlaceSkill 做盒级对中/
        # 落料判定。region 名解析见上方 target 分支；对物体面 On 目标为 None。
        contain_params: Dict[str, Any] = {}
        if predicate == "In" and target_name not in env.object_names:
            try:
                csite = resolve_site_name(env.mj_model, target_name)
                sid = int(env.mj_model.site_name2id(csite))
                sz = np.asarray(env.mj_model.site_size[sid], float)
                xmat = np.asarray(env.mj_data.site_xmat[sid], float).reshape(3, 3)
                half = np.abs(xmat @ sz)  # 与 in_box 的 this_mat @ size 一致
                contain_params = {"contain_site": csite,
                                  "contain_half": [float(x) for x in half]}
                # 转运高度下限：夹持物体的底部必须全程越过容器口沿（rim）+
                # 余量，否则 carry 平移时物体底部刮沿被撞脱爪（milk/basket
                # 实测：milk 底 0.129 < 沿 0.137，carry 末段脱爪）。TCP 悬深
                # 取实测典型 0.06，再加 1.5cm 余量。
                # 上限 carry_z_cap_m：转运高度必须高于容器口沿+物体半高
                # （微波炉 rim1.03+half0.05+悬深 0.06≈1.15，0.95 会撞炉前壁
                # move_timeout），但不得超 OSC 可达极限（1.29 实测跑飞，
                # 1.18 安全）。phys 通道 per-env 可调。
                from ..skills.primitives import _body_half_h
                from ..skills.physics_profile import phys_get
                rim = (float(np.asarray(env.get_site_pos(csite), float)[2])
                       + float(half[2]))
                _need = rim + 0.015 + _body_half_h(env, obj) + 0.06
                lift_h = min(max(lift_h, _need),
                             float(phys_get(env, "carry_z_cap_m", 0.95)))
            except Exception:
                contain_params = {}
        return [
            {"action": "ik_servo",
             "params": {"point": grasp, "mode": "above", "hover": hover,
                        "k": k, "vcap": vcap, "reach_tol": reach_tol,
                        "timeout": ts, **cmn}},
            {"action": "ik_servo",
             "params": {"point": grasp, "mode": "descend",
                        "stop_above": float(cfg.get("stop_above", -0.01)),
                        "contact_stop_band": float(cfg.get("contact_stop_band", 0.05)),
                        "reach_tol": reach_tol, "k": k, "vcap": vcap, "body": obj,
                        "timeout": ts, **cmn}},
            {"action": "close_gripper", "params": {**cmn}},
            {"action": "lift",
             "params": {"height": lift_h, "body": obj, "timeout": ts,
                        "vcap": vcap, **cmn}},
            {"action": "carry",
             # carry_vcap：搬运段独立限速（默认回落 cfg.vcap），低速平移减小
             # 惯性力，防止 40N 夹持的光滑罐体在爪内滑移（10_07 实测 3cm 滑脱）
             "params": {"target": target, "hover": hover + 0.02, "k": k,
                        "vcap": float(cfg.get("carry_vcap", vcap)),
                        # 预算随跨距自适应：实测每步有效位移 ≈ vcap/80
                        # （P 收敛+避障+末段容差损耗），固定 160 步在跨距
                        # >20cm 时必 move_timeout（spatial:09 实证）。1.5× 余量。
                        "timeout": _carry_timeout(target, grasp,
                                                  float(cfg.get("carry_vcap", vcap)),
                                                  ctx.timeout_scale), **cmn}},
            {"action": "place",
             "params": {"goal": target, "body": obj,
                        "release_offset": release_offset, "vcap": vcap,
                        "timeout": int(cfg.get("place_timeout", 200)
                                       * ctx.timeout_scale),
                        "k": float(cfg.get("place_k", 2.5)),
                        # 仅进容器(In)启用“高空对中 + 卡沿松爪落料”；On 宽面走旧路径
                        "container": bool(predicate == "In"),
                        **contain_params, **cmn}},
            {"action": "open_gripper", "params": {**cmn}},
        ]


class ReplayDemoMethod(Method):
    """LIBERO 演示回放方法：把任务 demo_0 的动作序列逐帧展开为 libero_action 链。

    env 适配器 reset 时已回到 demo 的录制初始状态（states[0]），因此回放是
    确定性的；链不含 home（初始构型本就由 reset 保证）。终态由 LiberoGoal
    的 BDDL 谓词核验 —— 回放本身不"假装成功"，失败同样走自适应/换方法路径。
    """
    name = "replay_demo"
    includes_home = True
    flavor = "libero"

    def can_achieve(self, cond, entry) -> bool:
        return isinstance(cond, LiberoGoal) and entry.get("mode") == "libero"

    def make_steps(self, ctx: PlanContext, cond: LiberoGoal) -> List[Dict[str, Any]]:
        from ..envs.libero_adapter import demo_actions
        suite = ctx.entry["libero_suite"]
        idx = int(ctx.entry["libero_task_idx"])
        acts = demo_actions(suite, idx)
        return [{"action": "libero_action",
                 "params": {"action": [float(x) for x in np.asarray(a, float)],
                            **ctx.cmn}}
                for a in acts]


class LiberoArticulateMethod(Method):
    """LIBERO 关节式谓词通用方法（Open/Close/Turnon/Turnoff）。

    home → 通用 articulate 原语把对应夹具关节驱动到语义目标 qpos；
    关节/目标在 make_steps 当下从 env.articulation_info 解析（region/夹具名
    全部来自 BDDL 子目标，无任务名硬编码）。
    """
    name = "libero_articulate"
    includes_home = True
    flavor = "libero"

    _TARGET_KEY = {"Open": "open_qpos", "Close": "close_qpos",
                   "Turnon": "turnon_qpos", "Turnoff": "turnoff_qpos"}

    def can_achieve(self, cond, entry) -> bool:
        return (entry.get("mode") == "libero"
                and isinstance(cond, LiberoSubGoal)
                and cond.skind in ("articulate", "toggle"))

    def make_steps(self, ctx: PlanContext, cond: LiberoSubGoal) -> List[Dict[str, Any]]:
        env, cmn = ctx.env, ctx.cmn
        info = env.articulation_info(cond.target)
        key = self._TARGET_KEY.get(cond.predicate)
        lo, hi = info["range"]
        # 语义目标 qpos（close_qpos/open_qpos/turnon_qpos/turnoff_qpos）= 
        # default_*_ranges 的中点，与 LIBERO is_close/is_open 判定口径一致；
        # 无语义目标时 Close/Turnoff=lo，Open/Turnon=中点
        target = info.get(key) if key else None
        if target is None:
            if cond.predicate in ("Close", "Turnoff"):
                target = lo
            else:
                target = 0.5 * (lo + hi)
        span = abs(info["range"][1] - info["range"][0])
        # 滑动/短行程抽屉容差 3cm；旋钮长行程（~2rad）放 0.35rad
        tol = 0.03 if span < 0.5 else 0.35
        return [
            {"action": "articulate",
             "params": {"joint_name": info["joint"],
                        "target_qpos": float(target), "tol": tol, **cmn}},
        ]


# 兼容旧 import 路径（runner_dynamic 用 METHOD_REGISTRY）
BUILTIN_METHODS = [OpenDrawerMethod(), GraspLiftMethod(), TransferCartMethod(),
                   TransferPoseMethod(), TransferPoseCartMethod(),
                   IKLiberoTransferMethod(), LiberoArticulateMethod(),
                   ReplayDemoMethod()]
METHOD_REGISTRY = BUILTIN_METHODS


def select_method(cond: GoalCond, entry: Dict[str, Any]) -> Method:
    """从内置方法库中选择能达成该条件的方法（数据驱动匹配，无任务名分支）。

    注意：这是旧版静态选择；生产环境应用 ChainRegistry.select（支持 YAML 覆盖）。
    """
    for m in METHOD_REGISTRY:
        if m.can_achieve(cond, entry):
            return m
    raise ValueError(f"no method registered for objective: {cond.describe()}")


__all__ = ["PlanContext", "Method", "OpenDrawerMethod", "GraspLiftMethod",
           "TransferCartMethod", "TransferPoseMethod", "BUILTIN_METHODS",
           "METHOD_REGISTRY", "select_method", "entry_get", "_grasp_chain"]
