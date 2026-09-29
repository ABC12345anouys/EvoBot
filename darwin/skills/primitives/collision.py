"""场景级碰撞监测 + 安全走廊路径规划（skill 层，直接跑在 mujoco env 上）。

定位（与 motion.py 互补）：
- motion.py 的 CollisionCheckSkill/PathPlanSkill 走 mplib backend（需 profile 注入），
  做关节空间 RRT-Connect 规划；env 无 backend 时返回 planner_unavailable。
- 本模块无需 backend：计算机械臂与障碍的最小距离，并把"升高→水平平移→下降"
  三段式安全走廊下沉进位移原语，上层脚本无需再写旁路逻辑。

两种监测模式（按场景 geom 类型自动选择）：
- geom 模式（robopal）：机械臂与障碍都有图元碰撞 geom，两两
  mj_geomDistance，精确。
- skeleton 模式（LIBERO/robosuite）：双方碰撞 geom 全是 mesh
  （mj_geomDistance 返回伪 0，图元过滤下连臂 geom 都收集不到），改用
  "机械臂骨架线段 vs 障碍世界 AABB" 的向量化距离：
  · 骨架 = grip_site 父链相邻 body 原点连线（近胶囊半径近似）；
  · 障碍 = 每个碰撞 geom 的世界 AABB（AABB 包含 mesh，偏保守）；
  · 每步随 data 位姿刷新，物体被推动后 AABB 跟着移动。

运行模式（phys key collision_mode）：
- abort（默认）：距离 < abort(4mm) 即中止，reason="collision_risk"。
- warn：只记录 min_clearance 不中止——新场景灰度上线用，先看误报率再切 abort。

约定：
- 障碍 body 名由 env.obstacle_bodies 注入；LIBERO 由 auto_obstacles 自动
  生成（桌面 + 夹具 + 非任务物体），动态排除持有物与放置目标容器。
- 所有位移原语的结果统一附加 min_clearance / coll_pair 字段。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import step_env, p_action, servo_step
from ..physics_profile import phys_get

SAFE_Z = 0.62          # 安全走廊高度（柜子顶 0.55 上方）
ABORT_DIST = 0.004     # 中止阈值：距离 < 4mm 视为碰撞风险
SAFE_VCAP = 0.05       # 速度上限：单步位移 7mm，保证碰撞监测"先见近距再中止"

_LINK_RADIUS = 0.024   # 骨架线段默认胶囊半径（panda 连杆宽约 4-5cm）
_HAND_RADIUS = 0.035   # 手掌段（wrist->eef）半径
_SEG_SAMPLES = 9       # 线段-盒距离采样点数（6cm 段误差 <=0.3mm，可忽略）


def _s1(x) -> int:
    """包装层标量安全取值（geom.type/bodyid 等可能是 len-1 数组）。"""
    a = np.asarray(x).reshape(-1)
    return int(a[0]) if a.size else 0


class CollisionMonitor:
    """机械臂 vs 障碍最小距离监测器（geom / skeleton 双模式）。"""
    def __init__(self, env, actor: str = "agent0", obstacle_bodies=None,
                 safe_z: float = SAFE_Z, abort: float = ABORT_DIST,
                 distmax: float = 2.0, grip_site: str = None):
        import mujoco
        self.mj = mujoco
        self.env = env
        self.safe_z = float(safe_z)
        self.abort = float(phys_get(env, "collision_abort_m", abort))
        self.distmax = float(distmax)
        # 模式来源优先级：env.collision_mode（runner 注入，Profile 只存
        # PHYS_SPEC 键会丢掉 str 配置）> physics profile > 默认 abort
        _mode = (getattr(env, "collision_mode", None)
                 or phys_get(env, "collision_mode", "abort"))
        self.warn_only = str(_mode).lower() == "warn"
        bodies = tuple(obstacle_bodies if obstacle_bodies is not None
                       else (getattr(env, "obstacle_bodies", None) or ()))
        self.obstacles = bodies
        prefix = actor[-1] + "_" if actor[-1].isdigit() else ""
        # mj_geomDistance 对 mesh geom 会返回伪 0.0（非物理），只保留图元类型
        _OK = {int(mujoco.mjtGeom.mjGEOM_SPHERE), int(mujoco.mjtGeom.mjGEOM_CAPSULE),
               int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_BOX),
               int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)}
        self._arm: List[int] = []
        self._obs: List[int] = []
        m = env.mj_model
        for i in range(m.ngeom):
            g = m.geom(i)
            gtype = _s1(g.type)
            if gtype not in _OK:
                continue
            bn = m.body(_s1(g.bodyid)).name
            if bn in bodies:
                self._obs.append(i)
            elif prefix and bn.startswith(prefix) and "finger" not in bn \
                    and _s1(g.contype) > 0:
                self._arm.append(i)
        if self._arm and self._obs:
            self.mode = "geom"
            self._skeleton_ok = True
            return
        # ---- skeleton 模式（LIBERO：碰撞 geom 全 mesh）----
        self.mode = "skeleton"
        gs = grip_site or "gripper0_grip_site"
        self._chain: List[int] = []
        try:
            bid = _s1(m.site_bodyid[_s1(m.site_name2id(gs))])
            while bid != 0:
                nm = m.body(bid).name
                if "base" in nm or "mount" in nm or "torso" in nm:
                    break   # 机器人根/固定座不算臂体（mount 与桌面恒接触）
                self._chain.append(bid)
                bid = _s1(m.body(bid).parentid)
            self._chain.reverse()   # base->eef，相邻原点连线
        except Exception:
            self._chain = []
        # 障碍 geom（含 mesh）：任何碰撞 geom 都转世界 AABB
        obs_set = set(bodies)
        self._obs_geoms: List[int] = []
        self._aabb: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(m.ngeom):
            g = m.geom(i)
            if _s1(g.contype) == 0 and _s1(g.conaffinity) == 0:
                continue   # 纯视觉 geom
            if m.body(_s1(g.bodyid)).name not in obs_set:
                continue
            a = np.asarray(m.geom_aabb[i], float).reshape(-1)
            if a.size < 6:
                continue
            self._obs_geoms.append(i)
            self._aabb.append((a[:3].copy(), a[3:6].copy()))
        self._skeleton_ok = bool(self._chain and self._obs_geoms)

    @property
    def enabled(self) -> bool:
        if self.mode == "geom":
            return bool(self._arm and self._obs)
        return self._skeleton_ok

    def violated(self, d: float) -> bool:
        """d 是否达到中止条件（warn 模式只记录 min_clearance 不中止）。"""
        return (not self.warn_only) and d < self.abort

    # ---- geom 模式：图元对精确距离 ----
    def _clearance_geom(self) -> tuple:
        m, d = self.env.mj_model, self.env.mj_data
        dmin, pair = 1.0, ""
        for gi in self._arm:
            for gj in self._obs:
                try:
                    v = self.mj.mj_geomDistance(m, d, gi, gj, self.distmax, None)
                except Exception:
                    v = 1.0
                if v < dmin:
                    dmin = float(v)
                    pair = (f"{m.body(_s1(m.geom(gi).bodyid)).name}"
                            f"/{m.body(_s1(m.geom(gj).bodyid)).name}")
        return dmin, pair

    # ---- skeleton 模式：骨架线段 vs 障碍世界 AABB（向量化）----
    def _clearance_skeleton(self) -> tuple:
        if not self._skeleton_ok:
            return 1.0, ""
        data = self.env.mj_data
        # 世界 AABB：center_w = xpos + R@aabb_c，half_w = |R|@aabb_h（保守包络）
        n = len(self._obs_geoms)
        C = np.empty((n, 3))
        H = np.empty((n, 3))
        for k, gi in enumerate(self._obs_geoms):
            R = np.asarray(data.geom_xmat[gi], float).reshape(3, 3)
            c0, h0 = self._aabb[k]
            C[k] = np.asarray(data.geom_xpos[gi], float) + R @ c0
            H[k] = np.abs(R) @ h0
        pts = np.asarray(data.body_xpos, float)[self._chain]   # (K,3)
        m = self.env.mj_model
        dmin, pair = 1.0, ""
        for si in range(len(pts) - 1):
            p0 = pts[si]
            ts = np.linspace(0.0, 1.0, _SEG_SAMPLES)[:, None]
            P = p0[None, :] + ts * (pts[si + 1] - p0)[None, :]        # (S,3)
            dd = np.linalg.norm(
                np.maximum(np.abs(P[:, None, :] - C[None, :, :])
                           - H[None, :, :], 0.0), axis=2)             # (S,N)
            k = int(np.argmin(dd))
            s_i, g_i = divmod(k, dd.shape[1])
            r = _HAND_RADIUS if si == len(pts) - 2 else _LINK_RADIUS
            v = float(dd[s_i, g_i]) - r
            if v < dmin:
                dmin = v
                ob = m.body(_s1(m.geom(self._obs_geoms[g_i]).bodyid)).name
                pair = f"{m.body(self._chain[si]).name}seg/{ob}"
        return dmin, pair

    def clearance(self) -> tuple:
        """(最小距离, 碰撞对名)；无障碍对时 (1.0, '')。"""
        if not self.enabled:
            return 1.0, ""
        if self.mode == "geom":
            return self._clearance_geom()
        return self._clearance_skeleton()


def get_monitor(env, actor: str = "agent0",
                grip_site: str = None) -> Optional[CollisionMonitor]:
    """从 env 构建监测器；env 未配置 obstacle_bodies 时返回 None（避障关闭）。"""
    mon = CollisionMonitor(env, actor=actor, grip_site=grip_site)
    return mon if mon.enabled else None


def auto_obstacles(env, entry: Dict[str, Any], objectives=()) -> List[str]:
    """LIBERO 自动障碍清单（body 名列表）：桌面 + 夹具 + 非任务物体。

    排除（防误杀）：
    - world 与机器人链（grip_site 父链、robot/mount 命名——mount 与桌面恒
      接触，算障碍会永远 collision_risk）；
    - 所有 goal 谓词的 object（多物体任务的第二物是后续抓取目标，绝不能
      当障碍）；
    - In 目标容器 body 及其名字前缀子树（basket 算障碍会让 place 一启动
      就误报碰撞）。
    - 无碰撞 geom 的纯视觉 body。
    """
    m = env.mj_model
    try:
        names = [m.body(i).name for i in range(int(m.nbody))]
    except Exception:
        return []
    excl = {"world"}
    gs = entry.get("grip_site") or "gripper0_grip_site"
    try:
        b = _s1(m.site_bodyid[_s1(m.site_name2id(gs))])
        while b != 0:
            excl.add(names[b])
            b = _s1(m.body(b).parentid)
    except Exception:
        pass
    gp = gs.split("_")[0]   # "gripper0_grip_site" -> "gripper0"
    for nm in names:
        if ("mount" in nm or nm.startswith("robot") or nm.startswith(gp)
                or "finger" in nm):
            excl.add(nm)   # 夹爪手指是 grip 链的子树，不是障碍

    def _body_of(obj_name: str) -> Optional[str]:
        for cand in (obj_name, f"{obj_name}_main"):
            try:
                return names[_s1(m.body_name2id(cand))]
            except Exception:
                continue
        return None

    def _add_subtree(root: str) -> None:
        for nm in names:
            if nm == root or nm.startswith(root + "_"):
                excl.add(nm)

    for c in objectives or ():
        obj = getattr(c, "object", None)
        if obj:
            bb = _body_of(obj)
            if bb:
                excl.add(bb)
        pred = getattr(c, "predicate", None)
        tgt = getattr(c, "target", None)
        if pred == "In" and tgt and tgt not in (getattr(env, "object_names",
                                                         None) or []):
            try:
                from ..agents.libero_tasks import resolve_site_name
                sn = resolve_site_name(m, tgt)
                if sn:
                    _add_subtree(names[_s1(m.site_bodyid[
                        _s1(m.site_name2id(sn))])])
            except Exception:
                pass
    gobj = (entry.get("libero_goal") or {}).get("object")
    if gobj:
        bb = _body_of(gobj)
        if bb:
            excl.add(bb)
    out: List[str] = []
    for i in range(m.ngeom):
        g = m.geom(i)
        if _s1(g.contype) == 0 and _s1(g.conaffinity) == 0:
            continue
        bn = names[_s1(g.bodyid)]
        if bn not in excl and bn not in out:
            out.append(bn)
    return out


def plan_corridor(cur, tgt, mon: Optional[CollisionMonitor]) -> List[np.ndarray]:
    """三段式安全走廊：升高到 safe_z -> 水平平移到目标 xy -> 下降到目标。

    - 未配置障碍(mon=None) -> 直线 [tgt]，行为同旧版
    - 起终点都在 safe_z 上方 -> 直线
    - waypoint 与当前点位重合(<2cm)时自动剔除
    """
    cur = np.asarray(cur, float).reshape(3)
    tgt = np.asarray(tgt, float).reshape(3)
    if mon is None:
        return [tgt]
    sz = mon.safe_z
    if tgt[2] >= sz - 0.005 and cur[2] >= sz - 0.005:
        return [tgt]
    wps: List[np.ndarray] = []
    if abs(cur[2] - sz) > 0.02:
        wps.append(np.array([cur[0], cur[1], sz]))
    if tgt[2] < sz - 0.005:
        wps.append(np.array([tgt[0], tgt[1], sz]))
    wps.append(tgt)
    out = [w for w in wps if np.linalg.norm(w - cur) > 0.02]
    return out or [tgt]


def follow_waypoints(env, actor: str, site: str, wps: List[np.ndarray],
                     mon: Optional[CollisionMonitor], *, gripper: float = 0.0,
                     k: float = 2.5, seg_tol: float = 0.02, timeout: int = 110,
                     vcap: float = SAFE_VCAP, final_check=None) -> Dict[str, Any]:
    """沿 waypoint 序列做闭环 P 控制，逐步监测碰撞。

    final_check(end)->bool：最后一段的成功判据（各原语自定义）；
    为 None 时用 norm(end-wps[-1])<seg_tol。
    返回 {success, reason, steps, min_clearance, coll_pair, end}。
    """
    total, clear_min, pair_min = 0, 1.0, ""
    last = np.asarray(wps[-1], float)
    stall_win, stall_ref = 0, None  # 连续停滞检测：撞墙磨停时 30 步内 fail-fast
    for wi, wp in enumerate(wps):
        wp = np.asarray(wp, float)
        is_last = wi == len(wps) - 1
        budget = timeout if is_last else max(40, timeout // 2)
        stall_ref, stall_win = None, 0  # 每段独立计停滞
        for _ in range(budget):
            total += 1
            end = np.asarray(env.get_site_pos(site), float)
            # 先判到达（final_check/waypoint 容差），再判停滞：TCP 楔在
            # 障碍物角上但 xy 已到位时（spatial:4 实证：碗紧贴柜，夹爪被
            # 柜沿顶住 21mm 降不下去，xy 已在容差内），旧代码会判 above
            # 成功交给 descend 继续；停滞检测必须先让位于到达判据，
            # 否则把"已到位"误报成 stall。
            if is_last:
                ok = final_check(end) if final_check is not None \
                    else bool(np.linalg.norm(end - last) < seg_tol)
                if ok:
                    return {"success": True, "reason": "ok", "steps": total,
                            "min_clearance": clear_min, "coll_pair": pair_min,
                            "end": end.tolist()}
            elif np.linalg.norm(end - wp) < seg_tol:
                break
            if stall_ref is None:
                stall_ref = end.copy()
            elif np.linalg.norm(end - stall_ref) < 0.0005:
                stall_win += 1
                if stall_win >= 30:
                    return {"success": False,
                            "reason": "move_timeout_stall",
                            "steps": total, "min_clearance": clear_min,
                            "coll_pair": pair_min, "end": end.tolist()}
            else:
                stall_ref = end.copy()
                stall_win = 0
            # 统一走 servo_step（跨 env：libero 走 OSC，robopal 走 CARTIK，
            # 夹爪语义 +1=合/-1=开/0=保持，由 servo_step 内部翻成各 env 物理值）
            servo_step(env, site, wp, gripper=gripper, k=k, vcap=vcap, actor=actor)
            if mon is not None:
                d, pr = mon.clearance()
                if d < clear_min:
                    clear_min, pair_min = d, pr
                if mon.violated(d):
                    return {"success": False, "reason": "collision_risk",
                            "coll_pair": pr, "min_clearance": d, "steps": total,
                            "end": end.tolist()}
    end = np.asarray(env.get_site_pos(site), float)
    return {"success": False, "reason": "move_timeout", "steps": total,
            "min_clearance": clear_min, "coll_pair": pair_min, "end": end.tolist()}
