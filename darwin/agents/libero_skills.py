"""LIBERO 通用操作技能库：模型求解选点+统计伺服。
本文件里**没有任务阈值**。所有数字分为四类，来源均可追溯：
1. 模型事实（每台机器人/环境自带，直接读取）：
   关节限位、site/body/geom 几何、夹爪开合行程；
2. 在线测量量（每次动作时从观测序列估计）：
   末端噪声 σ、接触力基线、关节跟踪误差、初始进展率；
3. 算法精度（求解器自身的收敛精度，非任务参数）：
   IK_TOL、DLS 阻尼、DE 迭代数；
4. 统计约定（推断的置信水平，不随任务变化）：
   3σ 包络、t 检验门限、回归窗口样本数。
"去哪抓"由约束优化在实时几何上现解（变量=插指偏移与深度，
约束=IK 可达/射线净空/走廊无碰撞），"是否到位"由约束复验与
噪声统计判定，"失败原因"由物理证据分类后交 planner。不存在
per-task 的容差、超时和候选偏移表。

技能（planner 按谓词种类映射）：
    grasp / place_at / articulate / toggle / move_to
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import functools
import json
import math
import os
import time
from collections import deque
from pathlib import Path

SNAP_DIR = Path(os.environ.get(
    "DARWIN_SNAPSHOT_DIR",
    Path.cwd() / "logs" / "debug_snapshots"))
# 快照开关：DARWIN_SNAPSHOT=1 或 LIBERO_DEBUG 任一开启。关闭时零开销
# （环形缓冲不写入）。
SNAP_ON = bool(os.environ.get("DARWIN_SNAPSHOT")) or \
    bool(os.environ.get("LIBERO_DEBUG"))
_SNAP_RING: "deque[dict]" = deque(maxlen=32)
# fail_ 现场保险队列（无上限，不被环形缓冲逐出，snap_dump 时落盘）
_SNAP_FAILS: "list[dict]" = []
# 最近一次快照关联的目标物（快照元数据里的 obj），供 attempt 超时/
# 异常路径兜底落盘——那两条路径不经 grasp 正常的 snap_dump。
_SNAP_LAST_OBJ: "Optional[str]" = None


def _snap_reset() -> None:
    _SNAP_RING.clear()
    _SNAP_FAILS.clear()


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def _snap(adapter, tag: str, **meta) -> None:
    """捕获当前仿真完整状态进内存环形缓冲（µs 级，纯数组拷贝）。

    快照 = 可复现的最小现场：qpos/qvel/act/time/warmstart + 现场元
    数据（stage/obj/site/R/失败原因）。配合 snap_dump（技能失败时落盘）
    与 replay 脚本，可在秒级从任意流水线环节重放，跳过 gsnet 推理、
    RRT 规划和长伺服——失败现场不再需要整轮重跑。

    fail_ 前缀的快照（各候选的失败现场）额外进 _SNAP_FAILS 保险
    队列：环形缓冲只有 32 槽，长轮询（几十次候选）会把早期失败现场
    挤出缓冲导致落盘缺失——失败现场是排障的核心证据，一条都不能丢。
    """
    if not SNAP_ON:
        return
    import mujoco
    global _SNAP_LAST_OBJ
    if meta.get("obj"):
        _SNAP_LAST_OBJ = str(meta["obj"])
    m, d = _native_md(adapter)
    rec = ({
        "qpos": np.asarray(d.qpos, float).copy(),
        "qvel": np.asarray(d.qvel, float).copy(),
        "act": np.asarray(d.act, float).copy(),
        "time": float(d.time),
        "warmstart": np.asarray(d.qacc_warmstart, float).copy(),
        "tag": tag,
        "meta": _jsonable(meta),
    })
    _SNAP_RING.append(rec)
    if tag.startswith("fail_"):
        _SNAP_FAILS.append(rec)


def _write_snap(out: Path, i: int, s: dict) -> None:
    np.savez_compressed(
        out / f"{i:02d}_{s['tag']}.npz",
        qpos=s["qpos"], qvel=s["qvel"], act=s["act"],
        time=s["time"], warmstart=s["warmstart"],
        meta_json=np.frombuffer(
            json.dumps({"tag": s["tag"], "meta": s["meta"]}).encode(),
            dtype=np.uint8))


def snap_dump(adapter, obj: str, reason: str) -> Optional[Path]:
    """把全部快照落盘（仅失败时调用），返回目录。

    落盘 = 环形缓冲（最近 32 环，完整时间线）+ _SNAP_FAILS 保险队列
    （全部 fail_ 现场，不被环形缓冲逐出）。两者文件名连续编号，
    failbox/ 子目录单独存放保险队列，避免与时间线重名覆盖。
    """
    if not SNAP_ON or (not _SNAP_RING and not _SNAP_FAILS):
        return None
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = SNAP_DIR / f"{ts}_{obj}"
    out.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(_SNAP_RING):
        _write_snap(out, i, s)
    if _SNAP_FAILS:
        fb = out / "failbox"
        fb.mkdir(exist_ok=True)
        for i, s in enumerate(_SNAP_FAILS):
            _write_snap(fb, i, s)
    (out / "FAIL.txt").write_text(f"obj={obj}\nreason={reason}\n")
    print(f"[snap] ring={len(_SNAP_RING)} failbox={len(_SNAP_FAILS)} → {out}",
          flush=True)
    _snap_reset()
    return out


def snap_dump_last(reason: str) -> Optional[Path]:
    """超时/异常路径的兜底落盘：不经过 grasp 正常的 snap_dump
    （SIGALRM 在伺服循环中途打断，候选枚举未完），但环形缓冲与
    failbox 里的现场同样珍贵——按最近快照的 obj 落盘，供 replay。"""
    return snap_dump(None, _SNAP_LAST_OBJ or "unknown", reason)


def snap_restore(adapter, path) -> dict:
    """从快照文件恢复仿真状态，返回其中的元数据（site/R/obj 等）。"""
    import mujoco
    z = np.load(path, allow_pickle=False)
    m, d = _native_md(adapter)
    d.qpos[:] = z["qpos"]
    d.qvel[:] = z["qvel"]
    d.act[:] = z["act"]
    d.time = float(z["time"])
    d.qacc_warmstart[:] = z["warmstart"]
    mujoco.mj_forward(m, d)
    return json.loads(bytes(z["meta_json"].tobytes()).decode())


def _stageln(skill: str, n: int, stage: str, ok: bool,
             reason: str = "", **kw) -> None:
    """单行结构化阶段日志（常开，grep 友好）：

        [grasp] try=2 pre_approach ok d=0.031
        [grasp] try=2 lift FAIL reason=闭爪后物体未随抬升 lifted=0.001

    每个流水线阶段恰好一行，取代散点 print 式的排查。
    """
    parts = []
    for k, v in kw.items():
        v = _jsonable(v)
        if isinstance(v, list):
            v = np.round(np.asarray(v, float), 3).tolist()
        parts.append(f"{k}={v}")
    mark = "ok" if ok else "FAIL"
    tail = f" reason={reason}" if reason else ""
    args = " " + " ".join(parts) if parts else ""
    print(f"[{skill}] try={n} {stage} {mark}{tail}{args}", flush=True)


def _timed(fn):
    """LIBERO_DEBUG 下的阶段计时插桩（秒 + 结果摘要），定位墙钟去向。"""
    @functools.wraps(fn)
    def w(*a, **k):
        if not os.environ.get("LIBERO_DEBUG"):
            return fn(*a, **k)
        t0 = time.time()
        try:
            r = fn(*a, **k)
        except Exception as e:
            print(f"[dbg] {fn.__name__} EXC {time.time() - t0:.1f}s {e}",
                  flush=True)
            raise
        tag = ""
        if isinstance(r, dict):
            tag = str(r.get("reason") or ("ok" if r.get("success") else "?"))
        elif isinstance(r, list):
            tag = f"n={len(r)}"
        elif r is None:
            tag = "ok" if fn.__name__ in ("serve", "execute_arm_path") else "none"
        print(f"[dbg] {fn.__name__} {time.time() - t0:.1f}s {tag}",
              flush=True)
        return r
    return w

# ==================== 算法精度 & 统计约定（非任务阈值） ====================
GRIP_SITE = "gripper0_grip_site"

IK_TOL = 0.003              # IK/几何"零距离"精度（m）：求解器精度，
                            # 兼作接触/落座的几何零判定
# 姿态 IK 精度：名义手部特征尺寸 0.06m（实测指尖杠杆 a_open/b_depth
# ≈0.031m、整手包络 ≈0.125m，0.05rad 介于两推导值之间，经 30 任务
# 回归验证；进度量中的旋转权重已改用 _tip_lever 实测值）
ROT_TOL = IK_TOL / 0.06
IK_ITERS = 80               # DLS 逆解迭代上限
IK_DAMP = 0.05              # 阻尼最小二乘 λ（奇异鲁棒）
DE_ITERS = 35               # 差分进化迭代上限
DE_POP = 8
ANALYTIC_TRIES = 12         # 分析型求解器单次 grasp 的尝试资源上限
                            # （=方位枚举数 12：窄长物体的末端挤压夹
                            # 开合余量小、打分垫底，8 次预算会在前 8
                            # 方位耗尽、唯二可达方位得不到尝试——object:1
                            # 长条盒实证）

CONF_K = 3.0                # 置信水平：噪声包络/显著性统一 3σ
TSTAT = 2.0                 # 斜率 t 统计门限（≈95%，低于视为无进展）
WIN = 30                    # 回归窗口样本数（t 检验最小样本约定）
SETTLE = 12                 # 命令恒定后稳定观察（步）
STATIONARY_MAX = 3          # 连续无进展窗口上限

VCAP = 1.0                  # OSC action 全量程
K_SERVO = 5.0               # OSC 比例增益（控制器参数，非任务参数）
K_FINE = 2.5

# ----- 关节空间采样运动规划（算法精度参数，非任务阈值） -----
RRT_STEP = 0.15             # 双树扩展步长（关节行程归一化）
RRT_MAX_NODES = 6000        # 双树节点总数上限
SHORTCUT_N = 80             # shortcut 平滑尝试次数
EXEC_DQ = 0.003             # 流式执行点间距（归一化关节行程）；细分
                            # 同时决定跟踪包络（见 _transit_inflation）
IK_RESTARTS = 6             # IK 多热启动次数（DLS 局部停滞的全局化）
PREGRASP_MAX = 600          # 预抓取点几何搜索步预算（步距=IK_TOL，
                            # 各半球方向平分）

# GraspNet 输入帧对齐：网络训练于相机系点云（相机系 +z 看向场景内 =
# 物理接近方向），世界系物理接近向为 -z。F 为世界系→头顶相机系的
# 基变换：x=world x，y=world -y，z=world -z（右手系，det=+1）。
GRASPNET_FRAME = np.diag([1.0, -1.0, -1.0])

# 捏持摩擦系数（保守值，橡胶/光面通用）：rim 捏靠摩擦承竖直载荷，
# 壁面法向须在摩擦锥内——|n·up| ≤ MU_PINCH/√(1+MU_PINCH²) 的壁面
# 才可捏，曲腹（法向倾斜）再深也夹不住。
MU_PINCH = 0.5
WALL_COS_MAX = MU_PINCH / math.sqrt(1.0 + MU_PINCH ** 2)  # ≈0.447
WALL_GATE_SOFT = 0.2        # 垂直度软门宽度（sigmoid 尺度，无量纲）

def _wall_verticality(cloud: np.ndarray, p: np.ndarray, radius: float
                      ) -> float:
    """接触点邻域实测点云的壁面垂直度（1=法向水平=立壁，0=水平面）。

    薄壁两侧的邻域点作主成分分析，最小方差方向即壁法向。样本不足
    时返回 1.0（不过滤）。曲壁容器（碗）的深位落在曲腹上，法向
    倾斜——垂直度是"捏不捏得住"的物理判据，非任务特征。
    """
    nb = cloud[np.linalg.norm(cloud - np.asarray(p, float), axis=1) < radius]
    if len(nb) < 6:
        return 1.0
    _, vecs = np.linalg.eigh(np.cov((nb - np.asarray(p, float)).T))
    n = vecs[:, 0]
    return 1.0 - abs(float(n[2]))


def _ok(**measures) -> Dict[str, Any]:
    return {"success": True, "measures": measures}


def _fail(reason: str, mechanism: str, **measures) -> Dict[str, Any]:
    if os.environ.get("LIBERO_DEBUG"):
        print(f"[dbg] FAIL {mechanism}: {reason} {measures}", flush=True)
    return {"success": False, "reason": reason, "mechanism": mechanism,
            "measures": measures}


def _eef(adapter) -> np.ndarray:
    return np.asarray(adapter.get_site_pos(GRIP_SITE), float)


# ==================== 模型事实访问 ====================

def _native_md(adapter):
    m = getattr(adapter.mj_model, "_model", adapter.mj_model)
    d = getattr(adapter.mj_data, "_data", adapter.mj_data)
    return m, d


def _arm_joints(m) -> List[int]:
    """机械臂单自由度关节：body 必须属于机器人本体（名以 robot 开头）。
    不能用"排除 gripper/finger/hand"的反向逻辑——环境中存在大量
    其他 hinge（橱柜搁板、炉具按钮），反向选择会误纳入。零任务耦合。"""
    import mujoco
    out = []
    for j in range(m.njnt):
        if int(m.jnt_type[j]) not in (2, 3):  # hinge / slide
            continue
        b = int(m.jnt_bodyid[j])
        nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
                 or "").lower()
        if nm.startswith("robot"):
            out.append(j)
    return out


def _finger_joints(m) -> List[int]:
    import mujoco
    out = []
    for j in range(m.njnt):
        nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or "")
        if "finger_joint" in nm:
            out.append(j)
    return out


def _scratch(adapter, grip: Optional[float]):
    """scratch MjData（复制 live qpos；grip: +1 闭/-1 开/None 保持）。
    所有规划探针在此运行，绝不扰动活的仿真。"""
    import mujoco
    m, live = _native_md(adapter)
    sd = mujoco.MjData(m)
    sd.qpos[:] = np.asarray(live.qpos, float)
    if grip is not None:
        for j in _finger_joints(m):
            lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
            # 手指关节镜像对称（一正一负行程）：开合语义端点按 |行程|
            # 判定——|端点| 大 = 张开，|端点| 小 = 闭合（0=指间相触）。
            open_v = hi if abs(hi) >= abs(lo) else lo
            close_v = lo if abs(hi) >= abs(lo) else hi
            sd.qpos[int(m.jnt_qposadr[j])] = open_v if grip < 0 else close_v
    mujoco.mj_forward(m, sd)
    return m, sd


def measure_hand(adapter) -> Dict[str, Any]:
    """从模型现测夹爪几何（finger 全开/全闭构型，site 局部系）：

    a_open/a_closed：两指 pad 内侧面沿开合轴的半间距；b_depth：pad
    接触面中心沿 site +z（接近方向）的坐标（带符号，可正可负——
    由 pad geom 现测，见下）；u_site：site 系下开合轴方向；r/up/down：
    全手水平半径与上下包络；tip_r：指尖（pad）最大半尺寸——插指/
    钩拉所需的"容下一根手指"净空全部以它度量。零经验常数。
    """
    import mujoco
    sid = mujoco.mj_name2id(_native_md(adapter)[0],
                            mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    m, sd = _scratch(adapter, -1.0)
    sp = np.asarray(sd.site_xpos[sid], float)
    sR = np.asarray(sd.site_xmat[sid], float).reshape(3, 3)

    # 指尖 body 集合（先于开口测量使用）
    tip_bodies = set()
    for b in range(m.nbody):
        nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
        if nm.endswith("_tip") and "gripper" in nm:
            tip_bodies.add(b)

    def tip_points(sd_):
        # tip body 的框架原点（作者标注的指尖参考点；geom 平均位置
        # 会被 pad 布局偏移），直接读 body xpos
        pts = [np.asarray(sd_.xpos[b], float).copy() for b in sorted(tip_bodies)]
        if len(pts) >= 2:
            return pts[:2]
        # 兜底：finger body 原点
        for key in ("leftfinger", "rightfinger"):
            cand = [b for b in range(m.nbody)
                    if key in str(mujoco.mj_id2name(
                        m, mujoco.mjtObj.mjOBJ_BODY, b) or "")]
            if cand:
                pts.append(np.asarray(sd_.xpos[cand[0]], float))
        return pts

    tps = tip_points(sd)
    rels = [sR.T @ (np.asarray(p, float) - sp) for p in tps]
    dv = rels[1] - rels[0]
    u = dv / max(float(np.linalg.norm(dv)), 1e-9)
    # 指尖（pad）碰撞 geom 中心沿 site +z 的均值。b_depth 的语义是
    # "捏夹接触面中心相对 site 原点的接近轴坐标"（调用侧统一按
    # 指尖=site+site_R·[0,0,b_depth] 建模），而物理接触只发生在
    # pad geom 上。_tip body 原点是 URDF 标注的参考系，实测它比
    # pad 接触面沿接近轴靠前 ~15mm（tip rel +0.0114 vs pad 中心
    # rel −0.0036）：薄/矮物体上捏夹点被系统性抬到目标顶沿上方，
    # 闭爪捏空（flat 盒实测）。pad 缺省时退回 _tip body 原点。
    pad_z = []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) not in tip_bodies:
            continue
        c_rel = sR.T @ (np.asarray(sd.geom_xpos[g], float) - sp)
        pad_z.append(float(c_rel[2]))
    if pad_z:
        b_depth = float(np.mean(pad_z))
    else:
        b_depth = float(np.mean([r[2] for r in rels]))

    def _inner_half(sd_):
        """两指 pad 内侧面沿开合轴的半间距（真实可抓开口）：pad 中心
        投影距 − pad 沿开合轴的半尺寸，两侧相加取半。无 pad geom 时
        退化为指尖 body 半间距。"""
        tot = 0.0
        n_pad = 0
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) not in tip_bodies:
                continue
            c_rel = sR.T @ (np.asarray(sd_.geom_xpos[g], float) - sp)
            sizes = np.asarray(m.geom_size[g], float)
            if int(m.geom_type[g]) == 6:  # box：局部半尺寸沿 u 投影
                Rl = sR.T @ np.asarray(sd_.geom_xmat[g], float).reshape(3, 3)
                e = float(np.sum(np.abs(Rl.T @ u) * sizes))
            else:
                e = float(sizes[0])
            tot += max(float(abs(float(np.dot(c_rel, u)))) - e, 0.0)
            n_pad += 1
        if n_pad == 0:
            return None
        return tot / 2.0

    _, sc = _scratch(adapter, +1.0)
    a_open = _inner_half(sd)
    a_closed = _inner_half(sc)
    if a_open is None:
        a_open = float(abs(np.dot(dv, u)) / 2.0)
        tpc = tip_points(sc)
        relc = [sR.T @ (np.asarray(p, float) - sp) for p in tpc]
        a_closed = float(np.linalg.norm(relc[1] - relc[0]) / 2.0)

    r = up_w = down_w = 0.0
    tip_r = 0.0
    tip_bodies = set()
    for b in range(m.nbody):
        nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
        if nm.endswith("_tip") and "gripper" in nm:
            tip_bodies.add(b)
    sp_world = np.asarray(sd.site_xpos[sid], float)
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        bn = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
        if not bn.startswith("gripper0"):
            continue
        # 纯世界坐标：相对 site 世界位置的水平半径与上下延伸
        gp = np.asarray(sd.geom_xpos[g], float)
        ext = float(np.asarray(m.geom_size[g], float)[0])
        r = max(r, float(np.hypot(gp[0] - sp_world[0],
                                 gp[1] - sp_world[1])) + ext)
        up_w = max(up_w, float(gp[2] - sp_world[2]) + ext)
        down_w = max(down_w, float(sp_world[2] - gp[2]) + ext)
        if b in tip_bodies:
            sizes = np.asarray(m.geom_size[g], float)
            n_ext = 3 if int(m.geom_type[g]) == 6 else 1
            tip_r = max(tip_r, float(np.max(sizes[:n_ext])))
    return {"u_site": u, "a_open": a_open, "a_closed": a_closed,
            "b_depth": b_depth, "r": r, "up": up_w, "down": down_w,
            "tip_r": tip_r,
            "tip_bodies": sorted(tip_bodies)}


def _tip_lever(adapter) -> float:
    """指尖杠杆臂（m）：姿态误差 Δθ 在指尖造成的横向偏移 = 本值×Δθ。
    实测：指尖相对 site 原点水平 a_open、竖直 b_depth（measure_hand）。"""
    h = measure_hand(adapter)
    return float(np.hypot(h["a_open"], h["b_depth"]))


# ==================== 精确射线探针 ====================

def _ray(adapter, pnt, vec, exclude_body: Optional[str] = None):
    """mujoco 精确射线（三角网格级）：(距离, geom_id)，无命中 -1。

    AABB 对开放腔体（抽屉/柜）是错的（外壳 AABB 罩住内腔），
    mj_ray 用真实碰撞网格，是唯一可信的几何探针。
    """
    import mujoco
    m, d = _native_md(adapter)
    bid = -1
    if exclude_body is not None:
        try:
            bid = int(m.body(adapter._resolve_body(exclude_body)).id)
        except Exception:
            bid = -1
    geomid = np.zeros(1, dtype=np.int32)
    dist = mujoco.mj_ray(m, d, np.asarray(pnt, float), np.asarray(vec, float),
                         None, 1, bid, geomid)
    return float(dist), int(geomid[0])


def _clearance(adapter, pnt, vec, exclude_body=None, cap=0.40) -> float:
    dist, _ = _ray(adapter, pnt, vec, exclude_body)
    return cap if dist < 0 else min(dist, cap)


def _tip_points_world(adapter, site_xyz: np.ndarray, hand: Dict[str, Any],
                      s: float) -> Tuple[np.ndarray, np.ndarray]:
    """全开工构型下 (内指尖, 外指尖) 世界坐标。site 姿态取世界固定姿态
    （OSC 锁定），开合轴/深度从现测 hand 几何经 site 框架变换。"""
    import mujoco
    m, _ = _native_md(adapter)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    # site 框架方向：OSC 姿态恒定 → 取当前 site 旋转
    sR = np.asarray(adapter.mj_data.site_xmat[sid], float).reshape(3, 3)
    u_world = sR @ hand["u_site"]
    # 指尖沿 site +z（朝下）伸出 b_depth
    z_off = sR @ np.array([0.0, 0.0, hand["b_depth"]])
    a = hand["a_open"]
    inner = np.asarray(site_xyz, float) - s * u_world * a + z_off
    outer = np.asarray(site_xyz, float) + s * u_world * a + z_off
    return inner, outer


# ==================== IK 可达性 ====================

def _rot_err(Rd: np.ndarray, Rc: np.ndarray) -> np.ndarray:
    """期望→当前的世界系旋转误差（轴角向量）：0.5·Σ cross(c,d)，
    与 mj_jacSite 的旋转 Jacobian（世界系角速度）同坐标。"""
    return 0.5 * (np.cross(Rc[:, 0], Rd[:, 0])
                  + np.cross(Rc[:, 1], Rd[:, 1])
                  + np.cross(Rc[:, 2], Rd[:, 2]))


def _ik_solve(adapter, target, joints, dof_adr, target_rot=None,
              q_init=None):
    """目标位姿 → DLS 逆解（scratch MjData）。target_rot=None 时为
    纯位置 IK（返回同旧）；给 3×3 旋转矩阵时为 6D 位姿 IK（第 4
    返回值为姿态残差 rad）。q_init 给定 arm 关节初始值（默认=live）。
    关节增量严格投影到限位内。
    返回 (qpos, 位置Jacobian, 位置残差m[, 姿态残差])。"""
    import mujoco
    m, live = _native_md(adapter)
    sd = mujoco.MjData(m)
    sd.qpos[:] = np.asarray(live.qpos, float)
    if q_init is not None:
        for j, v in zip(joints, np.asarray(q_init, float)):
            sd.qpos[int(m.jnt_qposadr[j])] = float(v)
    mujoco.mj_forward(m, sd)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    target = np.asarray(target, float)
    lam2 = IK_DAMP * IK_DAMP
    n_dims = 3 if target_rot is None else 6
    for _ in range(IK_ITERS):
        errp = target - np.asarray(sd.site_xpos[sid], float)
        Rcur = np.asarray(sd.site_xmat[sid], float).reshape(3, 3)
        if target_rot is None:
            err = errp
            if float(np.linalg.norm(errp)) < IK_TOL:
                break
        else:
            errr = _rot_err(np.asarray(target_rot, float), Rcur)
            err = np.concatenate([errp, errr])
            if float(np.linalg.norm(errp)) < IK_TOL \
                    and float(np.linalg.norm(errr)) < ROT_TOL:
                break
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv)) if target_rot is not None else None
        mujoco.mj_jacSite(m, sd, jacp, jacr, sid)
        if target_rot is None:
            J = jacp[:, dof_adr]
        else:
            J = np.vstack([jacp, jacr])[:, dof_adr]
        dq = J.T @ np.linalg.solve(
            J @ J.T + lam2 * np.eye(n_dims), err)
        for j, dqj in zip(joints, dq):
            adr = int(m.jnt_qposadr[j])
            lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
            sd.qpos[adr] = float(np.clip(sd.qpos[adr] + dqj, lo, hi))
        mujoco.mj_forward(m, sd)
    jacp = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, sd, jacp, None, sid)
    residual = float(np.linalg.norm(
        target - np.asarray(sd.site_xpos[sid], float)))
    if target_rot is None:
        return np.asarray(sd.qpos, float), jacp[:, dof_adr], residual
    Rcur = np.asarray(sd.site_xmat[sid], float).reshape(3, 3)
    res_rot = float(np.linalg.norm(
        _rot_err(np.asarray(target_rot, float), Rcur)))
    return np.asarray(sd.qpos, float), jacp[:, dof_adr], residual, res_rot


def _manipulability(J) -> float:
    return float(np.sqrt(max(float(np.linalg.det(J @ J.T)), 1e-12)))


def _limit_margin(m, q, joints) -> float:
    """最紧关节的归一化限位裕度（0=贴限位，1=行程正中）。"""
    worst = 1.0
    for j in joints:
        adr = int(m.jnt_qposadr[j])
        lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
        span = max(hi - lo, 1e-6)
        worst = min(worst, min(q[adr] - lo, hi - q[adr]) / span)
    return max(worst, 0.0)


def _max_reach_z(adapter, xy: np.ndarray, z_from: float) -> float:
    """某 xy 处 IK 可达的最高 z：从 z_from 向上按手掌半径步距扫描，
    取最高收敛高度。替代固定 CRUISE_CAP。"""
    m, _ = _native_md(adapter)
    joints = _arm_joints(m)
    dof_adr = [int(m.jnt_dofadr[j]) for j in joints]
    best = z_from
    step = max(measure_hand(adapter)["r"], 0.02)
    for k in range(16):
        z = z_from + (k + 1) * step
        _, _, residual = _ik_solve(adapter, [xy[0], xy[1], z],
                                   joints, dof_adr)
        if residual <= IK_TOL:
            best = z
    return best


# ==================== 走廊（几何现解，无固定巡航高度） ====================

def _point_free(adapter, p: np.ndarray, hand: Dict[str, Any],
                exclude: Optional[str]) -> bool:
    """单点容纳全手：上/下射线 ≥ 手包络，四水平射线 ≥ 1.5r
    （采样间距 r：连续屏障在距样本 ≤r/2 处必被检出，1.5r 为保守门）。"""
    p = np.asarray(p, float)
    if _clearance(adapter, p, (0, 0, 1), exclude, cap=0.5) < hand["up"]:
        return False
    if _clearance(adapter, p, (0, 0, -1), exclude, cap=0.3) < hand["down"]:
        return False
    for u in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        if _clearance(adapter, p, (u[0], u[1], 0), exclude, cap=0.3) \
                < 1.5 * hand["r"]:
            return False
    return True


def _vertical_clear(adapter, xy, z_lo, z_hi, hand, exclude) -> bool:
    """竖直段无碰撞：起点向上射线 ≥ 行程（精确），沿途按 r 间距
    采样验四侧净空。"""
    if z_hi - z_lo > _clearance(adapter, [xy[0], xy[1], z_lo],
                                (0, 0, 1), exclude, cap=0.6):
        return False
    step = max(hand["r"], 0.01)
    n = max(int(np.ceil((z_hi - z_lo) / step)), 1)
    for i in range(n + 1):
        z = z_lo + (z_hi - z_lo) * i / n
        for u in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if _clearance(adapter, [xy[0], xy[1], z],
                          (u[0], u[1], 0), exclude, cap=0.3) < 1.5 * hand["r"]:
                return False
    return True


def _horizontal_clear(adapter, p0, p1, hand, exclude) -> bool:
    """水平段无碰撞：沿段按 r 间距采样，验全手包络。"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    dist = float(np.linalg.norm(p1[:2] - p0[:2]))
    step = max(hand["r"], 0.01)
    n = max(int(np.ceil(dist / step)), 1)
    for i in range(n + 1):
        p = p0 + (p1 - p0) * i / n
        if not _point_free(adapter, p, hand, exclude):
            return False
    return True


def corridor_plan(adapter, target: np.ndarray, hand: Dict[str, Any],
                  exclude: Optional[str] = None
                  ) -> Optional[List[np.ndarray]]:
    """竖直–水平–竖直走廊现解：巡航高度从 z0 向上几何搜索（步距=手掌
    半径，第一个三段全通的高度），可达上界由 IK 在两端的最高可达 z
    取 min。无固定巡航高度/余量。"""
    cur = _eef(adapter)
    target = np.asarray(target, float)
    z0 = max(cur[2], target[2])
    zmax = min(_max_reach_z(adapter, cur[:2], cur[2]),
               _max_reach_z(adapter, target[:2], max(target[2], cur[2])))
    step = max(hand["r"], 0.01)
    z = z0
    while z <= zmax + 1e-9:
        legs = [np.array([cur[0], cur[1], z]),
                np.array([target[0], target[1], z]),
                target]
        if _vertical_clear(adapter, cur[:2], cur[2], z, hand, exclude) \
                and _horizontal_clear(adapter, legs[0], legs[1],
                                      hand, exclude) \
                and _vertical_clear(adapter, target[:2], z, target[2],
                                   hand, exclude) \
                and _point_free(adapter, target, hand, exclude):
            return legs
        z += step
    return None


# ==================== 关节空间运动规划（RRT-Connect） ====================

def _robot_geom_set(m) -> set:
    """机器人物体树的全部 geom（body 名前缀 robot0/gripper0）。"""
    import mujoco
    out = set()
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
        if nm.startswith(("robot0", "gripper0")):
            out.add(g)
    return out


def _pad_geom_set(m) -> set:
    """夹爪 pad（实际接触面）geom 集合：安装在 *_tip body 上的 geom。

    与 measure_hand 的指尖定义同源。区分 pad 与指身/腕部 geom 是
    接近段接触裁决的前提——腕下式接近时指身比 pad 高数厘米，矮目标
    上指身先碰顶沿被误判"到达抓深"，pad 悬在目标上方闭爪捏空
    （cream_cheese 1.8cm 盒实证）。"""
    import mujoco
    out = set()
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
        if nm.endswith("_tip") and "gripper" in nm:
            out.add(g)
    if not out:                       # 无 _tip 标注的夹爪：按名字兜底
        for g in range(m.ngeom):
            nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
            if "gripper" in nm and "pad" in nm:
                out.add(g)
    return out


def _body_static_flags(m) -> np.ndarray:
    """每个 body 的静态标志：自身到 world 的祖先链（含自身）上均无
    关节。geom 安装在该链上 ⟹ 其位形不随任何关节变化（下游子树的
    关节只动子树，不动本体）。地板/桌面/固定装修是 worldbody 直下
    无关节子体，为静态；机器人、自由关节物体为可动。

    不能用 body_rootid==0 当静态判据：本模型版本里 rootid 是子树根
    （每个顶层体自成一树），rootid==0 只命中 world 本体，地板
    （rootid=1）等静态面全被误判为可动——碰撞谓词"tether 只含可动
    物""浅接触静态面豁免""clean=对静态面零接触"三处因此集体失效
    （object:6 指身擦桌 -2.5mm 被判 clean、臂压桌被 tether 放行实证）。
    按 parent 链递推，每个模型 O(nbody) 一次。"""
    flags = getattr(m, "_darwin_body_static", None)
    if flags is not None:
        return flags
    flags = np.zeros(m.nbody, dtype=bool)
    flags[0] = True                    # world 本体
    for b in range(1, m.nbody):
        flags[b] = flags[int(m.body_parentid[b])] \
            and int(m.body_jntnum[b]) == 0
    try:
        m._darwin_body_static = flags  # 模型级缓存（MjModel 包装类可挂属性）
    except AttributeError:
        pass
    return flags


def _geom_is_static(m, g: int) -> bool:
    """geom 是否安装在静态环境体上（见 _body_static_flags）。"""
    return bool(_body_static_flags(m)[int(m.geom_bodyid[g])])


def _collision_status(m, sd, robot_geoms: set):
    """现算碰撞并分类接触（全部在 scratch MjData 上）：
    self_pairs：robot-robot 自接触 geom 对；
    re_geoms ：robot-env 接触中 env 侧 geom 集合；
    env_pairs：env-env 接触 geom 对；
    re_worst ：robot-env 接触中 env 侧 geom → 最深接触 dist（负=穿透），
               供"浅接触静态支撑面不算碰撞"的深度门使用。
    env_worst：env-env 接触对 → 最深接触 dist，供 tether 跨边界浅接触
               深度门（持物落座零间隙端点）使用。
    合法性由调用方对照"起点豁免集"判定——本函数不含任务知识。
    """
    import mujoco
    mujoco.mj_collision(m, sd)
    self_pairs, re_geoms, env_pairs = set(), set(), set()
    re_worst: Dict[int, float] = {}
    env_worst: Dict[tuple, float] = {}
    for i in range(sd.ncon):
        c = sd.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        r1, r2 = g1 in robot_geoms, g2 in robot_geoms
        if r1 and r2:
            if g1 != g2:
                self_pairs.add((min(g1, g2), max(g1, g2)))
        elif r1 or r2:
            eg = g2 if r1 else g1
            re_geoms.add(eg)
            d = float(c.dist)
            if d < re_worst.get(eg, 0.0):
                re_worst[eg] = d
        else:
            pr = (min(g1, g2), max(g1, g2))
            env_pairs.add(pr)
            d = float(c.dist)
            if d < env_worst.get(pr, 0.0):
                env_worst[pr] = d
    return self_pairs, re_geoms, env_pairs, re_worst, env_worst


def _make_free_fn(adapter, grip: float, target_rot=None,
                  inflation: float = 0.0,
                  allow_contact: "Optional[set]" = None,
                  weld_obj: "Optional[int]" = None):
    """构建 (arm 关节值序列 → 无碰撞 bool)，豁免全部从**起点构型**
    现测，规则任务无关：

    - 起点存在的 robot-robot 自接触对允许（装配自接触），新对=碰撞；
    - 起点与机器人接触的 env geom 记为 tether（被持物）：手与它的
      接触允许；它与环境的**新**接触=碰撞（持物不能磕碰）；
    - 其余任何 robot-env 新接触=碰撞，allow_contact 集合（可选）
      除外——接近恢复等"路径终点允许接触目标"的场景，与
      _approach_clear 的滑入力学同语义；
    - env-env 接触只检查跨 tether 边界的新对（桌面上既有的物体
      相互接触不算）；
    - inflation>0：碰撞检测时将 robot geom 的接触 margin 临时膨胀
      该值（transit 安全余量，值由调用方从跟踪误差导出），即手与
      障碍物几何距离 ≤ inflation 即视为碰撞；用于自由空间 transit；
      接近段调用方传 0（允许接触目标）。
    - weld_obj（可选）：被持物的 root body id。scratch 里物体是
      **冻结在起点位姿的自由体**——is_free 只重算臂，物体的 swept
      volume 对规划完全失明：持物 transit 会规划出横扫障碍正面的
      路径，执行时物体被物理按压、关节停滞判 reach_limit（goal:4
      实证：侧抓的碗在柜门面横向到位途中被门板顶住）。weld_obj
      给定后，每个检查构型把物体按"起点实测相对 site 位姿"重挂到
      手上（刚体夹持不变量），规划看到真实 swept volume。

    姿态是**终点约束**：路径中途腕部可自由变化，故 free_fn 不做
    姿态门；终点姿态由调用方在目标构型上单独复验（target_rot 参数
    仅为约定保留，不参与碰撞谓词）。
    """
    import mujoco
    m, sd0 = _scratch(adapter, grip)
    joints = _arm_joints(m)
    robot_geoms = _robot_geom_set(m)
    rglist = sorted(robot_geoms)
    margin0 = np.asarray(m.geom_margin, float).copy()
    allow = set(allow_contact) if allow_contact else set()
    _tip_r = float(measure_hand(adapter)["tip_r"])

    # ---- 持物焊接：起点实测物体-site 相对位姿 ----
    def _mat2quat(R):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
        return q

    def _qmul(a, b):
        r = np.zeros(4)
        mujoco.mju_mulQuat(r, np.asarray(a, float), np.asarray(b, float))
        return r

    def _qconj(a):
        r = np.zeros(4)
        mujoco.mju_negQuat(r, np.asarray(a, float))
        return r

    _weld = None
    _weld_geoms: List[int] = []
    if weld_obj is not None:
        sid0 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
        # 物体 root 须有唯一 free 关节（刚体道具的标准入库方式）
        jadr = -1
        if int(m.body_jntnum[weld_obj]) == 1:
            cand = int(m.body_jntadr[weld_obj])
            if int(m.jnt_type[cand]) == int(mujoco.mjtJoint.mjJNT_FREE):
                jadr = cand
        if jadr >= 0:
            sp0 = np.asarray(sd0.site_xpos[sid0], float)
            sR0 = np.asarray(sd0.site_xmat[sid0], float).reshape(3, 3)
            sq0 = _mat2quat(sR0)
            bp0 = np.asarray(sd0.xpos[weld_obj], float)
            bq0 = np.asarray(sd0.xquat[weld_obj], float)
            _weld = {
                "sid": sid0, "jadr": jadr,
                "adr": int(m.jnt_qposadr[jadr]),
                "rel_pos": sR0.T @ (bp0 - sp0),
                "rel_quat": _qmul(_qconj(sq0), bq0),
            }
            # 被持物子树全部 geom：inflation 膨胀同样作用于它们——
            # 持物的 swept volume 须与手同等的跟踪包络余量，否则
            # 零余量规划 + OSC 稳态偏置在执行时把物体压上障碍
            # （goal:4 实证：侧抓碗在柜门面/桌面上被物理按压至
            # 关节停滞，规划对零间隙路径一无所知）。
            _weld_geoms = sorted(
                g for g in range(m.ngeom)
                if int(m.body_rootid[int(m.geom_bodyid[g])]) == weld_obj)

    def _apply_weld():
        # 臂已就位（mj_forward 后）：把物体按 rel 重挂 site。
        sp = np.asarray(sd0.site_xpos[_weld["sid"]], float)
        sR = np.asarray(sd0.site_xmat[_weld["sid"]], float).reshape(3, 3)
        adr = _weld["adr"]
        sd0.qpos[adr:adr + 3] = sp + sR @ _weld["rel_pos"]
        sd0.qpos[adr + 3:adr + 7] = _qmul(_mat2quat(sR), _weld["rel_quat"])
        mujoco.mj_forward(m, sd0)

    def _set_margin(v: float):
        for g in rglist + _weld_geoms:
            m.geom_margin[g] = max(float(margin0[g]), float(v))

    s0, re0, env0, _rw0, _ew0 = _collision_status(m, sd0, robot_geoms)
    # tether 只含**可动**物体（被持物）：静态支撑面（地板/桌面，判据
    # 见 _body_static_flags）与机器人的接触不构成 tether——否则臂压桌
    # 的退化状态下桌面被记为 tether，此后所有 free 检查对压桌全部放行，
    # 规划会径直穿桌（object:6 link6 压桌 3.2cm、恢复路径穿桌实证）。
    tether = {g for g in re0 if not _geom_is_static(m, g)}
    allowed_self = set(s0)
    allowed_env_pairs = set(env0)

    def is_free(q_arm) -> bool:
        for j, v in zip(joints, q_arm):
            sd0.qpos[int(m.jnt_qposadr[j])] = float(v)
        mujoco.mj_forward(m, sd0)
        if _weld is not None:
            _apply_weld()
        if inflation:
            # 膨胀仅用于 robot-env 方向：第一轮以膨胀 margin 收集
            # robot-env / env-env，第二轮原始 margin 收集 robot-robot。
            _set_margin(inflation)
            try:
                _sp_i, reg, enp, _rw, enw = _collision_status(m, sd0, robot_geoms)
            finally:
                m.geom_margin[:] = margin0
            sp, _r0, _e0, _rw2, _ew2 = _collision_status(m, sd0, robot_geoms)
        else:
            sp, reg, enp, _rw3, enw3 = _collision_status(m, sd0, robot_geoms)
            # 接近语义（inflation=0）：浅接触静态支撑面不算碰撞——
            # 与 filter_site_config 的支撑深度门同一界（-0.5·tip_r，
            # 手几何比例）：薄物捏持的唯一可行构型常让指根/前臂擦过
            # 桌面 1-2mm（object:6 butter 工作区边界实证），filter 放行
            # 的构型若在 free 检查被零容差拒绝，pregrasp/RRT 永不通过。
            # 深压（< -0.5·tip_r）仍碰撞。膨胀（transit）分支不受影响。
            tol = -0.5 * _tip_r
            reg = {g for g in reg
                   if not (_geom_is_static(m, g)
                           and _rw3.get(g, 0.0) >= tol)}
        _dbg = bool(os.environ.get("LIBERO_DEBUG"))
        if not sp <= allowed_self:
            if _dbg:
                import mujoco as _mj
                new = sorted(set(sp) - allowed_self)[:3]
                print("LIVECHK code=S new=%s" % [
                    "%s~%s" % (_mj.mj_id2name(m, _mj.mjtObj.mjOBJ_GEOM, a),
                               _mj.mj_id2name(m, _mj.mjtObj.mjOBJ_GEOM, b))
                    for a, b in new])
            return False
        if not reg <= (tether | allow):
            if _dbg:
                import mujoco as _mj
                bad = sorted(set(reg) - tether - allow)[:4]
                print("LIVECHK code=R new=%s tether=%s" % (
                    [_mj.mj_id2name(m, _mj.mjtObj.mjOBJ_GEOM, g) for g in bad],
                    sorted(_mj.mj_id2name(m, _mj.mjtObj.mjOBJ_GEOM, g)
                           for g in tether)))
            return False
        enw_use = enw if inflation else enw3
        for a, b in enp:
            if (a, b) in allowed_env_pairs:
                continue
            if (a in tether) != (b in tether):
                if not inflation:
                    # 近距语义（起/终点逃逸泡、接近段）：持物与环境
                    # 的**浅**接触（≥ -0.5·tip_r，手几何比例）不构成
                    # 碰撞——On 放置终点是"物体底=支撑面"的零间隙几
                    # 何，落座接触正是目标关系本身；深度门以下（真
                    # 嵌入）仍碰撞。膨胀（transit）分支不豁免：路径
                    # 中段持物须满余量。
                    if enw_use.get((a, b), 0.0) >= -0.5 * _tip_r:
                        continue
                if _dbg:
                    import mujoco as _mj
                    print("LIVECHK code=E pair=%s~%s tether=%s" % (
                        _mj.mj_id2name(m, _mj.mjtObj.mjOBJ_GEOM, a),
                        _mj.mj_id2name(m, _mj.mjtObj.mjOBJ_GEOM, b),
                        sorted(_mj.mj_id2name(m, _mj.mjtObj.mjOBJ_GEOM, g)
                               for g in tether)))
                return False
        return True

    return is_free, {"joints": joints, "tether": sorted(tether)}


def _transit_inflation(adapter) -> float:
    """transit 安全余量（纯几何推导，无经验值）。

    流式执行在**关节归一化**空间以 EXEC_DQ 细分、跟踪门 2×细分；live
    构型与规划构型的关节偏差一个细分，故手上**任一 geom** 的最坏位置
    偏移 = |J_geom[:,dof]|·EXEC_DQ·关节行程。对全部机器人 geom、全部
    臂关节取最大值并×2 —— 即 live 手相对规划手的最坏跟踪包络。
    规划以此为余量避碰：只要规划手与障碍物距离 ≥ 该值，物理执行中
    手的任何部分都不会真正碰到障碍物。
    """
    import mujoco
    m, _ = _native_md(adapter)
    joints = _arm_joints(m)
    m2, sd = _scratch(adapter, -1.0)
    mujoco.mj_forward(m2, sd)
    robot_geoms = _robot_geom_set(m2)
    worst = 0.0
    Jp = np.zeros((3, m2.nv))
    for gg in robot_geoms:
        for j in joints:
            dof = int(m2.jnt_dofadr[j])
            Jp.fill(0.0)
            mujoco.mj_jacGeom(m2, sd, Jp, None, int(gg))
            span = float(m2.jnt_range[j][1] - m2.jnt_range[j][0])
            worst = max(worst,
                        float(np.linalg.norm(Jp[:, dof])) * EXEC_DQ * span)
    return float(2.0 * worst)


def _ik_solve_best(adapter, target_pos, target_rot, joints, dof_adr,
                   restarts: int = IK_RESTARTS):
    """多热启动 6D IK：live 构型优先；其余热启动取随机构型及 live
    与随机的插值（任务无关，解 DLS 的局部停滞）。取位置+姿态组合
    残差最优者，容差内提前退出。返回 (q_arm, res_p, res_rot)。
    """
    m, live = _native_md(adapter)
    adrs = [int(m.jnt_qposadr[j]) for j in joints]
    lo = np.array([float(m.jnt_range[j][0]) for j in joints])
    hi = np.array([float(m.jnt_range[j][1]) for j in joints])
    q_live = np.array([float(live.qpos[a]) for a in adrs])
    rng = np.random.default_rng(0)
    best = None
    for k in range(max(1, int(restarts))):
        if k == 0:
            qinit = None
        else:
            qr = lo + (hi - lo) * rng.random(len(joints))
            if k % 2:
                qinit = q_live + float(rng.random()) * (qr - q_live)
            else:
                qinit = qr
        qik, _J, rp, rr = _ik_solve(
            adapter, target_pos, joints, dof_adr,
            target_rot=np.asarray(target_rot, float), q_init=qinit)
        q = np.array([float(qik[a]) for a in adrs])
        if best is None or (rp + rr) < (best[1] + best[2]):
            best = (q, float(rp), float(rr))
        if rp <= IK_TOL and rr <= ROT_TOL:
            return best
    return best


def _ik_in_tol_solutions(adapter, target_pos, target_rot, joints, dof_adr,
                         restarts: int = IK_RESTARTS):
    """多热启动 6D IK 的**全部容差内解**生成器（任务无关，解 DLS 局部
    停滞）。"够得到"与"无碰撞"必须在同一解上成立：首个收敛解常是
    live 种子的伸展分支（前臂横扫邻物），碰撞拒绝后换肘/肩方向的重
    启解可能即无碰撞——过滤方应遍历多解而非只看最优一个。"""
    m, live = _native_md(adapter)
    adrs = [int(m.jnt_qposadr[j]) for j in joints]
    lo = np.array([float(m.jnt_range[j][0]) for j in joints])
    hi = np.array([float(m.jnt_range[j][1]) for j in joints])
    q_live = np.array([float(live.qpos[a]) for a in adrs])
    rng = np.random.default_rng(0)
    seen = set()
    for k in range(max(1, int(restarts))):
        if k == 0:
            qinit = None
        else:
            qr = lo + (hi - lo) * rng.random(len(joints))
            if k % 2:
                qinit = q_live + float(rng.random()) * (qr - q_live)
            else:
                qinit = qr
        qik, _J, rp, rr = _ik_solve(
            adapter, target_pos, joints, dof_adr,
            target_rot=np.asarray(target_rot, float), q_init=qinit)
        if rp > IK_TOL or rr > ROT_TOL:
            continue
        q = np.array([float(qik[a]) for a in adrs])
        sig = tuple(np.round(q, 3))
        if sig in seen:
            continue
        seen.add(sig)
        yield q, float(rp), float(rr)


def pregrasp_point(adapter, site, target_rot=None,
                   blocked: tuple = (), q_init=None):
    """抓取点 → 预抓取点（TAMP pregrasp），入口方向做通用半球搜索：

    以 site 为顶点，候选入口向量从腕接近轴 Rt[:,2] 出发，系统枚举
    倾角（0→π/2）× 方位；沿每个方向后退，取第一个"6D IK 收敛 +
    全手无碰撞"的点。枚举顺序：倾角优先（轴对齐最自然），同倾角按
    距离（最近优先）。总评估受 PREGRASP_MAX 预算。返回 (pre_site,
    q_pre) 或 None。规则任务无关。

    碰撞语义恒为**近距**（inflation=0，含浅接触静态支撑面豁免）：
    预抓取点是 transit→approach 的交接点，手必须进入"距目标/支撑面
    不足 transit 余量"的区域——膨胀余量在此物理上不成立（object:6
    butter 实证：1.8cm 余量下抓取位姿邻域内无任何可行预达点，搜索
    被逼退 60cm 外）。transit 余量由 plan_arm_path 的路径检查保证；
    末段由 _approach_clear（同为近距语义）守。

    blocked：已失败的撤退方向单位向量集合（调用方在接近段/ transit
    规划失败后回灌，避免反复选中同一坏方向——预抓取点本身无碰撞不
    代表从该点出发的接近段/路径可行）。

    q_init：抓取点处已通过碰撞复验的臂构型（调用方传 filter/
    analytic_side_grip 的解）。7DOF 臂对固定腕位有 1 维肘冗余，随机
    重启会落到不同分支——工作区边界处可达壳极薄，唯复验分支自身是
    干净构型，随机分支深压支撑面（object:6 butter 实证：_ik_solve_best
    在抓取点解 link5/6 压桌 8~30cm，而 filter 解零接触）。给定时以它
    为链种子并先作 site 构型直查。缺省回退 _ik_solve_best（旧行为）。"""
    import mujoco
    m, live = _native_md(adapter)
    joints = _arm_joints(m)
    adrs = [int(m.jnt_qposadr[j]) for j in joints]
    dof_adr = [int(m.jnt_dofadr[j]) for j in joints]
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    if target_rot is None:
        Rt = np.asarray(live.site_xmat[sid], float).reshape(3, 3)
    else:
        Rt = np.asarray(target_rot, float).reshape(3, 3)
    free_fn, _ = _make_free_fn(adapter, -1.0, Rt, inflation=0.0)
    site = np.asarray(site, float)

    q, rp, rr = _ik_solve_best(adapter, site, Rt, joints, dof_adr,
                               restarts=IK_RESTARTS)
    if q_init is not None:
        # 复验构型直查：它已零/浅接触通过 filter，大概率直接满足 free
        qv = np.asarray(q_init, float).ravel()
        if qv.shape[0] == len(adrs) and free_fn(qv):
            return site, qv
        q_seed = qv          # 链种子 = 复验分支
    else:
        q_seed = np.asarray(q, float)
    if q_init is None and rp <= IK_TOL and rr <= ROT_TOL and free_fn(q):
        return site, q
    # 分支链式热启动种子：7DOF 臂对固定腕位的 1 维肘部冗余下，随机
    # 重启在每步撤退点都落到不同肘分支——可达壳薄的工作区边界处，
    # 唯有已复验的抓取分支（q_init）能延续（随机分支深压支撑面被
    # free_fn 拒，object:6 butter 实证：_ik_solve_best 抓取点解
    # link5/6 压桌 8~30cm，filter 零接触解被丢弃后 19/19 方向全
    # freefail；传 q_init 后以复验分支为链种子）。
    z = Rt[:, 2] / np.linalg.norm(Rt[:, 2])
    # 倾角网格（含轴）；方位在垂直平面内取 6 向（倾角>0）
    tilt_dirs: List[np.ndarray] = [z]
    for k in range(1, 4):
        tilt = math.pi / 2.0 * k / 3
        for az in range(6):
            n = Rt[:, 0] * math.cos(az * math.pi / 3.0) \
                + Rt[:, 1] * math.sin(az * math.pi / 3.0)
            n = n / max(float(np.linalg.norm(n)), 1e-9)
            tilt_dirs.append(z * math.cos(tilt)
                             + np.cross(n, z) * math.sin(tilt))
    # 评估预算在各方向间平分，保证整个半球都被探索到（单方向 IK 可达
    # 但碰撞不通过时不能耗尽全局预算）。搜索分两轮：第一轮粗步距
    # （手掌半径之半——现测量，非经验值）在全半球找第一个可行方向；
    # 第二轮沿该方向以 IK_TOL 细步距重走，取精确的最近无碰撞预达点。
    # 单轮细步距会把预算耗在毫米级推进上而到不了自由空间（抽屉把手等
    # 深埋在夹缝中的目标，净空起点常在 10cm 量级）。
    # 屏蔽已失败撤退方向（-v 即实际撤退向）
    if blocked:
        tilt_dirs = [v for v in tilt_dirs
                     if all(float(np.dot(-v, r)) <= 0.9 for r in blocked)]
        if not tilt_dirs:
            return None
    steps_each = max(1, PREGRASP_MAX // len(tilt_dirs))
    coarse = max(measure_hand(adapter)["r"] / 2.0, IK_TOL)

    def _solve_chained(p, seed):
        """链式种子 IK（延续浅擦分支）；种子不收敛回退随机重启。
        返回 (arm_config, rp, rr) 或 None（完全不可解）。"""
        qf, _J, rp, rr = _ik_solve(adapter, p, joints, dof_adr,
                                   target_rot=np.asarray(Rt, float),
                                   q_init=seed)
        qa = np.array([float(qf[a]) for a in adrs])
        if rp <= IK_TOL and rr <= ROT_TOL:
            return qa, float(rp), float(rr)
        qb, rb, rb_r = _ik_solve_best(adapter, p, Rt, joints, dof_adr,
                                      restarts=IK_RESTARTS)
        if rb <= IK_TOL and rb_r <= ROT_TOL:
            return np.asarray(qb, float), float(rb), float(rb_r)
        return None

    # 粗搜：全半球各方向取**首个** free 点，再从中选**最近**的。
    # 预抓取点的语义是"距抓取点最近的自由 ingress"——按枚举序取首个
    # 命中方向会选到沿斜向退得很远的点，而轴向内 1~2cm 处可能就有
    # 自由口袋（object:7 milk 实证：枚举序首选退 58cm 的斜向，approach
    # 自碰撞连环失败；轴向近点一次通过）。预算在各方向间平分不变。
    hits = []
    for v in tilt_dirs:
        dist = 0.0
        seed = q_seed
        for k in range(1, steps_each + 1):
            dist += coarse
            p = site - dist * v
            r = _solve_chained(p, seed)
            if r is None:
                break  # 该方向 IK 已不可达，更远无意义
            q, rp, rr = r
            seed = q                      # 同分支沿撤退向延续
            if free_fn(q):
                hits.append((dist, v, (p, q)))
                break
    best_v, best_coarse = None, None
    if hits:
        hits.sort(key=lambda t: t[0])
        _, best_v, best_coarse = hits[0]
    if best_v is None:
        # 密排兜底：粗网格（掌半径之半）会跳过比步距浅的自由口袋——
        # 可达壳薄的边界构型，自由集可能只在抓取点外 1~2cm 内存在
        # （object:6 butter 实证：dir15 距 2cm 处有唯一 free 解，4cm
        # 处即无；粗步距 3.4cm 直接跨过）。粗搜全败时以 IK_TOL 步距
        # 重走全半球（同一谓词、更密采样——自由集是开集，任何自由
        # 口袋内含 IK_TOL 网格点的概率随步距趋零而趋于 1）。
        dhits = []
        for v in tilt_dirs:
            dist = 0.0
            seed = q_seed
            for k in range(1, steps_each + 1):
                dist += IK_TOL
                p = site - dist * v
                r = _solve_chained(p, seed)
                if r is None:
                    break
                q, rp, rr = r
                seed = q
                if free_fn(q):
                    dhits.append((dist, v, (p, q)))
                    break
        if dhits:
            dhits.sort(key=lambda t: t[0])
            return dhits[0][2]
        return None
    dist = 0.0
    seed = q_seed
    for k in range(1, steps_each + 1):
        dist += IK_TOL
        p = site - dist * best_v
        r = _solve_chained(p, seed)
        if r is None:
            break  # 细搜 IK 断点先于粗搜命中点：退回粗搜解
        q, rp, rr = r
        seed = q
        if free_fn(q):
            return p, q
    return best_coarse


@_timed
def plan_arm_path(adapter, target_xyz, *, grip: float = -1.0,
                  target_rot: Optional[np.ndarray] = None,
                  inflation: float = 0.0,
                  allow_contact: "Optional[set]" = None,
                  weld_obj: "Optional[int]" = None,
                  info: Optional[Dict[str, Any]] = None,
                  step: float = RRT_STEP,
                  max_nodes: int = RRT_MAX_NODES,
                  seed: int = 0):
    """自由空间关节路径：RRT-Connect 双树（Kuffner & LaValle 2000），
    关节行程归一化度量 + shortcut 平滑。终点 = target_xyz 的 6D 位姿
    IK 构型（目标姿态默认取当前 site R——即 OSC 锁定姿态；位置残差
    ≤ IK_TOL、姿态残差 ≤ ROT_TOL、自身无碰撞）。inflation>0 时整条
    路径按该安全余量避碰（transit 用）。weld_obj（被持物 root body
    id）给定后碰撞检查把物体按实测相对位姿重挂手上，规划看到持物
    swept volume（见 _make_free_fn）。step/max_nodes 为扩展步长与
    节点预算（缺省 RRT_STEP/RRT_MAX_NODES；窄走廊重试可降步长提
    预算，纯算法资源参数）。seed 为采样序列种子（缺省 0 保执行可复
    现；同参数重试应换种子——RRT 概率完备，单次确定性失败是搜索伪
    影而非不可达证据，object:7 milk 实证：同场景固定种子每次恰在
    235 节点告败，三个不同抓姿全同）。返回关节构型序列（q0→qg）
    或 None。概率完备：失败只可能来自节点预算 max_nodes。
    """
    m, live = _native_md(adapter)
    joints = _arm_joints(m)
    adrs = [int(m.jnt_qposadr[j]) for j in joints]
    dof_adr = [int(m.jnt_dofadr[j]) for j in joints]
    lo = np.array([float(m.jnt_range[j][0]) for j in joints])
    hi = np.array([float(m.jnt_range[j][1]) for j in joints])
    span = np.maximum(hi - lo, 1e-6)

    import mujoco
    csid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    if target_rot is None:
        target_rot = np.asarray(live.site_xmat[csid], float).reshape(3, 3)
    target_rot = np.asarray(target_rot, float)

    q0 = np.array([float(live.qpos[a]) for a in adrs])

    free_fn, _info = _make_free_fn(adapter, grip, target_rot,
                                   inflation=inflation,
                                   allow_contact=allow_contact,
                                   weld_obj=weld_obj)
    # 起/终点逃逸泡：执行残余（OSC 稳态偏置/接近段过冲）会让**物理
    # 起点**落在 transit 余量内（eef 距障 <inflation 但无真实接触）；
    # **终点**是 transit→approach 交接点，手必须进入目标 tip_r 邻域，
    # 膨胀余量在交接区物理上不成立（object:6 butter 实证：交接点
    # 邻域全余量下零可行构型，RRT 6000 节点耗尽、4 次预抓方向全败）。
    # 泡内（起/终点附近 3 步扩展范围）的点以**近距**语义重判（真实
    # 接触仍由 tether/支撑面深度门豁免），泡外全余量不变。泡尺寸=
    # 扩展步整数倍，树总能逐步爬出泡；爬出方向受全余量约束，不会贴
    # 着障碍钻。末段安全由 approach_clear（近距）+ 停滞检测 + 恢复兜底。
    free0, _ = _make_free_fn(adapter, grip, target_rot, inflation=0.0,
                             allow_contact=allow_contact, weld_obj=weld_obj)
    if not free_fn(q0) and not free0(q0):
        return None
    bubble = 3.0 * RRT_STEP

    def point_free(x) -> bool:
        if float(np.linalg.norm(x - x0)) <= bubble \
                or float(np.linalg.norm(x - xg)) <= bubble:
            return free0(lo + x * span)
        return free_fn(lo + x * span)

    # 终点构型多解筛选：同 filter_site_config——7DOF 冗余下首个收敛解
    # 常碰撞，遍历容差内解取第一个可行的（失败方是"找不到无碰撞终点"
    # 而非"终点不可达"）。判定用近距语义（free0，与终点逃逸泡同一
    # 语义）：膨胀余量在交接点不成立（同上，object:6 butter 实证）。
    qg = None
    _n_tol = 0
    for q, _rp, _rr in _ik_in_tol_solutions(
            adapter, np.asarray(target_xyz, float), target_rot,
            joints, dof_adr, restarts=4 * IK_RESTARTS):
        _n_tol += 1
        if free0(q):
            qg = q
            break
    if info is not None:
        info["endpoint_tol_solutions"] = _n_tol
        info["endpoint_free"] = qg is not None
    if qg is None:
        return None

    x0 = (q0 - lo) / span
    xg = (qg - lo) / span

    def seg_free(a, b):
        d = b - a
        # 碰撞细分与流式执行同精度（EXEC_DQ），杜绝"规划无碰撞、执行
        # 擦碰"的分辨率漏洞。点级自由判定走 point_free（起点逃逸泡）。
        n = max(int(np.ceil(float(np.linalg.norm(d)) / EXEC_DQ)), 1)
        for k in range(1, n + 1):
            if not point_free(a + d * k / n):
                return False
        return True

    if float(np.linalg.norm(xg - x0)) <= step and seg_free(x0, xg):
        return [q0, qg]

    rng = np.random.default_rng(seed)
    nd = len(joints)

    class Tree:
        def __init__(self, root):
            self.xs = [np.asarray(root, float).copy()]
            self.par = [-1]

    def nearest(T, x):
        A = np.asarray(T.xs)
        return int(np.argmin(np.linalg.norm(A - x, axis=1)))

    def steer(a, b):
        d = b - a
        nrm = float(np.linalg.norm(d))
        if nrm <= step:
            return b.copy(), True
        return a + step * d / nrm, False

    Ta, Tb = Tree(x0), Tree(xg)
    solution = None
    for _it in range(max_nodes):
        qr = rng.random(nd)
        iN = nearest(Ta, qr)
        xn, _reached = steer(Ta.xs[iN], qr)
        if np.array_equal(xn, Ta.xs[iN]) \
                or not seg_free(Ta.xs[iN], xn):
            Ta, Tb = Tb, Ta
            continue
        iNew = len(Ta.xs)
        Ta.xs.append(xn)
        Ta.par.append(iN)
        iC = -1
        while True:
            iN2 = nearest(Tb, xn)
            xc, reached = steer(Tb.xs[iN2], xn)
            if not seg_free(Tb.xs[iN2], xc):
                break
            iC = len(Tb.xs)
            Tb.xs.append(xc)
            Tb.par.append(iN2)
            if reached:
                pa = []
                i = iNew
                while i != -1:
                    pa.append(Ta.xs[i])
                    i = Ta.par[i]
                pa.reverse()
                pb = []
                i = iC
                while i != -1:
                    pb.append(Tb.xs[i])
                    i = Tb.par[i]
                solution = pa + pb
                break
        if solution is not None:
            break
        Ta, Tb = Tb, Ta
    if solution is None:
        if info is not None:
            info["rrt_nodes"] = len(Ta.xs) + len(Tb.xs)
        return None

    # 方向校正：首点必须为 x0
    if float(np.linalg.norm(solution[0] - x0)) \
            > float(np.linalg.norm(solution[-1] - x0)):
        solution.reverse()

    # shortcut 平滑：可直线直达的两点之间剪掉
    path = list(solution)
    for _ in range(SHORTCUT_N):
        if len(path) < 3:
            break
        i = int(rng.integers(0, len(path) - 2))
        j = int(rng.integers(i + 2, len(path)))
        if seg_free(path[i], path[j]):
            path = path[:i + 1] + path[j:]

    return [lo + x * span for x in path]


# ==================== GraspNet 学习型抓取综合（任务无关） ====================

def _grasp_backend_candidates(pc: np.ndarray, c0: np.ndarray,
                              hand: Dict[str, Any], top_k: int
                              ) -> List[Dict[str, Any]]:
    """点云（世界系）→ 可插拔抓取后端 → 世界系夹爪 site 候选。

    帧对齐：GraspNet/GSNet 训练于相机系点云（相机系 +z 看向场景内 =
    物理接近方向），仿真点云是世界系（物理接近向 -z）。先将点云
    平移到 c0 并经基变换 F=diag(1,-1,-1) 送入网络，预测后再变换回。

    GraspGroup 17 列：[score,width,height,depth,rot(9)=4:13,
    center(3)=13:16,obj_id]。旋转矩阵列语义（loss_utils.
    batch_viewpoint_params_to_matrix）：col0=approach（物理接近向）、
    col1=binormal、col2=axis（开合方向）。grip site 列序 col0=开合、
    col1=横向、col2=接近（指尖伸出向），保持右手系：
        site_R[:,0] = Rg[:,2]
        site_R[:,1] = -Rg[:,1]
        site_R[:,2] = Rg[:,0]
    center=手指基部点，指尖在 +depth；指尖=site+site_R·[0,0,b_depth]，
    故 site = center + (depth-b_depth)·site_R[:,2]。
    width 为指尖全宽，下游与 a_open（半间距）比较，统一换算半宽。
    """
    from darwin.skills.perception.grasp_backends import load_backend
    be = load_backend()
    if be is None:
        return []
    F = GRASPNET_FRAME
    pc_v = np.asarray((pc - c0) @ F, np.float32)
    gg = be.predict_gg(pc_v, top_k)
    if gg is None or len(gg) == 0:
        return []
    out: List[Dict[str, Any]] = []
    for g in gg:
        Rg = F @ np.asarray(g[4:13], float).reshape(3, 3)
        Rs = np.empty((3, 3), float)
        Rs[:, 0] = Rg[:, 2]
        Rs[:, 1] = -Rg[:, 1]
        Rs[:, 2] = Rg[:, 0]
        center = np.asarray(g[13:16], float) @ F.T + c0
        depth = float(g[3])
        site = center + (depth - hand["b_depth"]) * Rs[:, 2]
        out.append({"R": Rs, "site": site, "center": center,
                    "width": float(g[1]) / 2.0, "score": float(g[0])})
    return out


def graspnet_candidates(adapter, obj: str, *, top_k: int = 32
                        ) -> List[Dict[str, Any]]:
    """物体真值表面点云 → 抓取后端候选 → 世界系夹爪 site 表达。"""
    from darwin.skills.perception.grasp import sample_object_point_cloud
    hand = measure_hand(adapter)
    pc = sample_object_point_cloud(adapter, obj, n_points=20000)
    if len(pc) == 0:
        return []
    c0 = pc.mean(axis=0)
    return _grasp_backend_candidates(pc, c0, hand, top_k)


@_timed
def graspnet_scene_candidates(adapter, obj: str, *, top_k: int = 64
                              ) -> List[Dict[str, Any]]:
    """全场景点云 → 抓取后端候选 → 世界系 site 表达。

    与 graspnet_candidates 的唯一差别是输入：喂入除机器人外的全
    场景表面点（场景质心作为虚拟相机原点），使网络对支撑几何有感知；
    候选可落在任意物体上，由下游站点碰撞过滤选出真正抓在目标物上
    的候选。无任务分支。
    """
    from darwin.skills.perception.grasp import sample_scene_point_cloud
    hand = measure_hand(adapter)
    pc = sample_scene_point_cloud(adapter, n_points=20000)
    if len(pc) == 0:
        return []
    c0 = pc.mean(axis=0)
    return _grasp_backend_candidates(pc, c0, hand, top_k)


def _approach_clear(adapter, obj: str, pre_site, site, Rs,
                    q_init=None) -> bool:
    """接近段碰撞校验（TAMP straight-line 要求）：pre_site → site 的
    直线上以 IK_TOL 步距插值，每点 6D IK（链式热启动）+ 碰撞复验：
    自接触新对始终拒绝；与目标物的接触全程允许（滑入力学），与
    任何异物体的接触拒绝。规则任务无关。

    q_init：pre_site 处的已知可行臂构型（调用方传 pregrasp_point
    的解）——起点 IK 以它作种子，使整段链式跟踪同一肘部分支；
    薄可达壳下随机种子会落到自碰撞分支（object:6 butter 实证：
    link2~finger2 新自接触对）。缺省从 live 构型种子（旧行为）。"""
    import mujoco
    m, _ = _native_md(adapter)
    joints = _arm_joints(m)
    dof_adr = [int(m.jnt_dofadr[j]) for j in joints]
    p0 = np.asarray(pre_site, float)
    p1 = np.asarray(site, float)
    length = float(np.linalg.norm(p1 - p0))
    n = max(int(np.ceil(length / IK_TOL)), 2)
    m2, sd = _scratch(adapter, -1.0)
    robot_geoms = _robot_geom_set(m2)
    s0, _re, _, _rw0, _ew0 = _collision_status(m2, sd, robot_geoms)
    allowed = _site_grasp_geoms(adapter, obj) | _support_geoms(adapter, obj)
    _tip_r = float(measure_hand(adapter)["tip_r"])
    dbg = bool(os.environ.get("LIBERO_DEBUG"))
    # q_init（pre_site 构型种子）非空时首点从该分支出发；None 时
    # _ik_solve 用 live 构型种子（旧行为）。
    adrs = [int(m2.jnt_qposadr[j]) for j in joints]
    for k in range(n + 1):
        p = p0 + (p1 - p0) * (k / n)
        q_sol, _J, rp, rr = _ik_solve(adapter, p, joints, dof_adr,
                                      target_rot=np.asarray(Rs, float),
                                      q_init=q_init)
        q_init = np.array([float(q_sol[a]) for a in adrs])
        if float(rp) > IK_TOL or float(rr) > ROT_TOL:
            if dbg:
                print(f"[dbg] approach_clear IKfail k={k}/{n} p="
                      f"{np.round(p,4).tolist()} rp={float(rp):.4f} "
                      f"rr={float(rr):.3f}", flush=True)
            return False
        for j, v in zip(joints, q_init):
            sd.qpos[int(m2.jnt_qposadr[j])] = float(v)
        mujoco.mj_forward(m2, sd)
        sp, reg, _, rew, _ew = _collision_status(m2, sd, robot_geoms)
        if not sp <= s0:
            if dbg:
                g2n = lambda g: str(mujoco.mj_id2name(
                    m2, mujoco.mjtObj.mjOBJ_GEOM, g) or g)
                nm = [f"{g2n(a)}~{g2n(b)}" for a, b in (sp - s0)]
                print(f"[dbg] approach_clear SELFCOLL k={k}/{n} p="
                      f"{np.round(p,4).tolist()} new={nm}", flush=True)
            return False
        # 接近运动中与目标物的接触是抓取力学本身（手指沿内壁/边沿
        # 滑入），全程允许；与支撑面的**浅**接触（搁置/擦过 ≥ -0.5·tip_r，
        # 与 filter_site_config 同界）允许，深压拒绝；任何与异物体的
        # 接触均拒绝。
        _sup = set(reg) & (allowed - _site_grasp_geoms(adapter, obj))
        _deep = {g for g in _sup
                 if rew.get(g, 0.0) < -0.5 * _tip_r}
        if _deep or not set(reg) <= allowed:
            if dbg:
                bad = set(reg) - allowed
                nm = sorted({str(mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_GEOM,
                                                   g) or g) for g in bad})
                print(f"[dbg] approach_clear BLOCKED k={k}/{n} p="
                      f"{np.round(p,4).tolist()} by={nm}", flush=True)
            return False
    return True


def _site_grasp_geoms(adapter, obj: str):
    """目标物装配子树的全部 geom 集合（站点接触允许集）。"""
    import mujoco
    m, _ = _native_md(adapter)
    root = adapter._resolve_body(obj)
    rid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, root)
    return {g for g in range(m.ngeom)
            if int(m.body_rootid[int(m.geom_bodyid[g])]) == rid}


def _support_geoms(adapter, obj: str) -> set:
    """目标的支撑面 geom 集合（live 接触现测）：与目标底面作近垂直
    接触（|nz|>0.7）且接触点高度在底面 1cm 以内的 env 侧 geom。

    机制意义：薄于指尖直径的物体，合法的捏持构型必然让指尖/掌根
    搁置或掠过支撑面（手指贴桌滑入是夹薄物的唯一运动学解）——此类
    接触是支撑，不是阻挡。判据全部来自现测几何（接触法向+高度），
    无任务知识。"""
    import mujoco
    m, d = _native_md(adapter)
    allowed = _site_grasp_geoms(adapter, obj)
    b = adapter.object_bounds(obj)
    z_bot = float(b["z_bottom"])
    out = set()
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        o1, o2 = g1 in allowed, g2 in allowed
        if not (o1 or o2):
            continue
        nz = abs(float(c.frame[2]))
        if nz < 0.7:
            continue
        env_g = g2 if o1 else g1
        cz = float(c.pos[2])
        if abs(cz - z_bot) <= 0.01:
            out.add(env_g)
    return out


def filter_site_config(adapter, obj: str, site, Rs):
    """站点碰撞过滤（通用，无任务知识）：对抓取站点求 6D IK，该构型
    必须满足：
    - 无机器人自接触新对；
    - robot-env 接触的 env 侧 geom 全部属于目标物**或其支撑面**（允
      许接触目标；指尖搁置支撑面是夹薄物的合法运动学解，见
      _support_geoms；接触其他物体 = 抓点不可行）；
    - 位置与姿态在 IK 容差内。

    返回 (q_arm, res_p, res_r) 或 None（不可行）。
    """
    import mujoco
    m, _ = _native_md(adapter)
    joints = _arm_joints(m)
    dof_adr = [int(m.jnt_dofadr[j]) for j in joints]
    m2, sd = _scratch(adapter, -1.0)
    robot_geoms = _robot_geom_set(m2)
    tgt = _site_grasp_geoms(adapter, obj)
    support = _support_geoms(adapter, obj)
    allowed = tgt | support
    adrs = [int(m2.jnt_qposadr[j]) for j in joints]
    # 重启数取 4×IK_RESTARTS：7DOF 臂对固定腕位有 1 维肘部冗余，随机
    # 热启动要足够密才能采样到无碰撞的肘向（实测拥挤场景 6 重启全碰
    # 撞、24 重启出解）
    dbg = bool(os.environ.get("LIBERO_DEBUG"))
    n_sol, seen_self, seen_env = 0, set(), set()
    seen_buried = 0
    pads = _pad_geom_set(m2)
    tip_r = float(measure_hand(adapter)["tip_r"])
    # 多解择优：容差内全部解中返回**非 pad 机器人 geom 对静态 geom
    # 零接触间隙最大**的解（clearance = -最深接触 dist，无接触 = 1.0）。
    # 薄可达壳下首个通过解常是擦碰分支（邻域全部深压/IK 断，下游
    # pregrasp/approach 连环失败）；零接触解的可达管道是胖壳
    # （object:6 butter 实证：擦碰解 clearance≈0，+2mm 高度 25 个
    # 零接触解）。
    best_pass = None       # (clearance, q, rp, rr)

    def _clearance() -> float:
        worst = 0.0          # dist ≤ 0；取最深接触
        found = False
        for i in range(sd.ncon):
            c = sd.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            r1, r2 = g1 in robot_geoms, g2 in robot_geoms
            if not (r1 or r2):
                continue
            rg = g1 if r1 else g2
            eg = g2 if r1 else g1
            # pad 与静态面的浅接触是薄物捏持的合法搁置（设计语义），
            # 不计入间隙惩罚；非 pad geom（臂链接/掌/指身）必须零接触。
            if rg in pads:
                continue
            if not _geom_is_static(m2, eg):
                continue      # 可动物体（目标等）不计
            found = True
            if float(c.dist) < worst:
                worst = float(c.dist)
        return (-worst) if found else 1.0

    for q, rp, rr in _ik_in_tol_solutions(adapter, site, Rs, joints,
                                          dof_adr,
                                          restarts=4 * IK_RESTARTS):
        n_sol += 1
        for j, v in zip(joints, q):
            sd.qpos[int(m2.jnt_qposadr[j])] = float(v)
        mujoco.mj_forward(m2, sd)
        sp, reg, _enp, _rw2, _ew2 = _collision_status(m2, sd, robot_geoms)
        if sp:
            seen_self |= sp
            continue
        if not set(reg) <= allowed:
            seen_env |= (set(reg) - allowed)
            continue
        # 埋入复验（两类共用同一深度界）：
        # (a) pad∩目标——构型层面允许"接触目标"（按压/搁置是合法解），
        #     但 pad 整体埋入目标实体是退化构型（rim 捏落在实心盒面/角
        #     上时一侧 pad 穿入实物数 cm，闭爪捏空，object:6 butter
        #     f_hold=0 实证）；
        # (b) robot∩支撑面——支撑语义是"指尖/掌根搁置或掠过"（浅接
        #     触，手指贴桌滑入是夹薄物的唯一运动学解），臂链接压入支
        #     撑面（object:6 link6 压桌 3.2cm 致 OSC 伺服渐近卡死实证）
        #     是构型退化，不是搁置。接触量级穿透 ≤ 接触 margin（~mm），
        #     埋入 ≥ pad 半厚（cm 级）；以 0.5·tip_r 为界（手几何比例，
        #     任务无关）。
        buried = False
        for i in range(sd.ncon):
            c = sd.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            hit_target = (g1 in pads and g2 in tgt) or \
                (g2 in pads and g1 in tgt)
            hit_support = (g1 in support) or (g2 in support)
            if (hit_target or hit_support) and float(c.dist) < -0.5 * tip_r:
                buried = True
                break
        if buried:
            seen_buried += 1
            continue
        clr = _clearance()
        if best_pass is None or clr > best_pass[0]:
            best_pass = (clr, q, rp, rr)
    if best_pass is not None:
        _clr, q, rp, rr = best_pass
        return q, rp, rr
    if dbg:
        g2n = lambda g: str(mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_GEOM, g)
                            or g)
        self_nm = sorted({f"{g2n(a)}~{g2n(b)}" for a, b in seen_self})[:4]
        env_nm = sorted({g2n(g) for g in seen_env})[:4]
        print(f"[dbg] site_filter n_sol={n_sol} site={np.round(site,4).tolist()}"
              f" self={self_nm} env={env_nm} buried={seen_buried}", flush=True)
    return None


# ==================== 抓取点约束优化 ====================

def _dbg_dump_contacts(adapter, tag: str = "") -> None:
    """打印 live 接触对物证（geom 名 + 法向力 + 世界系法向）：停滞/闭爪
    等事件的现场快照，仅 LIBERO_DEBUG 下启用。法向带 z 分量是诊断
    "捏持滑脱/下压楔紧"的第一证据：水平捏持要求 |nz|≈0。"""
    if not os.environ.get("LIBERO_DEBUG"):
        return
    import mujoco
    m, d = _native_md(adapter)
    res = np.zeros(6)
    out = []
    for i in range(d.ncon):
        c = d.contact[i]
        mujoco.mj_contactForce(m, d, i, res)
        f = float(np.linalg.norm(res[:3]))
        if f < 0.05:
            continue
        n1 = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                   int(c.geom1)) or c.geom1)
        n2 = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM,
                                   int(c.geom2)) or c.geom2)
        nz = float(c.frame[2])
        out.append(f"{n1}~{n2}:{f:.1f}N(nz={nz:+.2f})")
    print(f"[dbg] CONTACTS[{tag}] n={len(out)} " + " ".join(out[:12]),
          flush=True)


def _dbg_force_balance(adapter, obj: str) -> None:
    """打印目标物受力的世界系合力（接触力 + 重力）：诊断"夹住了却提
    不起来"的裁决量——静止持物时合力 ≈ 0；提离过程中合力 +z 应 >
    重力。符号约定经静止基准自校验后取使"碗静置在支撑上时合力≈0"
    的方向。仅 LIBERO_DEBUG 下启用。"""
    if not os.environ.get("LIBERO_DEBUG"):
        return
    import mujoco
    m, d = _native_md(adapter)
    allowed = _site_grasp_geoms(adapter, obj)
    res = np.zeros(6)
    fsum = np.zeros(3)
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        o1, o2 = g1 in allowed, g2 in allowed
        if not (o1 or o2):
            continue
        mujoco.mj_contactForce(m, d, i, res)
        f_local = np.asarray(res[:3], float)   # 接触系（法向+2切向）
        R = np.asarray(c.frame[:9], float).reshape(3, 3)  # 接触系→世界系
        fv = R @ f_local
        n = np.asarray(c.frame[:3], float)
        f_n = float(np.dot(fv, n))
        # 法向 geom1→geom2；物体在 geom1 侧则受力反向
        fsum += (-f_n * n) if o1 else (f_n * n)
    bid = adapter._resolve_body(obj)
    rid = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bid))
    mass = float(m.body_subtreemass[rid])
    fsum[2] -= 9.81 * mass      # 加自重后的净合力
    print(f"[dbg] FBALANCE[{obj}] net=({fsum[0]:+.2f},{fsum[1]:+.2f},"
          f"{fsum[2]:+.2f})N mass={mass*1000:.0f}g", flush=True)


def _sigmoid(x: float, scale: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(5.0 * x / scale, -30, 30))))


def _optimize(bounds: List[Tuple[float, float]],
              score_fn: Callable[[np.ndarray], float]
              ) -> Tuple[np.ndarray, float]:
    from scipy.optimize import differential_evolution
    res = differential_evolution(lambda x: -float(score_fn(np.asarray(x))),
                                 bounds, maxiter=DE_ITERS, popsize=DE_POP,
                                 tol=1e-3, polish=True, seed=0)
    return np.asarray(res.x, float), float(-res.fun)


def _container_score(adapter, obj: str, hand: Dict[str, Any], s: float,
                     joints, dof_adr, prior: List[np.ndarray]
                     ) -> Callable[[np.ndarray], float]:
    """容器壁夹法的打分函数（变量 r=site 沿开合轴偏移，d=site 低于沿顶）。

    约束（sigmoid 软门）：
      IK 残差 ≤ IK_TOL；
      内指尖向内射线 ≥ tip_r（内腔容得下手指）；
      外指尖向外射线 ≥ tip_r（外侧容得下手指）；
      与历史解距离 ≥ 2·tip_r（重规划求新解，非离散候选表）；
    目标：内外净空充裕、可操作度高、限位裕度大。
    """
    m, _ = _native_md(adapter)
    b = adapter.object_bounds(obj)
    cx, cy = float(b["center"][0]), float(b["center"][1])
    z_top, z_bot = float(b["z_top"]), float(b["z_bottom"])
    half_y, wall_h = float(b["half_y"]), float(b["z_top"] - b["z_bottom"])

    def score(x: np.ndarray) -> float:
        r, d = float(x[0]), float(x[1])
        site = np.array([cx, cy + s * r, z_top - d])
        inner, outer = _tip_points_world(adapter, site, hand, s)
        u_w = s * (hand_dir(adapter, hand))
        d_in = _clearance(adapter, inner, u_w * -1, obj, cap=0.15)
        d_out = _clearance(adapter, outer, u_w, obj, cap=0.25)
        q_sol, J, residual = _ik_solve(adapter, site, joints, dof_adr)
        w = _manipulability(J)
        margin = _limit_margin(m, q_sol, joints)
        g = 1.0
        g *= _sigmoid(IK_TOL - residual, IK_TOL)
        g *= _sigmoid(d_in - hand["tip_r"], hand["tip_r"] + 1e-6)
        g *= _sigmoid(d_out - hand["tip_r"], hand["tip_r"] + 1e-6)
        for p in prior:
            g *= _sigmoid(float(np.linalg.norm(site - p))
                          - 2 * hand["tip_r"], hand["tip_r"] + 1e-6)
        # 目标量（全部归一/无量纲）
        space = (min(d_in, d_out) / max(half_y, 1e-3))
        return float(g * (0.55 * min(space, 1.5) / 1.5
                          + 0.25 * min(w / 0.05, 1.0)
                          + 0.20 * margin))
    return score


def analytic_inclined_grasp(adapter, obj: str, prior=()):
    """分析型倾斜抓求解（任务无关，无离散候选表）。

    参数化 rim 方位 θ、腕倾角 β、抓深微调 d：内指钩沿内上侧、外指
    搭沿外上侧，闭爪方向同时含向内与向下分量。6D IK + 自接触/异
    物体接触为硬门，钩挂角/可操作度/限位裕度为目标。prior 为已失
    败 site 位置（排斥，重调用求新解）。返回与 GraspNet 候选同 schema
    的 dict，无可行解返回 None。
    """
    import mujoco
    m, _ = _native_md(adapter)
    joints = _arm_joints(m)
    dof_adr = [int(m.jnt_dofadr[j]) for j in joints]
    b = adapter.object_bounds(obj)
    cx, cy = float(b["center"][0]), float(b["center"][1])
    hx, hy = float(b["half_x"]), float(b["half_y"])
    z_rim = float(b["z_top"])
    hand = measure_hand(adapter)
    a = float(hand["a_open"])
    bd = float(hand["b_depth"])
    tr = float(hand["tip_r"])
    allowed = _site_grasp_geoms(adapter, obj)
    m2, sd = _scratch(adapter, -1.0)
    robot_geoms = _robot_geom_set(m2)
    s0, _re0, _, _rw0, _ew0 = _collision_status(m2, sd, robot_geoms)
    prior_arr = [np.asarray(p, float) for p in prior]

    def geometry(x: np.ndarray):
        th, beta, d = float(x[0]), float(x[1]), float(x[2])
        c, s = math.cos(th), math.sin(th)
        e = np.array([c, s, 0.0])
        up = np.array([0.0, 0.0, 1.0])
        r_th = hx * hy / math.sqrt((hy * c) ** 2 + (hx * s) ** 2 + 1e-12)
        P = np.array([cx + r_th * c, cy + r_th * s, z_rim])
        cb, sb = math.cos(beta), math.sin(beta)
        xs = cb * e + sb * up            # 开合轴（向外翘起）
        zs = sb * e - cb * up           # 接近轴（向下偏外）
        ys = np.cross(zs, xs)           # 切向
        Rs = np.column_stack((xs, ys, zs))
        site = P - (bd - d) * zs
        return site, Rs

    def score(x: np.ndarray) -> float:
        site, Rs = geometry(x)
        q_sol, J, rp, rr = _ik_solve(adapter, site, joints, dof_adr,
                                     target_rot=Rs)
        if float(rp) > IK_TOL or float(rr) > ROT_TOL:
            return 0.0
        for j in joints:
            sd.qpos[int(m2.jnt_qposadr[j])] = float(
                q_sol[int(m2.jnt_qposadr[j])])
        mujoco.mj_forward(m2, sd)
        sp, reg, _, _rw1, _ew1 = _collision_status(m2, sd, robot_geoms)
        if not sp <= s0:
            return 0.0
        if not set(reg) <= allowed:
            return 0.0
        g = 1.0
        for p in prior_arr:
            g *= _sigmoid(float(np.linalg.norm(site - p)) - 2 * tr, tr + 1e-6)
        hook = math.sin(float(x[1]))
        w = _manipulability(J)
        margin = _limit_margin(m2, q_sol, joints)
        return float(g * (0.5 * hook
                          + 0.3 * min(w / 0.05, 1.0)
                          + 0.2 * margin))

    bounds = [(0.0, 2.0 * math.pi),
              (0.0, math.pi / 2.0),
              (-2.0 * tr, 2.0 * tr)]
    x, s = _optimize(bounds, score)
    if float(s) <= 1e-6:
        return None
    site, Rs = geometry(x)
    if filter_site_config(adapter, obj, site, Rs) is None:
        return None
    return {"R": Rs, "site": site, "center": site,
            "width": 2.0 * a, "score": float(s)}


def analytic_rim_pinch(adapter, obj: str, prior=()):
    """分析型 rim 深捏求解（任务无关，无离散候选表、无优化器）。

    物理模型：碗类薄壁容器开口宽（>手开合），无法对握两壁；唯一通用
    抓取是 rim 捏——两指一内一外夹住 rim 立壁，闭爪法向水平、摩擦
    垂直承重。故：

    - 腕竖直向下（z=-up），开合轴沿 rim 径向 e（两指内外分布）；
    - 系统枚举 rim 方位 θ（椭圆极径给出接触点），每方位从 rim 顶沿
      向下按 tip_r 步距求位点，取**同时满足三者的最深可行位点**：
      实测壁面垂直度（曲腹法向出摩擦锥）、实测壁厚中面（空心壁闭合
      中面必须在壁厚中点）、内 pad 指尖腔体净空 ≥ tip_r（曲底容器
      深位指尖顶内底，闭爪/转运变成向内腔下压）。深度天然由
      bounds（壁高−指尖半径）界死，物理到底由执行期统计事件兜底；
    - 可行性 = 6D IK + 自接触无新对 + 接触仅目标（filter_site_config）；
    - 质量 = 深度 + 可操作度 + 关节限位裕度；
    - prior（已失败 site）方位在角分辨率内排斥，保证重调用求新方位。

    返回与 GraspNet 候选同 schema 的 dict；无可行解返回 None。
    """
    import mujoco
    m, _ = _native_md(adapter)
    joints = _arm_joints(m)
    b = adapter.object_bounds(obj)
    cx, cy = float(b["center"][0]), float(b["center"][1])
    hx, hy = float(b["half_x"]), float(b["half_y"])
    ztop, zbot = float(b["z_top"]), float(b["z_bottom"])
    hand = measure_hand(adapter)
    bd = float(hand["b_depth"])
    tr = float(hand["tip_r"])
    up = np.array([0.0, 0.0, 1.0])
    # 实测物体点云：壁面垂直度过滤的输入（曲腹判据，替代盲目取最深）
    from darwin.skills.perception.grasp import sample_object_point_cloud
    cloud = sample_object_point_cloud(adapter, obj, n_points=4096)
    if len(cloud) < 32:            # 采空则不过滤（保持旧行为）
        cloud = np.zeros((0, 3))
    m3, sd3 = _scratch(adapter, -1.0)
    j3 = _arm_joints(m3)
    a3 = [int(m3.jnt_qposadr[j]) for j in j3]
    n_az = 12
    used = {int(round(math.atan2(float(p[1]) - cy, float(p[0]) - cx)
                      / (2 * math.pi / n_az)) % n_az)
            for p in prior}
    best = None
    for k in range(n_az):
        if k in used:
            continue
        th = 2 * math.pi * k / n_az
        c, s = math.cos(th), math.sin(th)
        e = np.array([c, s, 0.0])
        r_th = hx * hy / math.sqrt((hy * c) ** 2 + (hx * s) ** 2 + 1e-12)
        Rs = np.eye(3)
        Rs[:, 0] = e
        Rs[:, 2] = -up
        Rs[:, 1] = np.cross(Rs[:, 2], Rs[:, 0])
        dmax = max(0.0, ztop - zbot - tr)
        depth, site = None, None
        nd = int(dmax / tr)
        # 深度策略：由深向浅取第一个同时满足三者的位点——
        # ①壁面垂直度（曲腹法向出摩擦锥，捏持必滑）；
        # ②实测壁厚中面（bbox 极径是外壁面，空心壁的闭合中面必须在
        #   壁厚中点，否则腕部偏置、单侧受力）；
        # ③内 pad 指尖下方腔体净空 ≥ tip_r：曲底容器（碗）内腔随深度
        #   收拢，深位指尖顶到内底曲面，闭爪/转运变成向内腔下压。
        #   指尖 = 壁中面点；下方点云必须按**径向**区分壁面延续（同半
        #   径，不构成障碍——pad 沿壁下滑）与腔底/曲腹（入于壁内缘，
        #   才是真障碍）。壁厚半宽 hw 由邻域径向偏移的 75 分位现测。
        # 深位夹壁面大、转运抗滑移好；③把深度限制在腔体允许的最深处，
        # 直壁深腔容器（杯）不受限。
        hw = tr
        for dd in range(nd, 0, -1):
            D = tr * dd
            Pr = np.array([cx + r_th * c, cy + r_th * s, ztop - D])
            # bbox 极径 r_th 是**外壁面**；空心壁判别用**薄 z 板**（半高
            # tr/2，排除顶/底面）——顶面点在径向上同样跨中点两侧，宽板
            # 会把实心盒角误判成空心壁、均值把指尖抬到顶沿（薄盒实测
            # 夹空）。薄板内两侧都有点 → 空心壁，邻域均值即壁中面；单
            # 侧/无点 → 实心面（壁厚实测 0）或点不足：实心面跳过该方位
            # （捏宽 0 不构成 rim 捏，外壁对握归 side_grip 定义域）；
            # 点不足保持标称壁点（证据不够不妄断）。
            nb = cloud[(np.linalg.norm(cloud - Pr, axis=1) < 1.5 * tr)
                       & (np.abs(cloud[:, 2] - Pr[2]) <= 0.5 * tr)] \
                if len(cloud) else np.zeros((0, 3))
            if len(nb) >= 6:
                if _sigmoid(_wall_verticality(cloud, Pr, 1.5 * tr)
                            - WALL_COS_MAX, WALL_GATE_SOFT) < 0.5:
                    continue       # 曲腹位点：法向出摩擦锥，捏持必滑
                offs = (nb - Pr) @ e   # 径向偏移：+外壁 / -内壁
                hw = max(float(np.percentile(np.abs(offs), 75)), 1e-4)
                # 壁厚门（两条，均相对指尖尺度 tr，手几何无关任务）：
                # ① 径向跨度——真薄立壁的邻域点紧贴壁中面（±壁半厚，
                #   几 mm）；盒角相邻两面投影出 [-4cm, +8mm] 的宽平分布，
                #   跨度 ≫ 指尖尺度即两面伪影（butter 角点方位实证），
                #   捏宽为零、pad 埋入实物，不构成 rim 捏；
                # ② 壁半厚 75 分位——达厘米级同理（厚块外壁对握归
                #   side_grip 定义域）。
                if offs.max() - offs.min() > 4.0 * tr or hw > tr:
                    continue
                if offs.min() < -0.2 * tr and offs.max() > 0.2 * tr:
                    Pr = np.asarray(nb.mean(axis=0), float)
                else:
                    # 薄板内无外侧壁点 = 实心面/凸棱：壁厚实测为 0，
                    # 捏宽 0 的两 pad 共面，一个 pad 埋入实物体内——
                    # 几何上不构成 rim 捏（object:6 butter 8 方位
                    # 全体产出 w=0 退化候选，approach 后闭爪捏空
                    # f_hold=0 实证；更早薄盒场景 8 连夹空）。实心
                    # 外壁夹持是 side_grip 的定义域，跳过该方位。
                    continue
            if len(cloud):
                tip = Pr          # 指尖 = 接触点（cand_site - bd·up）
                r_tip = float(np.linalg.norm(tip[:2]
                                             - np.array([cx, cy])))
                rad = np.linalg.norm(cloud[:, :2]
                                     - np.array([cx, cy]), axis=1)
                # 腔内点：径向入于壁内缘 2·hw（壁面延续被排除）
                below = cloud[(cloud[:, 2] < tip[2])
                              & (rad < r_tip - 2.0 * hw)]
                d_clear = float(tip[2] - below[:, 2].max()) if len(below) \
                    else float("inf")
                if d_clear < tr:
                    continue       # 指尖顶内底：闭爪下压，非捏持
            cand_site = Pr + bd * up
            f = filter_site_config(adapter, obj, cand_site, Rs)
            if f is not None:
                depth, site = D, cand_site
                break
        if site is None and len(cloud):
            # 浅沿兜底带（深度 ≤ 2·tip_r，手几何尺度）：外撇/曲腹容器
            # （碗）的 bbox 极径在沿口最外缘达成、沿口以下壁面内收，深
            # 层模型的壁中面假设整体失效（实测 hw 被腹曲率撑到 > tr、
            # 单侧分布判实心，12 方位全灭）；而沿口浅 pinch 的接触是
            # **骑沿**——外 pad 贴外壁、内 pad 悬于腔内，不依赖壁中面
            # 模型。兜底带只保留可证伪的物理门：
            # ① 沿口壁料证据：薄 z 板内 ≥6 点（实心盒顶沿方位薄板位
            #   于实体内部、无表面点，自动排除）；
            # ② 沿下腔体证据：径向入于壁内缘 2·tr 的下方点非空且净空
            #   ≥ tip_r——"rim" 的定义即开口沿且沿下有内腔；实心盒顶面
            #   无内腔采样点 → 排除（object:6 butter 的 w=0 退化不复
            #   现），浅碟沿下腔深不足指尖高同样排除。
            # 壁面垂直度/空心壁中面两门是深捏抗滑模型，骑沿接触不适
            # 用，跳过；pad 埋入退化由 filter_site_config 的埋入复验兜
            # 底（0.5·tip_r 穿透界，手几何比例）。
            for dd in (2, 1):
                D = tr * dd
                Pr = np.array([cx + r_th * c, cy + r_th * s, ztop - D])
                nb = cloud[(np.linalg.norm(cloud - Pr, axis=1) < 1.5 * tr)
                           & (np.abs(cloud[:, 2] - Pr[2]) <= 0.5 * tr)]
                if len(nb) < 6:
                    continue       # 沿口无壁料证据
                tip = Pr
                r_tip = float(np.linalg.norm(tip[:2]
                                             - np.array([cx, cy])))
                rad = np.linalg.norm(cloud[:, :2]
                                     - np.array([cx, cy]), axis=1)
                below = cloud[(cloud[:, 2] < tip[2])
                              & (rad < r_tip - 2.0 * tr)]
                if not len(below):
                    continue       # 沿下无腔体：不是 rim，是实心顶沿
                d_clear = float(tip[2] - below[:, 2].max())
                if d_clear < tr:
                    continue       # 沿下腔深不足指尖高，不可骑沿
                cand_site = Pr + bd * up
                f = filter_site_config(adapter, obj, cand_site, Rs)
                if f is not None:
                    depth, site = D, cand_site
                    break
        if site is None:
            continue
        q_sol, _rp, _rr = f
        for jj, vv in zip(j3, q_sol):
            sd3.qpos[int(m3.jnt_qposadr[jj])] = float(vv)
        mujoco.mj_forward(m3, sd3)
        Jfull = np.zeros((6, m3.nv))
        sid3 = mujoco.mj_name2id(m3, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
        mujoco.mj_jacSite(m3, sd3, Jfull[:3], Jfull[3:], sid3)
        J = Jfull[:, [int(m3.jnt_dofadr[j]) for j in j3]]
        w = _manipulability(J)
        margin = _limit_margin(
            getattr(m, "_model", m),
            np.asarray(q_sol, float),
            joints)
        q = 0.5 * min(depth / max(dmax, 1e-9), 1.0) \
            + 0.3 * min(w / 0.05, 1.0) + 0.2 * margin
        if best is None or q > best[0]:
            best = (q, site, Rs, np.asarray(q_sol, float))
    if best is None:
        return None
    _q, site, Rs, qv = best
    # _qv：filter_site_config 已验证的无碰撞 IK 解。调用方复用可跳过
    # 重复过滤——IK 多重启的部分热启动依赖 live 臂构型，同一位点在
    # 枚举时刻与使用时刻的过滤结果可能不一致（实测：枚举过、主循环
    # 重滤被拒，浪费整次尝试）。
    return {"R": Rs, "site": site, "center": np.asarray(site),
            "width": 0.0, "score": float(best[0]), "_qv": qv}



def analytic_side_grip(adapter, obj: str, prior=()):
    """实心/厚壁物体的侧面外夹（通用，无任务参数）。

    物理模型：物体无内腔（或壁厚 > 指尖可入深度），rim 捏不可行；
    唯一通用抓取是两指对握外壁——开合轴沿切向，接近轴水平指向中心。
    site 取物体轴心、中上部；指尖沿接近轴从外侧伸入，闭爪后 pad 内面
    夹住两侧 flank。z 取中点 + 1/6 高度，避免夹底推倒。
    宽度 = 物体沿开合轴的等效直径（2 × 椭圆切向极径），供 grasp
    做开合可行性过滤。

    枚举 12 方位 × 3 俯仰，每组合先按几何宽度门剔除不可夹方向，再验证
    6D IK + 碰撞仅目标/支撑面 + 自接触无新对。prior 方位排斥，保证重
    调用求新解。

    俯仰自由度（接近轴绕开合轴抬起）：φ>0 时手从外上方斜向进入，
    掌根抬离支撑面——低位/薄物体水平进入时掌根-支撑面干涉（薄物捏
    持的唯一运动学障碍）随俯仰角消解；指尖目标点不变。纯手几何，
    与任务无关。
    """
    import mujoco
    m, _ = _native_md(adapter)
    joints = _arm_joints(m)
    b = adapter.object_bounds(obj)
    cx, cy = float(b["center"][0]), float(b["center"][1])
    hx, hy = float(b["half_x"]), float(b["half_y"])
    ztop, zbot = float(b["z_top"]), float(b["z_bottom"])
    hand = measure_hand(adapter)
    bd = float(hand["b_depth"])
    n_az = 12
    # prior 传的是已试 site：site - center = bd·cosφ·e（bd 为 measure_hand
    # 现测的带符号 pad 偏移），b_depth 为负（Panda：pad 在 site 原点靠
    # 近腕侧 3.6mm）时偏移方向 = -e——按裸偏移算方位会把 θ+180° 的桶
    # 踢掉、原方位反复重选（object:1 三连同 site 实证）。乘 sign(bd)
    # 恢复 e 的真实方位。
    sgn = 1.0 if bd >= 0 else -1.0
    used = {int(round(math.atan2(sgn * (float(p[1]) - cy),
                                  sgn * (float(p[0]) - cx))
                      / (2 * math.pi / n_az)) % n_az)
            for p in prior}
    best = None           # 零接触（clean）候选中的最优——接受即停
    best_graze = None     # 擦碰候选的最优兜底（所有高度均无零接触时用）
    # 中上部：避免夹底推倒
    h_obj = ztop - zbot
    z_nom = (ztop + zbot) / 2.0 + h_obj / 6.0
    # 抓取高度爬升：名义高度（中上部）全部被工作空间边界（前臂压支撑
    # 面）拒绝时，向顶缘逐步尝试（步距 h_obj/6，pad 覆盖率下限 1/4 为
    # 统计约定，pad_h = 2·tip_r 为手几何事实）——远工作区边缘的薄物
    # 唯一可行构型常是顶缘捏持（object:6 butter 实证：名义高度全部
    # IK 解前臂压桌 ≥1.5mm，z 上移 2mm 后出现 25 个零接触解）。
    z_cap = ztop - 0.5 * float(hand["tip_r"])
    zs_list, _zz = [], z_nom
    while _zz <= z_cap + 1e-9:
        zs_list.append(_zz)
        _zz += h_obj / 6.0
    if not zs_list:
        zs_list = [z_nom]
    elif zs_list[-1] < z_cap - 1e-9:
        zs_list.append(z_cap)
    a_reach = float(hand["a_open"]) + float(hand["tip_r"])
    up = np.array([0.0, 0.0, 1.0])
    for zi, z_site in enumerate(zs_list):
        for k in range(n_az):
            # prior 只对应名义高度已试的方位；爬升后的新高度是另一组
            # 候选（同方位不同 z 可行性不同），全部重试。
            if zi == 0 and k in used:
                continue
            th = 2 * math.pi * k / n_az
            c, s = math.cos(th), math.sin(th)
            e = np.array([c, s, 0.0])        # 径向（手在 +e 外侧逼近）
            t = np.array([-s, c, 0.0])       # 切向 = 开合轴
            # 物体沿开合轴的半宽（椭圆极径）：可达半开距 a_reach = a_open +
            # tip_r 必须能触到两侧 flank——pad 表面比球心内缩 tip_r，宽至
            # 多 tip_r/侧的物体仍被 pad 挤压夹持（接触事件停止接近，纯几
            # 何可行性，与 grasp 的宽度门同理）。
            r_t = hx * hy / math.sqrt((hy * s) ** 2 + (hx * c) ** 2 + 1e-12)
            if r_t > a_reach:
                continue
            for pitch in (0.0, 0.6, 1.0):
                cp, sp = math.cos(pitch), math.sin(pitch)
                # 右手系正交姿态：开合轴 = 切向，接近轴 = -e 绕 t 抬 pitch
                Rs = np.eye(3)
                Rs[:, 0] = t
                Rs[:, 2] = -cp * e + sp * up
                Rs[:, 1] = np.cross(Rs[:, 2], Rs[:, 0])
                # site：GRIP_SITE 沿接近轴回拉 bd（指尖目标点 = 轴心,
                # z_site）；俯仰把 GRIP_SITE 抬升 bd·sin(pitch)
                site = np.array([cx + bd * cp * c, cy + bd * cp * s,
                                 z_site + bd * sp])
                # prior 的精确去重（任意高度）：方位桶排斥只管名义高度
                # （爬升后同方位不同 z 是另一组候选，故意重试），但若
                # 上一次返回的候选本身就在爬升高度，同 site 会被再返回
                # 一次（object:9 OJ 实证：#0#1 同一 z_cap 位点重复），
                # 白烧 try 预算。site 级近重复（≤IK_TOL）全部跳过。
                if any(float(np.linalg.norm(site - np.asarray(p, float)))
                       <= IK_TOL for p in prior):
                    continue
                f = filter_site_config(adapter, obj, site, Rs)
                if f is None:
                    continue
                q_sol, _rp, _rr = f
                # 可操作度与裕度
                m3, sd3 = _scratch(adapter, -1.0)
                j3 = _arm_joints(m3)
                for jj, vv in zip(j3, q_sol):
                    sd3.qpos[int(m3.jnt_qposadr[jj])] = float(vv)
                mujoco.mj_forward(m3, sd3)
                Jfull = np.zeros((6, m3.nv))
                sid3 = mujoco.mj_name2id(m3, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
                mujoco.mj_jacSite(m3, sd3, Jfull[:3], Jfull[3:], sid3)
                J = Jfull[:, [int(m3.jnt_dofadr[j]) for j in j3]]
                w = _manipulability(J)
                margin = _limit_margin(getattr(m, "_model", m),
                                       np.asarray(q_sol, float), joints)
                # 质量：可操作度 + 限位裕度 + 开合余量（越窄的轴向越好夹）
                # + 俯仰奖励（斜向进入掌根 clearance 更好，物理上更稳）
                q = 0.4 * min(w / 0.05, 1.0) + 0.3 * margin \
                    + 0.3 * (1.0 - r_t / max(a_reach, 1e-9)) \
                    + 0.1 * sp
                # clean = 间隙感知（margin 探针）：非 pad 机器人 geom 对
                # 一切 env geom 间隙 ≥ 0.5·tip_r。零接触判据只管穿透
                # （dist<0），间隙 0.1mm 与 1cm 无别；物理伺服的毫米级
                # 下垂/跟踪误差会吃掉亚毫米间隙——object:9 OJ 实证：几
                # 何零接触的腕-肩间隙在接近段被压上瓶肩水平面（nz≥0.7）
                # 连环淘汰全部候选。把机器人 geom margin 暂设为
                # 0.5·tip_r 再生成接触（dist < margin 即报接触），间隙
                # 不足的候选归 grazing，枚举转向更低/更斜的构型。
                # pad∩目标装配豁免：pad 按设计贴近 flank（b_depth 偏
                # 移）。0.5·tip_r 与支撑面深度门同一手几何比例约定。
                _rg3 = _robot_geom_set(m3)
                _pads3 = _pad_geom_set(m3)
                _tgt3 = _site_grasp_geoms(adapter, obj)
                _marg = 0.5 * float(hand["tip_r"])
                _mg0 = np.asarray(m3.geom_margin, float).copy()
                for _g in _rg3:
                    m3.geom_margin[_g] = max(float(_mg0[_g]), _marg)
                try:
                    mujoco.mj_forward(m3, sd3)
                    _clean = True
                    _worst = 1.0     # 非豁免接触的最小（真实）间隙/穿透
                    for _ci in range(sd3.ncon):
                        _c = sd3.contact[_ci]
                        _g1, _g2 = int(_c.geom1), int(_c.geom2)
                        _r1 = _g1 in _rg3
                        _r2 = _g2 in _rg3
                        if not (_r1 or _r2):
                            continue
                        _rgm = _g1 if _r1 else _g2
                        _eg = _g2 if _r1 else _g1
                        if _rgm in _pads3 and _eg in _tgt3:
                            continue           # pad 贴近目标 flank：设计如此
                        _clean = False
                        if float(_c.dist) < _worst:
                            _worst = float(_c.dist)
                finally:
                    m3.geom_margin[:] = _mg0
                _cand = (q, site, Rs, 2 * r_t, np.asarray(q_sol, float))
                if _clean:
                    if best is None or q > best[0]:
                        best = _cand
                elif _worst >= -0.5 * _marg:
                    # 浅擦碰兜底（穿透 ≥ -0.25·tip_r，与支撑面深度门同
                    # 比例）：全无零间隙候选时仍可用。深穿透候选直接
                    # 跳过——object:9 OJ 实证：腕部穿瓶肩 -8.6mm 的候
                    # 选曾被兜底返回，pregrasp/approach 必败，白烧 try。
                    if best_graze is None or q > best_graze[0]:
                        best_graze = _cand
        if best is not None:
            # 该高度已出零接触候选：名义高度优先，不再爬升
            break
    if best is None:
        best = best_graze
    if best is None:
        return None
    _q, site, Rs, width, qv = best
    return {"R": Rs, "site": site, "center": np.asarray(site),
            "width": float(width), "score": float(best[0]), "_qv": qv}


def hand_dir(adapter, hand: Dict[str, Any]) -> np.ndarray:
    """site 系开合轴 → 世界单位向量。"""
    import mujoco
    m, _ = _native_md(adapter)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    sR = np.asarray(adapter.mj_data.site_xmat[sid], float).reshape(3, 3)
    return sR @ hand["u_site"]


def _container_bounds(hand: Dict[str, Any], b: Dict[str, Any]
                      ) -> List[Tuple[float, float]]:
    a, hy = hand["a_open"], float(b["half_y"])
    # 外指越过外壁：r+a ≥ hy；内指不出对侧壁：r-a ≥ -hy
    r_lo = max(0.0, hy - a, a - hy)
    r_hi = hy + a
    # 指尖在腔内：0 ≤ d，且指尖不碰底：d+b_depth ≤ wall_h
    d_lo = 0.0
    d_hi = max(float(b["z_top"] - b["z_bottom"]) - hand["b_depth"], IK_TOL)
    return [(r_lo, r_hi), (d_lo, d_hi)]


def _solid_score(adapter, obj: str, hand: Dict[str, Any], joints, dof_adr
                 ) -> Callable[[np.ndarray], float]:
    """实心物体打分（唯一变量 d=site 低于顶面）：居中，两指在物体两侧，
    指尖向外两侧射线净空、IK 可达、可操作度/裕度最大化。"""
    m, _ = _native_md(adapter)
    b = adapter.object_bounds(obj)
    cx, cy = float(b["center"][0]), float(b["center"][1])
    z_top, thick = float(b["z_top"]), float(b["z_top"] - b["z_bottom"])

    def score(x: np.ndarray) -> float:
        d = float(x[0])
        site = np.array([cx, cy, z_top - d])
        q_sol, J, residual = _ik_solve(adapter, site, joints, dof_adr)
        u_w = hand_dir(adapter, hand)
        tips = _tip_points_world(adapter, site, hand, 1.0)
        d1 = _clearance(adapter, tips[1], u_w, obj, cap=0.2)
        d2 = _clearance(adapter, tips[0], -u_w, obj, cap=0.2)
        g = _sigmoid(IK_TOL - residual, IK_TOL)
        g *= _sigmoid(min(d1, d2) - hand["tip_r"], hand["tip_r"] + 1e-6)
        w = _manipulability(J)
        margin = _limit_margin(m, q_sol, joints)
        return float(g * (0.55 * min(d1, d2) / 0.1
                          + 0.25 * min(w / 0.05, 1.0)
                          + 0.20 * margin))
    return score


def solve_grasp_target(adapter, obj: str, rank: int = 0,
                       prior: Optional[List[np.ndarray]] = None
                       ) -> Optional[Dict[str, Any]]:
    """抓取目标通用求解：容器两侧各跑一次连续约束优化，取优；
    rank>0 时对 prior 解加排斥（求"下一个不同的可行解"）。
    实体走单变量优化。返回 {site, side, score} 或 None。"""
    prior = list(prior or [])
    hand = measure_hand(adapter)
    m, _ = _native_md(adapter)
    joints = _arm_joints(m)
    dof_adr = [int(m.jnt_dofadr[j]) for j in joints]

    if adapter._is_container(obj):
        b = adapter.object_bounds(obj)
        bounds = _container_bounds(hand, b)
        if bounds[0][0] > bounds[0][1] or bounds[1][1] <= bounds[1][0]:
            return None
        cands = []
        for s in (-1.0, 1.0):
            fn = _container_score(adapter, obj, hand, s, joints,
                                  dof_adr, prior)
            x, sc = _optimize(bounds, fn)
            site = np.array([float(b["center"][0]),
                             float(b["center"][1]) + s * float(x[0]),
                             float(b["z_top"]) - float(x[1])])
            cands.append({"site": site, "side": s, "score": sc})
        cands.sort(key=lambda c: -c["score"])
        return cands[0] if cands and cands[0]["score"] > 0 else None

    b = adapter.object_bounds(obj)
    if hand["a_open"] < float(b["half_y"]) - IK_TOL:
        return None  # 物体宽于最大开指：几何上不可夹
    bounds = [(0.0, max(float(b["z_top"] - b["z_bottom"])
                        - hand["b_depth"], IK_TOL))]
    x, sc = _optimize(bounds, _solid_score(adapter, obj, hand,
                                           joints, dof_adr))
    site = np.array([float(b["center"][0]), float(b["center"][1]),
                     float(b["z_top"]) - float(x[0])])
    return {"site": site, "side": 0.0, "score": sc} if sc > 0 else None


# ==================== 统计伺服 ====================

class _ServoStats:
    """窗口回归：末端距离序列的斜率/标准误/残差σ；关节/力基线同理。"""

    def __init__(self):
        self.ds: List[Tuple[int, float]] = []
        self.qs: List[Tuple[int, np.ndarray]] = []
        self.fs: List[float] = []

    def add(self, t, d, q, f):
        self.ds.append((t, d))
        self.qs.append((t, np.asarray(q, float)))
        self.fs.append(float(f))

    def fit(self):
        arr = np.array(self.ds[-WIN:], float)
        x, y = arr[:, 0], arr[:, 1]
        slope, icept = np.polyfit(x, y, 1)
        resid = y - (slope * x + icept)
        sigma = float(resid.std()) + 1e-6
        den = float(np.sqrt(np.sum((x - x.mean()) ** 2)))
        se = sigma / max(den, 1e-9)
        # 关节
        qa = np.array([q for _, q in self.qs[-WIN:]], float)
        qmove = float(np.sum(np.abs(qa[-1] - qa[0])))
        # 力基线（近窗口中位数 = 自由空间背景）
        farr = np.asarray(self.fs[max(0, len(self.fs) - WIN * 3):], float)
        if farr.size == 0:
            farr = np.zeros(1)
        base = float(np.median(farr))
        sigf = float(farr.std()) + 1e-6
        return {"slope": float(slope), "se": se, "sigma": sigma,
                "qmove": qmove, "fbase": base, "sigf": sigf}


def _top_overlap(obj_c, obj_h, env_kind, env_c, env_ax, env_ay,
                 env_r) -> bool:
    """顶视 footprint 二维相交判定（物体为轴对齐矩形，保守）。

    env_kind="rect"：有向矩形（中心 + 两条半边轴向量），SAT 精确判定；
    env_kind="circle"：圆（中心 + 半径），矩形最近点判定；env_kind=
    "aabb"：世界 AABB（兜底，仅用于无精确顶视原语的 geom 类型）。
    """
    o_lo = obj_c - obj_h
    o_hi = obj_c + obj_h
    if env_kind == "aabb":
        return not (env_ax[0] > o_hi[0] or env_ay[0] < o_lo[0]
                    or env_ax[1] > o_hi[1] or env_ay[1] < o_lo[1])
    if env_kind == "circle":
        q = np.clip(env_c, o_lo, o_hi)
        return float(np.dot(q - env_c, q - env_c)) <= env_r * env_r
    # rect vs rect SAT（轴 = 物体两轴 + env 两轴）
    axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0]),
            env_ax / max(float(np.linalg.norm(env_ax)), 1e-12),
            env_ay / max(float(np.linalg.norm(env_ay)), 1e-12)]
    corners = [obj_c + s1 * np.array([obj_h[0], 0.0])
               + s2 * np.array([0.0, obj_h[1]])
               for s1 in (-1.0, 1.0) for s2 in (-1.0, 1.0)]
    ecorners = [env_c + s1 * env_ax + s2 * env_ay
                for s1 in (-1.0, 1.0) for s2 in (-1.0, 1.0)]
    for a in axes:
        if abs(float(a[0])) + abs(float(a[1])) < 1e-12:
            continue
        o_proj = [float(np.dot(c, a)) for c in corners]
        e_proj = [float(np.dot(c, a)) for c in ecorners]
        if min(o_proj) > max(e_proj) or min(e_proj) > max(o_proj):
            return False
    return True


def _clearance_height(adapter, obj: str) -> float:
    """物体 footprint 内所有环境 geom 的最高顶面 + IK_TOL。

    通用"越过周边障碍"几何：场景各 env geom（排除物体自身子树与机器
    人）的**顶视 footprint** 与物体 xy AABB 相交者的顶面最大值。
    footprint 用精确二维原语：box → 有向矩形（世界旋转矩阵两列×
    半尺寸，SAT 精确判定）；sphere/cylinder/capsule → 圆（半径 =
    r + 长轴在水平面投影份额，保守包络）；其余类型退回世界 AABB
    （保守兜底）。不可用世界 AABB 相交代替 footprint：旋转薄板
    （半开柜门、斜靠面板）的世界 AABB 远大于其真实占空，与物体
    AABB 毫米级擦边的 AABB 相交会把"无需越障"误判成"须抬升到板
    顶"——纯 AABB 伪重叠对毫米级物理噪声双稳（goal:4 实证：同一
    scene cl_z 在 0.98/1.17 间跳变，后者抬升目标逼近工作空间顶，
    竖直伺服在差 4mm 处停滞判 reach_limit、place_at 直接败退）。
    物体底面高于该值时，任何腕部转动/水平 transit 都不会与周边
    障碍磕碰。纯几何查询，无任务信息。
    """
    import mujoco
    m, sd = _scratch(adapter, +1.0)
    mujoco.mj_forward(m, sd)
    b = adapter.object_bounds(obj)
    cxy = np.asarray(b["center"][:2], float)
    h2 = np.array([float(b["half_x"]), float(b["half_y"])])
    own = _site_grasp_geoms(adapter, obj)
    robot = _robot_geom_set(m)
    top = -np.inf
    for g in range(m.ngeom):
        if g in own or g in robot:
            continue
        if int(m.geom_type[g]) == mujoco.mjtGeom.mjGEOM_PLANE:
            continue  # 无限平面（地面）：非可越过障碍
        center = np.asarray(sd.geom_xpos[g], float)
        rmat = np.asarray(sd.geom_xmat[g], float).reshape(3, 3)
        sz = np.asarray(m.geom_size[g], float)
        gt = int(m.geom_type[g])
        # 各 geom 类型的局部半尺寸（MuJoCo 约定）：
        # box=size3, sphere=(r,r,r), capsule/cylinder=(r,half_len,r),
        # ellipsoid=size3；mesh 无 size 语义，局部 AABB 由顶点现算。
        hloc = None
        if gt == mujoco.mjtGeom.mjGEOM_BOX:
            h = sz[:3].copy()
        elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
            h = np.array([sz[0]] * 3)
        elif gt in (mujoco.mjtGeom.mjGEOM_CAPSULE,
                    mujoco.mjtGeom.mjGEOM_CYLINDER):
            h = np.array([sz[0], sz[1], sz[0]])
        elif gt == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            h = sz[:3].copy()
        elif gt == mujoco.mjtGeom.mjGEOM_MESH:
            mid = int(m.geom_dataid[g])
            adr = int(m.mesh_vertadr[mid])
            nv = int(m.mesh_vertnum[mid])
            vv = np.asarray(m.mesh_vert[adr:adr + nv], float).reshape(nv, 3)
            qw, qx, qy, qz = [float(x) for x in m.mesh_quat[mid]]
            Rm = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz),
                 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz),
                 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx),
                 1 - 2 * (qx * qx + qy * qy)]])
            vv = vv @ Rm.T + np.asarray(m.mesh_pos[mid], float)
            hloc = (vv.max(axis=0) - vv.min(axis=0)) / 2.0
            h = hloc.copy()
        else:
            h = np.array([max(sz[0], sz[1])] * 3)
        # 世界 AABB 顶面（z 半长 = |R|·h 的 z 分量）
        hw = np.abs(rmat) @ h
        z_top = float(center[2] + hw[2])
        # 顶视 footprint 精确原语
        if gt in (mujoco.mjtGeom.mjGEOM_BOX, mujoco.mjtGeom.mjGEOM_MESH):
            # box 用 size 半长；mesh 用顶点 AABB 半长（保守有界盒），
            # 均按世界旋转投影成顶视有向矩形。柜门/家具面板多为
            # mesh，其世界 AABB 远大于真实占空，是 AABB 伪重叠的
            # 主要来源。
            kind = "rect"
            ec = center[:2]
            eax = rmat[:2, 0] * h[0]
            eay = rmat[:2, 1] * h[1]
            er = 0.0
        elif gt in (mujoco.mjtGeom.mjGEOM_SPHERE,
                    mujoco.mjtGeom.mjGEOM_CYLINDER,
                    mujoco.mjtGeom.mjGEOM_CAPSULE):
            kind = "circle"
            ec = center[:2]
            eax = eay = None
            # 长轴（局部 y）在水平面的投影份额：直立→r，水平→r+半长
            er = float(h[0] + h[1] * np.linalg.norm(rmat[:2, 1]))
        else:
            kind = "aabb"
            ec = center[:2]
            eax = np.array([center[0] - hw[0], center[1] - hw[1]])
            eay = np.array([center[0] + hw[0], center[1] + hw[1]])
            er = 0.0
        if not _top_overlap(cxy, h2, kind, ec, eax, eay, er):
            continue
        top = max(top, z_top)
    if not np.isfinite(top):
        return float(b["z_bottom"])
    return top + IK_TOL


def _site_step_envelope(adapter) -> float:
    """当前构型下 site 在一个 EXEC_DQ 归一化细分内的最坏位移 ×2
    （= execute 的跟踪门几何）。用于放置段"指令越过接触"的穿透量：
    把目标设在接触面下方该值，物理接触必在到达目标前发生，接触
    事件先于收敛/预算裁决。纯 Jacobian 推导。"""
    import mujoco
    m2, sd = _scratch(adapter, +1.0)
    joints = _arm_joints(m2)
    sid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    mujoco.mj_forward(m2, sd)
    Jp = np.zeros((3, m2.nv))
    mujoco.mj_jacSite(m2, sd, Jp, None, sid)
    worst = 0.0
    for j in joints:
        dof = int(m2.jnt_dofadr[j])
        span = float(m2.jnt_range[j][1] - m2.jnt_range[j][0])
        worst = max(worst,
                    float(np.linalg.norm(Jp[:, dof])) * EXEC_DQ * span)
    return float(2.0 * worst)


@_timed
def serve(adapter, target, *, gripper: float = 0.0, k: float = K_SERVO,
          verify: Optional[Callable[[np.ndarray], Tuple[bool, dict]]] = None,
          budget: Optional[int] = None,
          target_rot: Optional[np.ndarray] = None,
          kr: float = K_SERVO) -> Optional[Dict[str, Any]]:
    """伺服到 target（给 target_rot 时为位姿伺服）。到位判据：
    verify(实测末端) 通过（几何约束复验），或位置残差 ≤ IK_TOL 且
    姿态残差 ≤ ROT_TOL（残差进入 3σ 噪声包络内同样放行）。无进展
    判据：窗口斜率 t<2（统计上与零无异），连续 3 窗则按物理证据
    分类死因：接触力超基线 3σ → contact_blocked；关节不动 →
    ik_unreachable。**不设预测步数预算**：慢而持续的进展必须走完，
    预算误杀会把物理上 3mm 内的接触事件截断；真实物理止挡由上述
    统计分类保证终止。显式 budget 参数仅保留给调用方硬资源限制。
    """
    tgt = np.asarray(target, float)
    Rt = None if target_rot is None else np.asarray(target_rot, float)
    st = _ServoStats()
    m0, _ = _native_md(adapter)
    arm_adrs = [int(m0.jnt_qposadr[j]) for j in _arm_joints(m0)]
    stationary = 0
    sigma: Optional[float] = None

    # 米制等价组合：姿态误差×实测指尖杠杆臂
    hs = _tip_lever(adapter) if Rt is not None else 0.0
    t = 0
    while True:
        end = _eef(adapter)
        d_pos = float(np.linalg.norm(end - tgt))
        d_rot = 0.0
        if Rt is not None:
            Rc = np.asarray(adapter.get_site_rot(GRIP_SITE), float)
            d_rot = float(np.linalg.norm(_rot_err(Rt, Rc)))
        fnow = float(adapter.gripper_contact_force())
        qnow = np.asarray(adapter.mj_data.qpos, float)[arm_adrs]
        st.add(t, d_pos + hs * d_rot, qnow, fnow)

        if verify is not None:
            ok, vinfo = verify(end)
            if ok:
                if os.environ.get("LIBERO_DEBUG"):
                    print(f"[dbg] serve_ok ticks={t} d={d_pos:.4f} "
                          f"dr={d_rot:.3f}", flush=True)
                return None
        else:
            pos_room = max(IK_TOL, CONF_K * (sigma or 0.0))
            if d_pos < pos_room and d_rot <= ROT_TOL:
                if os.environ.get("LIBERO_DEBUG"):
                    print(f"[dbg] serve_ok ticks={t} d={d_pos:.4f} "
                          f"dr={d_rot:.3f}", flush=True)
                return None

        if len(st.ds) >= WIN and t % WIN == WIN - 1:
            ft = st.fit()
            sigma = ft["sigma"]
            tstat = -ft["slope"] / max(ft["se"], 1e-9)
            # 绝对进展地板（同 _approach_to_grasp）：饱和伺服低噪声爬行
            # 对 t 检验"统计显著"，窗口闭距不足 0.25·IK_TOL 亦计静止。
            progressed = st.ds[-WIN][1] - st.ds[-1][1]
            if tstat < TSTAT or progressed < 0.25 * IK_TOL:
                stationary += 1
                if ft["fbase"] + CONF_K * ft["sigf"] <= fnow \
                        and fnow > IK_TOL:
                    return _fail(
                        f"contact_blocked @ {np.round(tgt,3).tolist()}",
                        "contact_blocked", end=end.tolist(),
                        target=tgt.tolist(),
                        contact_n=round(fnow, 2))
                if ft["qmove"] < IK_TOL:
                    return _fail(
                        f"reach_limit @ {np.round(tgt,3).tolist()}",
                        "ik_unreachable", end=end.tolist(),
                        target=tgt.tolist(),
                        d_pos=round(d_pos, 4), d_rot=round(d_rot, 4),
                        force=round(fnow, 2))
                if stationary >= STATIONARY_MAX:
                    return _fail(
                        f"ik_stalled @ {np.round(tgt,3).tolist()}",
                        "ik_unreachable", end=end.tolist(),
                        target=tgt.tolist())
            else:
                stationary = 0

        if budget is not None and t > int(budget):
            return _fail(f"slow_budget @ {np.round(tgt,3).tolist()}",
                         "timeout", end=_eef(adapter).tolist(),
                         target=tgt.tolist())
        adapter.servo_step(GRIP_SITE, tgt, gripper=gripper, k=k, vcap=VCAP,
                           target_rot=Rt, kr=kr)
        t += 1


def goto(adapter, target, *, gripper: float = 0.0, k: float = K_SERVO,
         tol: Optional[float] = None,
         timeout: Optional[int] = None,
         target_rot: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
    """兼容接口：tol 作为几何复验容差（缺省走噪声包络），
    timeout 作为显式预算（缺省由进展率推导），target_rot 给定时
    同时伺服姿态。"""
    if tol is None:
        return serve(adapter, target, gripper=gripper, k=k, budget=timeout,
                     target_rot=target_rot)

    def _verify(end):
        return float(np.linalg.norm(np.asarray(end, float)
                                   - np.asarray(target, float))) <= tol, {}
    return serve(adapter, target, gripper=gripper, k=k, verify=_verify,
                 budget=timeout, target_rot=target_rot)


def execute_corridor(adapter, legs: List[np.ndarray], hand: Dict[str, Any],
                     gripper: float,
                     verify_last: Optional[Callable] = None
                     ) -> Optional[Dict[str, Any]]:
    """执行走廊：前序腿噪声包络收敛（巡航增益），末腿精细增益+
    约束复验。"""
    for i, leg in enumerate(legs):
        last = i == len(legs) - 1
        r = serve(adapter, leg, gripper=gripper,
                  k=K_FINE if last else K_SERVO,
                  verify=verify_last if last else None)
        if r is not None:
            return r
    return None


def _reach_site(adapter, site: np.ndarray,
                Rs: Optional[np.ndarray] = None,
                tgt_geoms: Optional[set] = None,
                zone: Optional[np.ndarray] = None,
                preshape: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """通用位点到达（接触允许）：抓取管线的 TAMP 四段原语搬到任意
    目标位点——预抓半球后退（全手零碰撞）→ RRT-Connect 关节路径 →
    流式执行 → 事件驱动接近（几何到达/接触/止挡三通用事件）。

    与 corridor_plan 的区别：corridor 的终点门禁要求**全手包络净空**，
    只适用于自由空间目标；此处终点是**接触点**（把手/旋钮），碰撞
    谓词在接近段被接触事件取代。姿态取当前 site 旋转（OSC 锁定）；
    Rs 显式给出时按调用侧求解的接近姿态（如 prismatic 正面捏夹）。
    tgt_geoms 给定时（articulate），接近段只认这些 geom 的接触。
    zone（[[lo],[hi]] AABB）给定时，接触还必须落在作用 geom 邻域内：
    夹具 body 可能把多个旋钮/外壳 geom 焊在同一刚体（goal:7 灶台），
    geom 级判定会把邻居擦碰误判到达，按接触点位置再分一遍。
    preshape（半开口）给定时，路径执行后先在预达点闭环收爪到该宽度
    再接近（旋钮薄耳片捏夹的进入构型：全张指身覆盖邻居旋钮排，goal:7
    实证所有可达构型固有碰撞；窄手是唯一通路），接近段保持该宽度。
    """
    import mujoco
    m, live = _native_md(adapter)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    if Rs is None:
        Rs = np.asarray(live.site_xmat[sid], float).reshape(3, 3)
    Rs = np.asarray(Rs, float).reshape(3, 3)
    inflation = _transit_inflation(adapter)
    pre = pregrasp_point(adapter, site, Rs)
    if pre is None:
        return _fail(f"目标位点无可行预达构型（碰撞/IK）: "
                     f"{np.round(site, 3).tolist()}", "ik_unreachable")
    pre_site, _q = pre
    path = plan_arm_path(adapter, pre_site, grip=-1.0, target_rot=Rs,
                         inflation=inflation)
    if path is None:
        return _fail(f"目标位点无可达无碰撞关节路径: "
                     f"{np.round(site, 3).tolist()}", "ik_unreachable")
    r = execute_arm_path(adapter, path, grip=-1.0, strict=False)
    if r is not None:
        return r
    if preshape is not None:
        _preshape(adapter, preshape)
    return _approach_to_grasp(adapter, site, Rs, tgt_geoms=tgt_geoms,
                              zone=zone,
                              grip=0.0 if preshape is not None else -1.0)


@_timed
def execute_arm_path(adapter, path, *, grip: float,
                     verify_last: Optional[Callable] = None,
                     k: float = K_SERVO, strict: bool = True,
                     allow_contact: bool = False,
                     watch_obj: Optional[str] = None):
    """流式执行关节路径：按 EXEC_DQ 归一化间距细分构型，逐 FK 点以
    跟踪门控闭环（末端进门再前进）。每个门的通过还必须满足 **live
    碰撞复验**：OSC 的实际零空间构型可能与 scratch 规划构型不同，
    故在真实物理状态上重算接触——自接触新对拒绝；与 watch_obj 之外
    的任何异物体接触拒绝；watch_obj 的擦碰允许，但其整体位移必须
    ≤ IK_TOL（弹性擦碰不推移物体是允许的，推/滑动即拒绝）。
    allow_contact=True（持物 transit）时起点 tether 接触允许。

    走完末端：
    - strict=True：按 verify/几何零位做精细闭环收敛（适用于放置等
      必须精确到点的动作）；
    - strict=False：流结束即放行（OSC 稳态有数 mm 跟踪偏置），残余
      距离与姿态由后续事件驱动原语（如接近）闭环消除。
    """
    import mujoco
    joints = _arm_joints(_native_md(adapter)[0])
    m, sd = _scratch(adapter, grip)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    lo = np.array([float(m.jnt_range[j][0]) for j in joints])
    hi = np.array([float(m.jnt_range[j][1]) for j in joints])
    span = np.maximum(hi - lo, 1e-6)

    def fk_site(q_arm):
        for j, v in zip(joints, q_arm):
            sd.qpos[int(m.jnt_qposadr[j])] = float(v)
        mujoco.mj_forward(m, sd)
        return (np.asarray(sd.site_xpos[sid], float).copy(),
                np.asarray(sd.site_xmat[sid], float).reshape(3, 3).copy(),
                np.asarray(q_arm, float).copy())

    pts = [fk_site(np.asarray(path[0], float))]
    for qa, qb in zip(path[:-1], path[1:]):
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        dnrm = float(np.max(np.abs((qb - qa) / span)))
        n = max(int(np.ceil(dnrm / EXEC_DQ)), 1)
        for kk in range(1, n + 1):
            pts.append(fk_site(qa + (qb - qa) * kk / n))

    # live 起点接触基线（transit 时两者通常为空；持物时 reg0=tether）
    _lm, _lsd = _native_md(adapter)
    _rg = _robot_geom_set(_lm)
    _live_self0, _live_reg0, _, _live_rw0, _live_ew0 = _collision_status(_lm, _lsd, _rg)
    _live_reg0 = set(_live_reg0)
    _live_tol = -0.5 * float(measure_hand(adapter)["tip_r"])

    _watch_geoms = (set(_site_grasp_geoms(adapter, watch_obj))
                    if watch_obj is not None else set())
    _watch_c0 = (np.asarray(adapter.object_bounds(watch_obj)["center"], float)
                 if watch_obj is not None else None)
    # 持物模式：记录物体中心在 site 系下的相对位姿（刚性夹持不变量），
    # 滑移上限=实测半开口 a_open——超出即脱手。世界系位移对持物无意义。
    # object_bounds["center"] 只有 xy，3D 中心 z 取包围盒上下中点
    # （与 place_at 的 rel 口径一致）。
    _watch_rel0 = None
    _slip_tol = 0.0
    if allow_contact and watch_obj is not None:
        _b0 = adapter.object_bounds(watch_obj)
        _sp0 = np.asarray(adapter.get_site_pos(GRIP_SITE), float)
        _sR0 = np.asarray(adapter.get_site_rot(GRIP_SITE), float)
        _c3 = np.array([_b0["center"][0], _b0["center"][1],
                        (_b0["z_top"] + _b0["z_bottom"]) / 2.0])
        _watch_rel0 = _sR0.T @ (_c3 - _sp0)
        _slip_tol = float(measure_hand(adapter)["a_open"])

    def _live_safe() -> bool:
        mm, ds = _native_md(adapter)
        sp, reg, _, rew, _ew = _collision_status(mm, ds, _robot_geom_set(mm))
        dbg = os.environ.get("LIBERO_DEBUG")
        if not set(sp) <= _live_self0:
            new_sp = set(sp) - _live_self0
            if dbg:
                import mujoco
                def _gname(g):
                    return str(mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM, g) or g)
                def _bname(g):
                    return str(mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY,
                                                 int(mm.geom_bodyid[g])) or "?")
                desc = ["%s(%s)~%s(%s)" % (_gname(a), _bname(a), _gname(b), _bname(b))
                        for a, b in sorted(new_sp)[:3]]
                print("LIVECHK code=S new=%s" % desc)
            return False
        new_reg = set(reg) - _live_reg0
        # 浅接触静态支撑面不算碰撞事件（与 filter_site_config/free_fn
        # 同界 -0.5·tip_r）：薄物捏持构型常让指根/前臂擦过桌面 1-2mm，
        # 伺服压入深度瞬时加深由停滞检测+恢复兜底。
        new_reg = {g for g in new_reg
                   if not (_geom_is_static(mm, g)
                           and rew.get(g, 0.0) >= _live_tol)}
        if allow_contact:
            # 持物转运：tether 接触与持物刷碰均允许；唯一硬约束是
            # 被持物未脱手（site 系滑移 ≤ 半开口）。
            if watch_obj is not None:
                _b = adapter.object_bounds(watch_obj)
                c = np.array([_b["center"][0], _b["center"][1],
                              (_b["z_top"] + _b["z_bottom"]) / 2.0])
                spn = np.asarray(adapter.get_site_pos(GRIP_SITE), float)
                sRn = np.asarray(adapter.get_site_rot(GRIP_SITE), float)
                slip = float(np.linalg.norm(sRn.T @ (c - spn) - _watch_rel0))
                if slip > _slip_tol:
                    if dbg: print("LIVECHK code=D slip=%.4f" % slip)
                    return False
            elif new_reg:
                if dbg: print("LIVECHK code=T n=%d" % len(new_reg))
                return False
        elif watch_obj is not None:
            # 只许擦碰目标；目标整体位移必须在几何零内。
            if not new_reg <= _watch_geoms:
                if dbg: print("LIVECHK code=F n=%d"
                              % len(new_reg - _watch_geoms))
                return False
            c = np.asarray(adapter.object_bounds(watch_obj)["center"], float)
            if float(np.linalg.norm(c - _watch_c0)) > IK_TOL:
                if dbg: print("LIVECHK code=D disp=%.4f"
                              % float(np.linalg.norm(c - _watch_c0)))
                return False
        elif new_reg:
            if dbg: print("LIVECHK code=N n=%d" % len(new_reg))
            return False
        return True

    # 跟踪门控流式：逐 FK 点闭环——末端进入该点跟踪门 **且 live 无
    # 碰撞**后才前进。跟踪门 = max(2×邻点间距, IK_TOL)（纯几何量，
    # 非任务参数）；每点统计无进展则整路失败。非严格执行的末点门槛
    # 放宽到 _site_step_envelope（OSC 稳态数 mm 跟踪偏置的 Jacobian
    # 包络）：残余由后续事件驱动原语（接近/接触）闭环消除——这是
    # 非严格语义的本来合同，IK_TOL 级门槛会把止挡在限位前数 mm 的
    # 合法 transit 误判为不可达。strict 时按 verify_last/IK_TOL 精确收敛。
    env = _site_step_envelope(adapter)
    prev_p = pts[0][0]
    for idx, (p, R, q_t) in enumerate(pts[1:]):
        if os.environ.get("LIBERO_DEBUG") and idx % 25 == 0:
            print(f"[dbg] exec_pt {idx}/{len(pts) - 1} "
                  f"eef={np.round(_eef(adapter), 3).tolist()}", flush=True)
        last_pt = idx == len(pts) - 2
        # 零空间姿态锚定到当前规划点：live 构型跟随规划构型，
        # 从根上消除 scratch/live 零空间分歧。
        adapter.set_nullspace_posture(q_t)

        _floor = env if (last_pt and not strict) else IK_TOL

        def _gate_ok(end, _p=p, _g=max(2.0 * float(np.linalg.norm(p - prev_p)),
                                       _floor)):
            _d = float(np.linalg.norm(end - _p))
            if _d > _g:
                _gate_ok.n = getattr(_gate_ok, "n", 0) + 1
                if os.environ.get("LIBERO_DEBUG") \
                        and _gate_ok.n % 25 == 1:
                    print("LIVECHK code=P d=%.4f g=%.4f n=%d"
                          % (_d, _g, _gate_ok.n), flush=True)
                return False, {}
            return _live_safe(), {}

        if strict and last_pt:
            def _wrap(fn):
                def chk(end):
                    ok, info = fn(end)
                    return (ok and _live_safe()), info
                return chk
            if verify_last is not None:
                r = serve(adapter, p, gripper=grip, k=K_FINE,
                          verify=_wrap(verify_last), target_rot=R)
            else:
                def _exact_ok(end, _p=p):
                    return (float(np.linalg.norm(end - _p)) <= IK_TOL, {})
                r = serve(adapter, p, gripper=grip, k=K_FINE,
                          verify=_wrap(_exact_ok), target_rot=R)
        else:
            r = serve(adapter, p, gripper=grip,
                      k=K_FINE if last_pt else k,
                      verify=_gate_ok, target_rot=R)
        if r is not None:
            return r
        prev_p = p
    return None


# ==================== 夹爪/升降原语（事件驱动，无固定步数） ====================

def _live_half_gap(adapter) -> Optional[float]:
    """现测两 pad 内侧面沿开合轴的半间距（与 measure_hand 的
    _inner_half 同口径，但读 live 状态）。preshape 的闭环反馈量。"""
    import mujoco
    m, d = _native_md(adapter)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    sp = np.asarray(d.site_xpos[sid], float)
    sR = np.asarray(d.site_xmat[sid], float).reshape(3, 3)
    tip_bodies = set()
    for b in range(m.nbody):
        nm = str(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
        if nm.endswith("_tip") and "gripper" in nm:
            tip_bodies.add(b)
    cents, pads = [], []
    for g in range(m.ngeom):
        if int(m.geom_bodyid[g]) not in tip_bodies:
            continue
        cents.append(np.asarray(d.geom_xpos[g], float))
        pads.append(g)
    if len(cents) < 2:
        return None
    dv = cents[1] - cents[0]
    u = dv / max(float(np.linalg.norm(dv)), 1e-9)
    mid = 0.5 * (cents[0] + cents[1])
    tot, n = 0.0, 0
    for c, g in zip(cents, pads):
        sizes = np.asarray(m.geom_size[g], float)
        if int(m.geom_type[g]) == 6:  # box：局部半尺寸沿 u 投影
            Rl = sR.T @ np.asarray(d.geom_xmat[g], float).reshape(3, 3)
            e = float(np.sum(np.abs(Rl.T @ u) * sizes))
        else:
            e = float(sizes[0])
        tot += max(abs(float(np.dot(c - mid, u))) - e, 0.0)
        n += 1
    return tot / max(n, 1)


def _preshape(adapter, half_gap: float) -> None:
    """闭环收爪到目标半开口：每步现测 pad 半间距，达标即转 0（保持）。
    旋钮薄耳片的进入构型——全张钳口（±4cm 实测）在密集旋钮场里指身
    必然覆盖邻居排（goal:7 实证：所有可达 IK 解固有碰撞），必须先收到
    捏夹包络（耳片宽 + 指尖球 + 裕量，现测）再接近。停止条件全是现测：
    达标 / 指关节不动（限位或夹住异物，以现宽度交接近段裁决）。"""
    m, _ = _native_md(adapter)
    fj = _finger_joints(m)
    fadr = [int(m.jnt_qposadr[j]) for j in fj]
    R_lock = np.asarray(adapter.get_site_rot(GRIP_SITE), float)
    q_prev = np.asarray(adapter.mj_data.qpos[fadr], float)
    still = 0
    while True:
        gap = _live_half_gap(adapter)
        if gap is not None and gap <= half_gap:
            for _ in range(SETTLE):
                pos = _eef(adapter)
                adapter.servo_step(GRIP_SITE, pos, gripper=0.0,
                                   target_rot=R_lock)
            return
        pos = _eef(adapter)
        adapter.servo_step(GRIP_SITE, pos, gripper=+1.0, target_rot=R_lock)
        q_now = np.asarray(adapter.mj_data.qpos[fadr], float)
        if float(np.max(np.abs(q_now - q_prev))) < 1e-5:
            still += 1
            if still >= 3:
                return
        else:
            still = 0
        q_prev = q_now


@_timed
def _move_gripper(adapter, want: float) -> None:
    """发夹爪指令直到 finger 关节停止运动（贴限位/夹住物体），
    停止由 q 无变化判定，不数固定步数。期间锁定当前位姿（闭爪
    力不得推动腕部姿态漂移）。"""
    m, _ = _native_md(adapter)
    fj = _finger_joints(m)
    R_lock = np.asarray(adapter.get_site_rot(GRIP_SITE), float)
    q_batch = np.asarray(adapter.mj_data.qpos, float)
    while True:
        for _ in range(SETTLE):
            pos = _eef(adapter)
            adapter.servo_step(GRIP_SITE, pos, gripper=want,
                               target_rot=R_lock)
        q_now = np.asarray(adapter.mj_data.qpos, float)
        if float(np.max(np.abs(q_now[[int(m.jnt_qposadr[j]) for j in fj]]
                                - q_batch[[int(m.jnt_qposadr[j])
                                           for j in fj]]))) < 1e-5:
            return
        q_batch = q_now


def _open_on_support(adapter, press_tgt) -> None:
    """支撑面释放（On 落座）：开爪全程保持放置下压伺服（目标=接触
    面下方一个跟踪门，该目标已在接触事件中验证可达——物理接触先于
    到达发生）。开爪时内 pad 外推物体的水平力由支撑面摩擦（μ·N，
    N=下压保持力）抵抗，消除落座滑移；纯力学机制，无任务参数。
    关节停止判定与 _move_gripper 同（q 无变化）。"""
    m, _ = _native_md(adapter)
    fj = _finger_joints(m)
    Rt = np.asarray(adapter.get_site_rot(GRIP_SITE), float)
    q_batch = np.asarray(adapter.mj_data.qpos, float)
    while True:
        for _ in range(SETTLE):
            pos = _eef(adapter)
            adapter.servo_step(GRIP_SITE, press_tgt, gripper=-1.0,
                               k=K_FINE, target_rot=Rt)
        q_now = np.asarray(adapter.mj_data.qpos, float)
        if float(np.max(np.abs(q_now[[int(m.jnt_qposadr[j]) for j in fj]]
                                - q_batch[[int(m.jnt_qposadr[j])
                                           for j in fj]]))) < 1e-5:
            return
        q_batch = q_now


@_timed
def _lift_until_free(adapter, obj: str, support_z: float,
                     hand: Dict[str, Any]) -> Tuple[float, bool]:
    """抬升到物体几何脱离支撑（底面 > support_z + IK_TOL 的几何零）。

    serve 单轮未收敛不是失败信号：持物负载下末端逐轮慢爬是正常物理；
    裁决完全基于几何：
    - 碗底已越过 support_z + IK_TOL → 成功（脱离支撑，可进入 transit）；
    - 进展只认**物体上升**：被抓物随手是刚性夹持的物理不变量，
      手升而物不升 = 物体正从指间滑脱（或未夹住），与"手被环境
      止挡"同判——连续多轮物体无 ≥IK_TOL 上升即失败，淘汰假抓取；
      轮数预算是**柔度预算**而非几何量：腱传动手指在负载下会先伸
      展再带动物体（手指是手的串联柔度，手升≠pad 升），轻物体
      打破支撑预载同样需要数轮力建涨——2 轮会把这两种正常过渡
      误判成滑脱（实测 6g 碗第 3 轮才离地），取 4 轮（统计约定）；
    - 抬升行程超过 0.30m（IK 可达上界，模型导出）→ 失败。
    全程锁定抓取姿态、保持闭爪。返回 (抬升量, 是否成功)。
    """
    start = _eef(adapter)
    R_lock = np.asarray(adapter.get_site_rot(GRIP_SITE), float)
    target = np.array(start, float)
    last_bowl = float(support_z)
    stuck = 0
    while True:
        target[2] += 0.01
        serve(adapter, target, gripper=1.0, k=K_FINE, target_rot=R_lock)
        b = adapter.object_bounds(obj)
        bowl_z = float(b["z_bottom"])
        hand_z = float(_eef(adapter)[2])
        f_now = float(adapter.contact_force_on_body(obj))
        if os.environ.get("LIBERO_DEBUG"):
            print(f"[dbg] lift hand_z={hand_z:.3f} bowl_z={bowl_z:.3f} "
                  f"sup={support_z:.3f} tgt={target[2]:.3f} "
                  f"stuck={stuck} f={f_now:.1f}", flush=True)
            _dbg_force_balance(adapter, obj)
            _dbg_dump_contacts(adapter, "lift")
        if bowl_z > support_z + IK_TOL:
            return bowl_z - support_z, True
        progress = bowl_z - last_bowl >= IK_TOL
        last_bowl = bowl_z
        # 夹持力丢失 = 物体已滑脱，立即失败（比 stuck 更紧迫）
        if f_now < 0.5:
            if os.environ.get("LIBERO_DEBUG"):
                print(f"[dbg] lift force_lost f={f_now:.1f} → fail", flush=True)
            return bowl_z - support_z, False
        stuck = 0 if progress else stuck + 1
        if stuck >= 4:
            return bowl_z - support_z, False
        if target[2] - start[2] > 0.30:
            return bowl_z - support_z, False


# ==================== 抓取 ====================

def _approach_to_grasp(adapter, site, Rs, obj: str = "",
                       tgt_geoms: Optional[set] = None,
                       zone: Optional[np.ndarray] = None,
                       grip: float = -1.0):
    """位姿锁定下沿 site 接近向伺服，事件语义（通用，无任务分支）：

    1) 末端几何到达目标位点（位姿收敛）→ 停止；
    2) 末端运动在统计窗口内无进展（斜率与零无显著差异）= 手被
       环境物理止挡，按到达处理 → 停止；
    3) 机器人-目标/环境接触（obj 非空时按机器人侧 geom 细分，
       pad = 实际接触面，见 _pad_geom_set）：
       - pad 接触目标 geom → 到达抓深，停止；
       - 指身/腕部接触目标（body_target）→ 构型不适配：腕下式
         pinch 中 pad 悬在指身下方数厘米，指身先触目标顶沿说明
         该方位放不下这只手，继续接近只会把指身压上目标 → 该候选
         失败交轮询换候选；
       - 仅支撑面 geom 接触（指尖搁置/滑过桌面，夹薄物的正常力学）
         → 继续接近；
       - 其他 env geom 接触（指尖被异物阻挡）→ 该候选此路不通，
         返回失败交轮询换候选；
       obj 为空且 tgt_geoms 给定时（articulate 夹具接近）：只有夹具
       子树 geom 的接触算到达证据，其余擦碰（途经桌面杂物）不停车，
       由进展/停滞统计裁决；obj 与 tgt_geoms 均为空：任何接触即停。
       zone 给定时（多旋钮同 body 的夹具，goal:7 实证）：夹具 geom
       的接触还须落在作用 geom 邻域 AABB 内才算到达——邻居旋钮/
       外壳 geom 与作用耳片焊在同一刚体，geom 级判定会在距作用点
       数厘米处把邻居擦碰当到达，手在作用点外停住闭爪夹空。

    是否真抓住由后续闭爪 + 抬升几何复验裁决（没夹住则抬不起来，
    候选自然失败）。慢进展允许走完，但设两类停滞兜底：窗口绝对闭距
    不足 0.25·IK_TOL = 物理止挡（近距离按到达处理）；远距离（>指尖
    尺度）无接触停滞 = 伺服跟踪饱和（病态分支），返回 ik_unreachable
    交调用方做关节路径恢复。
    """
    Rt = np.asarray(Rs, float)
    tgt = np.asarray(site, float)
    zone = None if zone is None else np.asarray(zone, float)
    m0, _ = _native_md(adapter)
    arm_adrs = [int(m0.jnt_qposadr[j]) for j in _arm_joints(m0)]
    st = _ServoStats()
    hs = _tip_lever(adapter)
    tip_r = float(measure_hand(adapter)["tip_r"])
    t = 0
    while True:
        end = _eef(adapter)
        Rc = np.asarray(adapter.get_site_rot(GRIP_SITE), float)
        d_pos = float(np.linalg.norm(end - tgt))
        d_rot = float(np.linalg.norm(_rot_err(Rt, Rc)))
        if d_pos <= IK_TOL and d_rot <= ROT_TOL:
            return None  # 事件 1：几何到达
        # 事件 2：手指接触。接触分类（obj 非空时）：
        # - 目标 geom 接触 → 到达抓深，停止（原语义）；
        # - 仅支撑面 geom 接触（_support_geoms：指尖搁置/滑过桌面）→
        #   夹薄物的正常力学，继续接近；
        # - 其他 env geom 接触 → 指尖被异物阻挡，该候选此路不通，
        #   返回失败让轮询换候选（原先误停在此处闭爪必夹空/夹错）。
        # obj 为空（articulate 到达）时保持原语义：任何接触即停。
        f_obj = float(adapter.gripper_contact_force())
        if f_obj > 0.0:
            if obj:
                import mujoco
                mm, dd = _native_md(adapter)
                robot = _robot_geom_set(mm)
                pads = _pad_geom_set(mm)
                tgt_geoms = _site_grasp_geoms(adapter, obj)
                sup = _support_geoms(adapter, obj)
                pad_touch, body_touch = set(), set()
                blocked_touch = set()
                # 支撑面豁免是**接触点级**的，不是 geom 级：地板/桌面是
                # 目标的支撑面 geom，但手与它的接触只有发生在目标足印
                # 走廊内才算"指尖搁置/滑过桌面"（夹薄物的正常力学）。
                # geom 级豁免会让手在距目标 10cm 处压进桌面（254N，
                # object:9 OJ 实证）也被判"支撑面擦碰"放行，接近伺服
                # 一路压到停滞兜底才退。走廊 = 目标 AABB 外扩 a_reach
                # （可达半开距，现测量）：接触点距目标中心超过半宽+
                # a_reach 不可能是夹持接近走廊的一部分。
                bb = adapter.object_bounds(obj)
                mrg = float(measure_hand(adapter)["a_open"]) + tip_r
                x_lo = float(bb["center"][0]) - float(bb["half_x"]) - mrg
                x_hi = float(bb["center"][0]) + float(bb["half_x"]) + mrg
                y_lo = float(bb["center"][1]) - float(bb["half_y"]) - mrg
                y_hi = float(bb["center"][1]) + float(bb["half_y"]) + mrg
                for i in range(dd.ncon):
                    cc = dd.contact[i]
                    g1, g2 = int(cc.geom1), int(cc.geom2)
                    r1, r2 = g1 in robot, g2 in robot
                    if r1 and r2:
                        continue           # 自接触不分类
                    if r1 or r2:
                        rg = g1 if r1 else g2
                        eg = g2 if r1 else g1
                        if eg in tgt_geoms:
                            (pad_touch if rg in pads else body_touch).add(eg)
                        elif eg in sup and \
                                x_lo <= float(cc.pos[0]) <= x_hi and \
                                y_lo <= float(cc.pos[1]) <= y_hi:
                            continue       # 足印走廊内支撑面擦碰：放行
                        else:
                            blocked_touch.add(eg)
                blocked = blocked_touch
                if os.environ.get("LIBERO_DEBUG"):
                    names = lambda gs: sorted(
                        {str(mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM,
                                               g) or g) for g in gs})
                    # 接触明细：机器人侧 geom + 位置。早期误停（如离
                    # 目标还有 2cm 就判"到达"）不看明细无法定位。
                    det = []
                    for i in range(dd.ncon):
                        cc = dd.contact[i]
                        g1, g2 = int(cc.geom1), int(cc.geom2)
                        r1, r2 = g1 in robot, g2 in robot
                        if r1 == r2:
                            continue
                        det.append(f"{'R:' if r1 else ''}{names({g1})}[{'R:' if r2 else ''}{names({g2})}]"
                                   f"@({cc.pos[0]:.3f},{cc.pos[1]:.3f},{cc.pos[2]:.3f})")
                    print(f"[dbg] approach_contact d={d_pos:.4f} "
                          f"dr={d_rot:.3f} f={f_obj:.1f} "
                          f"pad_target={names(pad_touch & tgt_geoms)} "
                          f"body_target={names(body_touch & tgt_geoms)} "
                          f"blocked={names(blocked)} det={' | '.join(det[:6])}",
                          flush=True)
                if blocked:
                    return {"reason": "接近段被环境阻挡",
                            "mechanism": "ik_unreachable", "measures": {}}
                if pad_touch & tgt_geoms:
                    return None            # pad 到达抓深
                if body_touch & tgt_geoms:
                    # 指身/腕部先触目标，分两种力学：
                    # - **压近水平面**（nz≥0.7，同 _support_geoms 的"近垂
                    #   直支撑法向"界）：腕下式 pinch 中 pad 悬在指身下方
                    #   数厘米，指身压目标顶沿 = 该方位构型不适配此目标
                    #   尺寸，继续只会把指身压上目标 → 淘汰该候选；
                    # - **斜坡/侧棱擦碰**（nz<0.7）：侧外夹高物时腕底掠
                    #   过目标肩部/侧棱（object:9 OJ 实证，1~2N 小力），
                    #   继续接近 pad 可达 flank，指身接触随深入自然消
                    #   失——放行，由 pad 到达抓深 / 停滞兜底再裁决。
                    press = False
                    for _ci in range(dd.ncon):
                        _cc = dd.contact[_ci]
                        _g1, _g2 = int(_cc.geom1), int(_cc.geom2)
                        _r1, _r2 = _g1 in robot, _g2 in robot
                        if _r1 == _r2:
                            continue
                        _rg = _g1 if _r1 else _g2
                        _eg = _g2 if _r1 else _g1
                        if _rg in pads or _eg not in (body_touch & tgt_geoms):
                            continue
                        if abs(float(_cc.frame[2])) >= 0.7:
                            press = True
                            break
                    if press:
                        return {"reason": "指身压目标近水平面（非 pad），"
                                          "构型不适配",
                                "mechanism": "ik_unreachable",
                                "measures": {}}
                # 支撑面走廊擦碰 / 目标斜坡擦碰：指尖搁置，继续接近
                pass
            else:
                if tgt_geoms is not None:
                    # 只认夹具子树接触；其他擦碰继续接近（进展/停滞
                    # 统计兜底真阻塞）。zone 给定时（多旋钮同 body）还
                    # 须落在作用 geom 邻域内——邻居旋钮/外壳擦碰不算
                    # 到达（goal:7 实证：手掌压邻居旋钮顶面 20N 被误判
                    # 到达，停在作用耳片 1.5cm 外闭爪夹空、拖 q 纯挂机）。
                    import mujoco
                    mm, dd = _native_md(adapter)
                    robot = _robot_geom_set(mm)
                    touch = False
                    foreign = False
                    pads = _pad_geom_set(mm)
                    for _ci in range(dd.ncon):
                        _cc = dd.contact[_ci]
                        _g1, _g2 = int(_cc.geom1), int(_cc.geom2)
                        _r1, _r2 = _g1 in robot, _g2 in robot
                        if _r1 == _r2:
                            continue
                        _rg = _g1 if _r1 else _g2
                        _eg = _g2 if _r1 else _g1
                        if _eg in tgt_geoms:
                            if _rg not in pads:
                                # 到达证据必须是 pad 接触：指身/腕部擦到
                                # 夹具侧面不算（goal:7 实证：指身擦 site
                                # geom 侧棱 2.7N 误判到达，pad 悬在顶面
                                # 外空跟随）——记 foreign 由停滞/进展裁决。
                                foreign = True
                                continue
                            if zone is not None and not (
                                    zone[0, 0] <= _cc.pos[0] <= zone[1, 0]
                                    and zone[0, 1] <= _cc.pos[1] <= zone[1, 1]
                                    and zone[0, 2] <= _cc.pos[2] <= zone[1, 2]):
                                foreign = True
                                continue
                            touch = True
                            break
                    if foreign and not touch:
                        if os.environ.get("LIBERO_DEBUG"):
                            print(f"[dbg] approach_contact "
                                  f"foreign_body_contact d={d_pos:.4f} "
                                  f"eef={np.round(end,3).tolist()}",
                                  flush=True)
                            _dbg_dump_contacts(adapter, "approach_contact")
                        return {"reason": "同 body 非作用 geom 阻挡"
                                          "（邻居旋钮/外壳）",
                                "mechanism": "ik_unreachable",
                                "measures": {"d_pos": round(d_pos, 4)}}
                    if touch:
                        if os.environ.get("LIBERO_DEBUG"):
                            det = []
                            for _ci in range(dd.ncon):
                                _cc = dd.contact[_ci]
                                _g1, _g2 = int(_cc.geom1), int(_cc.geom2)
                                _r1, _r2 = _g1 in robot, _g2 in robot
                                if _r1 == _r2:
                                    continue
                                _n1 = mujoco.mj_id2name(
                                    mm, mujoco.mjtObj.mjOBJ_GEOM, _g1) or _g1
                                _n2 = mujoco.mj_id2name(
                                    mm, mujoco.mjtObj.mjOBJ_GEOM, _g2) or _g2
                                det.append(
                                    f"{_n1}[{_n2}]@({ _cc.pos[0]:.3f},"
                                    f"{_cc.pos[1]:.3f},{_cc.pos[2]:.3f})"
                                    f"n=({ _cc.frame[0]:.2f},"
                                    f"{_cc.frame[1]:.2f},{_cc.frame[2]:.2f})")
                            print(f"[dbg] approach_contact d={d_pos:.4f} "
                                  f"dr={d_rot:.3f} f={f_obj:.1f} "
                                  f"eef={np.round(end,3).tolist()} (fixture) "
                                  f"det={' | '.join(det[:6])}",
                                  flush=True)
                        return None
                elif os.environ.get("LIBERO_DEBUG"):
                    print(f"[dbg] approach_contact d={d_pos:.4f} "
                          f"dr={d_rot:.3f} f={f_obj:.1f} "
                          f"eef={np.round(end,3).tolist()}", flush=True)
                    _dbg_dump_contacts(adapter, "approach_contact")
                    return None
                else:
                    return None
        qall = np.asarray(adapter.mj_data.qpos, float)
        fnow = float(adapter.gripper_contact_force())
        st.add(t, d_pos + hs * d_rot, qall[arm_adrs], fnow)
        if len(st.ds) >= WIN and t % WIN == WIN - 1:
            ft = st.fit()
            tstat = -ft["slope"] / max(ft["se"], 1e-9)
            # 绝对进展地板：t 检验对低噪声爬行盲目——饱和伺服在关节/
            # 物理边界处可以 0.04mm/10 步地"统计显著"爬行（butter
            # 侧夹接近段 4cm 残差处实测），永远触发不了无进展事件、
            # 挂死整条 attempt。窗口内闭距不足 0.25·IK_TOL 即视为物
            # 理止挡（与 t 检验同语义：到达处理，后续闭爪+抬升复验）。
            progressed = st.ds[-WIN][1] - st.ds[-1][1]
            if tstat < TSTAT or progressed < 0.25 * IK_TOL:
                if os.environ.get("LIBERO_DEBUG"):
                    print(f"[dbg] approach_stall d={d_pos:.4f} "
                          f"dr={d_rot:.3f} eef={np.round(end,3).tolist()}",
                          flush=True)
                    _dbg_dump_contacts(adapter, "approach_stall")
                # 远距离停滞 ≠ 物理到底：无接触事件且残差远超接触
                # 尺度（指尖半径/数个 IK 容差）时，是直笛卡尔伺服在
                # 病态关节分支上跟踪失败（关节余量充足、站点多解
                # 实证）——判跟踪饱和交调用方做关节路径恢复；近距离
                # 停滞维持原语义：物理止挡按到达处理（后续闭爪+抬升
                # 复验）。
                if d_pos > max(0.5 * tip_r, 4.0 * IK_TOL):
                    return {"reason": f"接近跟踪饱和（d={d_pos:.3f}）",
                            "mechanism": "ik_unreachable",
                            "measures": {"d_pos": round(d_pos, 4)}}
                if obj == "" and tgt_geoms is not None and zone is not None:
                    # knob 夹具模式（多旋钮同 body，zone 门控）：到达证据
                    # 必须是 zone 内 pad 接触。无接触的近距停滞 = 该构型
                    # 够不到（伺服饱和/限位），按到达处理会让手悬在表面
                    # 上方闭爪空跟随（goal:7 顶压实证：残差 1cm 处饱和、
                    # pad 悬空 5mm、q 恒不转）。判失败交调用方换腿。
                    # zone=None 的单把手夹具（柜门/抽屉）维持原语义：
                    # 近距停滞按物理止挡到达处理——闭爪行程本身会把
                    # 把手最后几毫米带上（goal:3/4 回归实证：抽屉把手
                    # 擦碰前 3~4mm 干净停滞，旧语义下闭爪即挂住）。
                    return {"reason": f"夹具接近停滞无接触（d={d_pos:.3f}）",
                            "mechanism": "ik_unreachable",
                            "measures": {"d_pos": round(d_pos, 4)}}
                return None  # 事件 3：物理到底（无接触无进展）
        adapter.servo_step(GRIP_SITE, tgt, gripper=grip, k=K_FINE,
                           target_rot=Rt)
        t += 1


def _retreat(adapter, pre_site, Rs, inflation) -> None:
    """候选失败后撤离：开爪状态沿无碰撞关节路径退回该候选的预抓取点。

    手停在物体旁（尤其容器/抽屉内）时，后续候选的 transit 规划从
    楔入位形出发会成片报"无可达无碰撞关节路径"，把可解候选饿死。
    退回 pre_site（pregrasp_point 构造上无碰撞）是机制级卫生动作，
    与任务无关；规划失败不追罚（尽力而为，世界状态已变）。
    """
    path = plan_arm_path(adapter, pre_site, grip=-1.0, target_rot=Rs,
                         inflation=inflation)
    if path is not None:
        execute_arm_path(adapter, path, grip=-1.0, strict=False)


@_timed
def _closure_holds(adapter, obj: str, qv, min_depth: float) -> bool:
    """闭爪仿真门：构型 qv 下把手指关节置闭合限位再 forward，检查
    pad∩目标的最深接触是否达到 min_depth（强制闭位下 pad 球体对目
    标的贯穿深度）。

    无接触 = 闭爪捏空；接触浅于 min_depth = 捏在锥面/棱缘上——抬升
    时法向力有滑出分量，同样脱手。object:9 实证：瓶颈候选理想位形
    下 pad~cap 贯穿仅 1.1mm（颈径 ≈ tip_r 量级），闭爪力 24.6N 全
    压在指间（手指互斥无接触对），lift 初段 3.1N 即滑脱；瓶身候选
    贯穿 ≫ min_depth 则 40N 稳持。深度门取 0.25·tip_r：与支撑面
    深度门/擦碰兜底同一手几何比例约定，非任务阈值。
    """
    import mujoco
    m, sd = _scratch(adapter, None)
    j3 = _arm_joints(m)
    for jj, vv in zip(j3, np.asarray(qv, float)):
        sd.qpos[int(m.jnt_qposadr[jj])] = float(vv)
    for j in _finger_joints(m):
        lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
        # 镜像对称手指（一正一负行程）：|端点| 小 = 闭合（与
        # _scratch 的 grip 语义同判定）。
        close_v = lo if abs(hi) >= abs(lo) else hi
        sd.qpos[int(m.jnt_qposadr[j])] = close_v
    mujoco.mj_forward(m, sd)
    pads = _pad_geom_set(m)
    tgt = _site_grasp_geoms(adapter, obj)
    deepest = 0.0
    for _ci in range(sd.ncon):
        _c = sd.contact[_ci]
        _g1, _g2 = int(_c.geom1), int(_c.geom2)
        if (_g1 in pads and _g2 in tgt) or (_g2 in pads and _g1 in tgt):
            deepest = min(deepest, float(_c.dist))
    return deepest <= min_depth


def grasp(adapter, obj: str, cand: int = 0,
          mem: Optional[Dict[str, dict]] = None) -> Dict[str, Any]:
    """任务无关抓取：GraspNet 对**全场景**点云生成抓取候选（学习型
    抓取综合，网络需感知支撑环境），逐候选走 TAMP 管线——站点碰撞
    过滤（6D IK + 只许接触目标）→ pregrasp 几何后退（6D IK + 零碰撞）
    → RRT-Connect 无碰撞关节路径 → 位姿伺服接近 → 闭爪（关节停止
    事件）→ 抬升到几何脱离支撑。任一环节不过即换下一个候选；
    已失败候选记入 mem["__failed__"]，重规划轮内不重复。成功后
    按闭爪后实测写入夹持几何（含闭爪滑移）。无任何 per-task 分支。
    """
    mem = mem if mem is not None else {}
    failed_lst = mem.setdefault("__failed__", {}).setdefault(obj, [])
    tried = {tuple(x) for x in failed_lst}
    # 枚举穷尽记忆：候选集是（目标位姿/几何）的纯函数。上次 grasp
    # 已穷尽全部候选且目标几何签名未变时，重枚举只会重算一遍同样的
    # 昂贵过滤（rim/side 各 12 方位 × 24 重启 IK ≈ 分钟级）再被
    # failed 键逐个跳过——失败任务每重规划轮白烧一次（object:6
    # 实证：round 2 grasp 秒败后仍有 300s 耗在重枚举上）。世界被
    # 动作改变（签名变）时正常重枚举。
    sig_now = None
    enum_sig = mem.setdefault("__enum_sig__", {})
    try:
        _b = adapter.object_bounds(obj)
        sig_now = (round(float(_b["center"][0]), 3),
                   round(float(_b["center"][1]), 3),
                   round(float(_b["z_top"]), 3),
                   round(float(_b["half_x"]), 3),
                   round(float(_b["half_y"]), 3))
    except Exception:
        pass
    if sig_now is not None and enum_sig.get(obj) == sig_now:
        return _fail("候选已穷尽（目标几何未变，重枚举结果不变）",
                     "no_candidate", obj=obj, cand=cand, n_tried=0)
    _snap_reset()
    hand = measure_hand(adapter)
    cands = graspnet_scene_candidates(adapter, obj, top_k=64)

    # 目标候选缺页升级：学习型综合按可抓性排序，全场景易抓物体（罐
    # 头、高瓶）常占满学习者的 top 页，低矮/遮挡目标的候选一页内一
    # 个不落（goal:1 浅碗实证 64 候选零落目标，rim/side 分析解又被环
    # 境门饿死）。按"宽度可达且 3D AABB 距 ≤ a_reach（不看 z，宽口径）"
    # 复算落目标的候选数，为零时翻页到 top_k=256 重取一次。
    a_reach0 = float(hand["a_open"]) + float(hand["tip_r"])
    try:
        _b0 = adapter.object_bounds(obj)
        _on = 0
        for _c in cands:
            _s = np.asarray(_c["site"], float)
            _dx = max(abs(float(_s[0]) - _b0["center"][0]) - _b0["half_x"], 0.0)
            _dy = max(abs(float(_s[1]) - _b0["center"][1]) - _b0["half_y"], 0.0)
            _dz = max(float(_s[2]) - _b0["z_top"],
                      _b0["z_bottom"] - float(_s[2]), 0.0)
            if float(_c.get("width", 0.0)) <= a_reach0 \
                    and math.hypot(_dx, _dy, _dz) <= a_reach0:
                _on += 1
        if _on == 0 and cands:
            cands = graspnet_scene_candidates(adapter, obj, top_k=256)
    except Exception:
        pass

    last_reason = "GraspNet 无候选"
    n_tried = 0
    inflation = _transit_inflation(adapter)

    def _solutions():
        # 求解源一：GraspNet 学习型抓取综合（全场景点云）。物理可行
        # 性过滤：候选夹持半宽不得超过手的可达半开距。可达半开距两档：
        # ① a_open + tip_r——指尖球心最大间距 a_open，按压发生在**pad
        #   表面**（球心内 tip_r 处），比 a_open 宽 tip_r/侧的物体 pad 仍
        #   能压到；
        # ② a_open + 2*tip_r——pad 表面从球心向内缩 tip_r，两侧合计
        #   可再包容 2*tip_r 的挤压量（挤压摩擦抓持，Libero 罐头/浅碗
        #   rim 类物体的可行抓法：候选宽 5.0cm vs 档①4.75cm，档②5.55cm
        #   goal:1 实证）。两档均为纯物理量（实测手开合、指尖半径），
        #   无任务参数。优先用档①（挤压量小、握持稳），档②降序排在
        #   档①候选之后（同分数带内不抢优先位）。
        a_reach = float(hand["a_open"]) + float(hand["tip_r"])
        a_sqz = float(hand["a_open"]) + 2.0 * float(hand["tip_r"])
        tip_r = float(hand["tip_r"])
        b0 = adapter.object_bounds(obj)
        obj_h = float(b0["z_top"]) - float(b0["z_bottom"])
        # 抓取点几何约束：z 必须在物体中上部（≥1/3 高度），避免夹底推倒。
        # 该带的物理作用是防指尖探到支撑面附近发生 scooping/撬翻——
        # 只有高物体才有 1/3 高度的撬翻行程；浅物体（碗碟类，高度仅数
        # cm）有害带只到支撑面以上指尖尺度的深度。钳位到 3*tip_r：
        # 两个纯几何量（物体高度、指尖半径），无任务参数。
        z_min = float(b0["z_bottom"]) + min(obj_h / 3.0, 3.0 * tip_r)

        def _scene_sols():
            # 目标邻近门：site 是 GRIP_SITE 目标点，两 pad 触物发生在
            # site 邻域 a_reach 内（可达半开距，本函数上段现测）。site
            # 距目标 AABB 超过 a_reach 的候选在几何上不可能接触目标——
            # 必抓邻居或空气（gsnet 全场景候选常落在 20cm 外其他物体上，
            # object:3/object:6 实证白烧 try 预算）。AABB 用 b0（含旋转
            # 修正的真值包围盒），距离取三维（悬于目标上方 < a_reach
            # 的顶抓候选仍合法）。
            ox, oy = float(b0["center"][0]), float(b0["center"][1])
            hx0, hy0 = float(b0["half_x"]), float(b0["half_y"])
            zt0, zb0 = float(b0["z_top"]), float(b0["z_bottom"])

            def _on_target(s) -> bool:
                ddx = max(abs(float(s[0]) - ox) - hx0, 0.0)
                ddy = max(abs(float(s[1]) - oy) - hy0, 0.0)
                ddz = max(float(s[2]) - zt0, zb0 - float(s[2]), 0.0)
                return math.hypot(ddx, ddy, ddz) <= a_reach

            # 宽度两档（物理 squeeze 模型见上）：档①稳握优先全出，
            # 档②大挤压降序排后——不在同分数带内与档①抢优先位。
            for wcap in (a_reach, a_sqz):
                for sol in cands:
                    w = float(sol.get("width", 0.0))
                    if (w <= a_reach) != (wcap == a_reach):
                        continue
                    if w <= wcap and float(sol["site"][2]) >= z_min:
                        s = np.asarray(sol["site"], float)
                        if _on_target(s):
                            yield dict(sol, _src="scene")

        # 求解源二：分析解（rim 深捏 ↔ 侧外夹 轮询交错）。rim 捏优先
        # （先出）但侧外夹在前几轮即可被尝试：rim 捏在实心/无腔物体上
        # 也能产出"挤捏"解，串行产出会把侧外夹饿死（薄盒场景实测 8 连
        # rim 捏夹空、side_grip 从未被尝试）。prior 方位排斥使每次调用
        # 求新方位。
        def _analytic_sols():
            prior: List[np.ndarray] = []
            # 执行失败方位记忆（跨 grasp 调用持久，存于 mem）：同一物体
            # 几何签名下，闭爪/抬升阶段失败过的 rim/side 方位在 prior
            # 方位角分辨率内持久排斥——否则每次 grasp 调用 prior 从空
            # 开始，会把上一轮执行失败的最佳方位重新提议一遍（goal:4
            # 实证：8 次 attempt 反复重试相同 2-4 个 rim 方位烧光预算，
            # 其余可行方位从未到达）。签名变化（物体被移动）记忆作废。
            xf = mem.setdefault("__exec_fail__", {}).get(obj)
            if xf:
                if xf.get("sig") == sig_now:
                    prior = [np.asarray(p, float) for p in xf["sites"]]
                else:
                    mem["__exec_fail__"].pop(obj, None)

            def _rim():
                for _ in range(ANALYTIC_TRIES):
                    sol = analytic_rim_pinch(adapter, obj, prior)
                    if sol is None:
                        return
                    prior.append(np.asarray(sol["site"], float))
                    if float(sol["site"][2]) >= z_min:
                        yield dict(sol, _src="rim_pinch")

            def _side():
                for _ in range(ANALYTIC_TRIES):
                    sol = analytic_side_grip(adapter, obj, prior)
                    if sol is None:
                        return
                    prior.append(np.asarray(sol["site"], float))
                    if float(sol.get("width", 0.0)) / 2.0 <= a_reach \
                            and float(sol["site"][2]) >= z_min:
                        yield dict(sol, _src="side_grip")

            its = [iter(_rim()), iter(_side())]
            alive = [True, True]
            while any(alive):
                for i, it in enumerate(its):
                    if not alive[i]:
                        continue
                    try:
                        yield next(it)
                    except StopIteration:
                        alive[i] = False

        # 轮询交错产出：网络候选悬停在容器口上方/夹缝中等成串坏候选
        # 时，逐一穷举会耗尽重规划预算而饿死精确的分析解；交替产出
        # 保证任一源的好解在前几轮即可被尝试。源间顺序保持
        # 学习优先（网络候选先出）。
        its = [iter(_scene_sols()), iter(_analytic_sols())]
        alive = [True, True]
        while any(alive):
            for i, it in enumerate(its):
                if not alive[i]:
                    continue
                try:
                    yield next(it)
                except StopIteration:
                    alive[i] = False

    for sol in _solutions():
        Rs = sol["R"]
        site = sol["site"]
        key = tuple(np.round(sol["center"], 4).tolist()
                    + np.round(Rs, 3).reshape(-1).tolist())
        if key in tried:
            continue
        n_tried += 1
        if os.environ.get("LIBERO_DEBUG"):
            b0 = adapter.object_bounds(obj)
            print("DBG try=%d site=%s eef=%s bowl=%s" % (
                n_tried, np.round(site, 3), np.round(_eef(adapter), 3),
                np.round(b0["center"], 3)))
        src = sol.get("_src", "?")
        _snap(adapter, "try", obj=obj, n_tried=n_tried, site=site, R=Rs,
              src=src)
        qv = sol.get("_qv")
        if qv is None:
            f = filter_site_config(adapter, obj, site, Rs)
            if f is None:
                last_reason = "站点构型不可行（碰撞/IK）"
                _stageln("grasp", n_tried, "site_filter", False,
                         last_reason, src=src)
                _snap(adapter, "fail_site_filter", obj=obj, site=site,
                      R=Rs, reason=last_reason)
                failed_lst.append(list(key))
                if src in ("rim_pinch", "side_grip") and sig_now is not None:
                    _e = mem.setdefault("__exec_fail__", {}) \
                             .setdefault(obj, {"sig": sig_now, "sites": []})
                    _e["sites"].append(list(map(float, site)))
                continue
            qv = f[0]
        # 闭爪仿真门：几何过滤只管"能到达且不碰"，不管"闭爪捏得到"。
        # 窄颈/瓶盖类候选闭爪后 pad 零接触（捏空），approach+lift
        # 全过但零持力（object:9 实证 28 连）。仿真拒掉，枚举转向
        # 能真实夹持的候选，不烧 try 预算。
        if not _closure_holds(adapter, obj, qv,
                              min_depth=-0.25 * float(hand["tip_r"])):
            last_reason = "闭爪仿真无 pad-目标接触（捏空）"
            _stageln("grasp", n_tried, "closure_gate", False,
                     last_reason, src=src)
            _snap(adapter, "fail_closure", obj=obj, site=site, R=Rs,
                  reason=last_reason)
            failed_lst.append(list(key))
            if src in ("rim_pinch", "side_grip") and sig_now is not None:
                _e = mem.setdefault("__exec_fail__", {}) \
                         .setdefault(obj, {"sig": sig_now, "sites": []})
                _e["sites"].append(list(map(float, site)))
            continue
        # 预抓取方向重试：预抓取点本身无碰撞不代表从该点出发的接近段
        # /transit 可行（拥挤场景第一个可行方向常退向邻物，接近段即
        # 擦碰）。接近段/transit 失败方向记入 blocked，重选下一个可行
        # 撤退方向（机制级，与任务无关）。
        blocked_dirs: List[np.ndarray] = []
        path = None
        pre_site = None
        for _pre_attempt in range(4):
            pre = pregrasp_point(adapter, site, Rs,
                                 blocked=tuple(blocked_dirs),
                                 q_init=sol.get("_qv"))
            if pre is None:
                if blocked_dirs:
                    last_reason = "预抓取方向重试后仍不可达"
                else:
                    last_reason = "无可达预抓取点"
                _stageln("grasp", n_tried, "pregrasp", False, last_reason,
                         src=src)
                _snap(adapter, "fail_pregrasp", obj=obj, site=site, R=Rs,
                      reason=last_reason)
                break
            pre_site, _q_pre = pre
            dv = pre_site - site
            nv = float(np.linalg.norm(dv))
            if nv > 1e-9:
                blocked_dirs.append(dv / nv)
            if not _approach_clear(adapter, obj, pre_site, site, Rs,
                                   q_init=_q_pre):
                last_reason = "接近段不畅通（碰撞/IK）"
                _stageln("grasp", n_tried, "approach_clear", False,
                         last_reason, attempt=_pre_attempt, src=src)
                _snap(adapter, "fail_approach_clear", obj=obj, site=site,
                      R=Rs, reason=last_reason)
                if nv <= 1e-9:
                    # 零撤退预抓点（复验构型直查命中）：blocked 无方向可
                    # 记，重试只会返回同一点——直接换下一个抓取候选，不
                    # 烧重试预算（object:7 milk 实证：同点空转 4 轮）。
                    break
                continue
            path = plan_arm_path(adapter, pre_site, grip=-1.0, target_rot=Rs,
                                 inflation=inflation)
            if path is None:
                last_reason = "无可达无碰撞关节路径"
                _stageln("grasp", n_tried, "plan_path", False, last_reason,
                         attempt=_pre_attempt, src=src)
                _snap(adapter, "fail_plan", obj=obj, site=site, R=Rs,
                      reason=last_reason)
                continue
            break
        if path is None:
            failed_lst.append(list(key))
            if src in ("rim_pinch", "side_grip") and sig_now is not None:
                _e = mem.setdefault("__exec_fail__", {}) \
                         .setdefault(obj, {"sig": sig_now, "sites": []})
                _e["sites"].append(list(map(float, site)))
            continue
        # 非严格路径执行允许物理残余（OSC 稳态偏置），故无论几何
        # 上 pre_site 与 site 多近，都必须在路径到达后以事件驱动接近
        # 把末端真正送到接触/物理止挡（否则闭爪位置与规划无关）。
        _snap(adapter, "pre_exec", obj=obj, n_tried=n_tried, site=site,
              R=Rs, pre_site=pre_site)
        r = execute_arm_path(adapter, path, grip=-1.0, strict=False,
                             watch_obj=obj)
        if os.environ.get("LIBERO_DEBUG"):
            print("DBG  post_path eef=%s force=%.1f" % (
                np.round(_eef(adapter), 3),
                adapter.contact_force_on_body(obj)))
        if r is not None:
            last_reason = r.get("reason") or "path_exec_fail"
            _stageln("grasp", n_tried, "exec_path", False, last_reason,
                     src=src)
            _snap(adapter, "fail_exec", obj=obj, site=site, R=Rs,
                  reason=last_reason)
            failed_lst.append(list(key))
            if src in ("rim_pinch", "side_grip") and sig_now is not None:
                _e = mem.setdefault("__exec_fail__", {}) \
                         .setdefault(obj, {"sig": sig_now, "sites": []})
                _e["sites"].append(list(map(float, site)))
            continue
        _snap(adapter, "pre_approach", obj=obj, n_tried=n_tried, site=site,
              R=Rs)
        r = _approach_to_grasp(adapter, site, Rs, obj)
        if os.environ.get("LIBERO_DEBUG"):
            b0 = adapter.object_bounds(obj)
            print("DBG  post_app eef=%s force=%.1f bowl=%s" % (
                np.round(_eef(adapter), 3),
                adapter.contact_force_on_body(obj),
                np.round(b0["center"], 3)))
        if r is not None and r.get("mechanism") == "ik_unreachable" \
                and float(r.get("measures", {}).get("d_pos", 0.0)) > 0.0:
            # 接近跟踪饱和（直笛卡尔伺服在病态分支上停滞：残差 4-5cm、
            # 关节余量充足、站点多 IK 解可达——butter 实证）。恢复：
            # 关节路径直取站点（plan_arm_path 终点只要求无自碰撞，
            # 终点接触目标/支撑可容忍），再短伺服收尾接触事件。
            _stageln("grasp", n_tried, "approach_recover", True,
                     "关节路径重接近", src=src)
            allow = _site_grasp_geoms(adapter, obj) | _support_geoms(adapter,
                                                                     obj)
            path2 = plan_arm_path(adapter, site, grip=-1.0, target_rot=Rs,
                                  inflation=0.0, allow_contact=allow)
            if path2 is not None:
                r2 = execute_arm_path(adapter, path2, grip=-1.0,
                                      strict=False, watch_obj=obj)
                if r2 is None:
                    r = _approach_to_grasp(adapter, site, Rs, obj)
        if r is not None:
            last_reason = r.get("reason") or "approach_fail"
            _stageln("grasp", n_tried, "approach", False, last_reason,
                     src=src)
            _snap(adapter, "fail_approach", obj=obj, site=site, R=Rs,
                  reason=last_reason)
            failed_lst.append(list(key))
            if src in ("rim_pinch", "side_grip") and sig_now is not None:
                _e = mem.setdefault("__exec_fail__", {}) \
                         .setdefault(obj, {"sig": sig_now, "sites": []})
                _e["sites"].append(list(map(float, site)))
            _retreat(adapter, pre_site, Rs, inflation)
            continue
        _stageln("grasp", n_tried, "approach", True,
                 d=round(float(np.linalg.norm(_eef(adapter) - site)), 4),
                 src=src)
        _snap(adapter, "pre_close", obj=obj, n_tried=n_tried, site=site,
              R=Rs)

        _move_gripper(adapter, +1.0)
        if os.environ.get("LIBERO_DEBUG"):
            b0 = adapter.object_bounds(obj)
            print("DBG  post_close force=%.1f gripf=%.1f eef=%s bowl=%s" % (
                adapter.contact_force_on_body(obj),
                adapter.gripper_contact_force(),
                np.round(_eef(adapter), 3),
                np.round(b0["center"], 3)))
            _dbg_dump_contacts(adapter, "post_close")
            _dbg_force_balance(adapter, obj)

        # 支撑面高度（抬升前实测物体底面 = 与支撑接触零位）
        support_z = float(adapter.object_bounds(obj)["z_bottom"])
        _snap(adapter, "pre_lift", obj=obj, n_tried=n_tried, site=site,
              R=Rs, support_z=support_z)
        lifted, free = _lift_until_free(adapter, obj, support_z, hand)
        # 夹持力验证：物体脱离支撑时，手指必须仍与物体保持接触
        # （刚性夹持的力学不变量）。lifted > 0 但 force = 0 是假抓取。
        f_hold = float(adapter.contact_force_on_body(obj))
        if os.environ.get("LIBERO_DEBUG"):
            print(f"[dbg] grasp_verify lifted={lifted:.3f} "
                  f"free={free} f_hold={f_hold:.1f}", flush=True)
        if not free or f_hold < 0.5:
            _move_gripper(adapter, -1.0)
            last_reason = (f"闭爪后物体未随抬升脱离（Δbottom={lifted:.3f}）"
                           if not free else
                           f"抬升后夹持力丢失（f={f_hold:.1f}N）")
            _stageln("grasp", n_tried, "lift", False, last_reason,
                     lifted=round(lifted, 3), f_hold=round(f_hold, 1),
                     src=src)
            _snap(adapter, "fail_lift", obj=obj, site=site, R=Rs,
                  reason=last_reason, lifted=lifted, f_hold=f_hold)
            failed_lst.append(list(key))
            if src in ("rim_pinch", "side_grip") and sig_now is not None:
                _e = mem.setdefault("__exec_fail__", {}) \
                         .setdefault(obj, {"sig": sig_now, "sites": []})
                _e["sites"].append(list(map(float, site)))
            _retreat(adapter, pre_site, Rs, inflation)
            continue
        _stageln("grasp", n_tried, "lift", True, lifted=round(lifted, 3),
                 src=src)

        b1 = adapter.object_bounds(obj)
        tip = _eef(adapter)
        half_h = (float(b1["z_top"]) - float(b1["z_bottom"])) / 2.0
        mem[obj] = {
            "rel": float((b1["z_top"] + b1["z_bottom"]) / 2.0 - tip[2]),
            "half_h": half_h,
            "off_xy": [float(b1["center"][0] - tip[0]),
                       float(b1["center"][1] - tip[1])],
            "grasp_R": Rs.tolist()}
        mem.get("__exec_fail__", {}).pop(obj, None)
        return _ok(obj=obj, cand=cand,
                   grasp_pt=np.round(site, 3).tolist(),
                   lifted=round(lifted, 3))

    snap_dump(adapter, obj, last_reason)
    _stageln("grasp", n_tried, "grasp_all", False, last_reason)
    # 记录穷尽时的目标几何签名：签名不变的后继 grasp 调用直接快败，
    # 避免昂贵且无新信息的重枚举（见函数入口）。
    if sig_now is not None:
        enum_sig[obj] = sig_now
    return _fail(last_reason, "no_candidate", obj=obj, cand=cand,
                 n_tried=n_tried)


# ==================== 放置 ====================

def _upright_site_R(adapter) -> np.ndarray:
    """任务无关放置姿态：site 接近轴 = 世界 -z（从上方落座）；开合轴
    取当前开合轴向水平面的投影（最小转动扶正），投影退化时取横向
    轴投影；右手系第三轴由叉乘给出。"""
    Rc = np.asarray(adapter.get_site_rot(GRIP_SITE), float)
    x = None
    for col in (Rc[:, 0], Rc[:, 1]):
        v = np.array([col[0], col[1], 0.0], float)
        n = float(np.linalg.norm(v))
        if n > 1e-3:
            x = v / n
            break
    if x is None:
        x = np.array([1.0, 0.0, 0.0])
    z = np.array([0.0, 0.0, -1.0])
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def _bodies_contact(adapter, name_a: str, name_b: str) -> bool:
    """两个 BDDL 物体装配子树之间是否存在接触（几何接触事件，非力值）。"""
    m = adapter.mj_model
    d = adapter.mj_data
    ra = int(m.body_name2id(adapter._resolve_body(name_a)))
    rb = int(m.body_name2id(adapter._resolve_body(name_b)))
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = int(m.geom_bodyid[c.geom1])
        b2 = int(m.geom_bodyid[c.geom2])
        r1, r2 = int(m.body_rootid[b1]), int(m.body_rootid[b2])
        if (r1 == ra and r2 == rb) or (r1 == rb and r2 == ra):
            return True
    return False


def _resolve_target(adapter, target: str) -> Tuple[np.ndarray, float]:
    """目标名 → (xy 中心, 支撑/沿口 z)。物体用 support_point，region 用 site。"""
    if target in (getattr(adapter, "object_names", []) or []):
        p = np.asarray(adapter.support_point(target), float)
        return p[:2], float(p[2])
    from .libero_tasks import resolve_site_name
    site = resolve_site_name(adapter.mj_model, target)
    if site is None:
        raise KeyError(f"无法解析放置目标: {target}")
    p = np.asarray(adapter.get_site_pos(site), float)
    return p[:2], float(p[2])


def _release_preview_contained(adapter, obj: str, target: str) -> bool:
    """原位开爪丢放的 scratch 预览 + 容器几何包含判定。

    释放的目标是"In"关系被满足，不是手到某个固定位姿——工作区边界
    处最后数 cm 可能超出局部分支可达集（object:9 实证：修正伺服与
    RRT 重规划均失败，但物体实际落点仍在容器内）。仿真方式：复制
    live 状态，手指全开、手臂构型每步重置冻结（近似"原地松手"），
    步进至物体静止；包含判据全为几何量：物体质心水平偏移 + 物体
    水平半径 ≤ region 中心到容器壁内缘的最小边距，且物体底面低于
    壁顶（沿口）。
    """
    import mujoco
    m, live = _native_md(adapter)
    from .libero_tasks import resolve_site_name
    site = resolve_site_name(adapter.mj_model, target)
    if site is None:
        return False
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
    if sid < 0:
        return False
    region_xy = np.asarray(live.site_xpos[sid], float)[:2]
    region_z = float(live.site_xpos[sid][2])
    bid = int(m.site_bodyid[sid])
    walls = []
    for g in range(int(m.body_geomadr[bid]),
                   int(m.body_geomadr[bid] + m.body_geomnum[bid])):
        R = np.asarray(live.geom_xmat[g], float).reshape(3, 3)
        top = float(live.geom_xpos[g][2] + np.abs(R[2, :]) @ m.geom_size[g])
        c = np.asarray(live.geom_xpos[g], float)[:2]
        ex = np.abs(R[:2, :]) @ np.asarray(m.geom_size[g], float)
        walls.append((c, ex, top))
    if not walls:
        return False
    rim_top = max(w[2] for w in walls)
    # 壁/底分辨：壁顶明显高于 region 底（中点自适应尺度），矮者为
    # 容器底/沿口装饰，不构成内缘约束。
    real_walls = [w for w in walls if w[2] > 0.5 * (region_z + rim_top)]
    if not real_walls:
        return False
    # 内缘边距：region 中心到各壁 2D AABB 的最小水平距离
    margin = min(float(np.hypot(max(abs(float(region_xy[0] - c[0])) - ex[0], 0.0),
                                max(abs(float(region_xy[1] - c[1])) - ex[1], 0.0)))
                 for c, ex, _t in real_walls)
    obj_geoms = sorted(_site_grasp_geoms(adapter, obj))
    if not obj_geoms:
        return False
    b = adapter.object_bounds(obj)
    obj_r = max(float(b["half_x"]), float(b["half_y"]))
    sd = mujoco.MjData(m)
    sd.qpos[:] = np.asarray(live.qpos, float)
    sd.qvel[:] = 0.0
    sd.ctrl[:] = 0.0
    arm_adrs = [int(m.jnt_qposadr[j]) for j in _arm_joints(m)]
    arm_q0 = np.asarray(sd.qpos[arm_adrs], float)
    for j in _finger_joints(m):
        lo, hi = float(m.jnt_range[j][0]), float(m.jnt_range[j][1])
        sd.qpos[int(m.jnt_qposadr[j])] = hi if abs(hi) >= abs(lo) else lo
    mujoco.mj_forward(m, sd)
    n_steps = int(1.5 / float(m.opt.timestep))
    for _ in range(n_steps):
        mujoco.mj_step(m, sd)
        sd.qpos[arm_adrs] = arm_q0     # 手臂冻结：近似原地松手
        sd.qvel[arm_adrs] = 0.0
    cen = np.mean([sd.geom_xpos[g] for g in obj_geoms], axis=0)
    half_h = (float(b["z_top"]) - float(b["z_bottom"])) / 2.0
    bot = float(cen[2]) - half_h
    off = float(np.linalg.norm(cen[:2] - region_xy))
    return off + obj_r <= margin and bot <= rim_top


@_timed
@_timed
def push_to(adapter, obj: str, target: str, predicate: str = "On",
            mem: Optional[Dict[str, dict]] = None) -> Dict[str, Any]:
    """沿支撑面把物体推进目标区域——On/In 的第二种物理实现。

    抓取管线把候选空间实测耗尽（grasp 返回 no_candidate，即几何上无
    可夹候选）时，放置子目标退回推/拨归约：谓词只约束物体最终位置，
    推与放同样满足（TAMP 的标准析取规划：按可行性顺序尝试物理实现）。
    全程模型与在线测量，无 per-task 分支：
      - 推向 = 区域中心 xy − 物体质心 xy（世界系单位向量）；
      - 触点 = 物体 AABB 沿 −推向 的支持函数边界再退 tip_r（指尖球
        心位置）；
      - 腕姿态：开合轴竖直、接近轴水平沿 −u（双指上下叠放，下指压
        物体侧沿，上指悬空——叠放指在 Panda 腕限位下 IK 解充裕，实
        测可达；竖直接近轴解近零，工作区低位腕关节限界）。指落差
        drop = site_z − 下指球底 在 transit 终点构型上实测（手几何
        在线测量，非固定常数）；
      - transit 到触点正上方 **全场景可动物体最高顶 + 3·tip_r**
        （跨越走廊时不从可动物体之间穿：低位横向伸展的臂链必扫过
         途经物体——goal:5 实证，从高位越顶后链路不再接触任何物体）；
      - serve 下探到指高 = 物体底面 + drop + IK_TOL（跟踪精度地板，
        下指球底不探入支撑面）。下探与工作区远缘的侧推中 serve 的
        停滞/接触分类是预期物理（指尖已抵物体/目标在限位残余上），
        不判败——以**物体随动**为准：质心位移连续 3 步不足 0.3×步距
        （STATIONARY_MAX 同约定）判 contact_blocked（滑动受阻/脱触），
        世界已变由重规划轮从现状继续；
      - 终止：逐步查询谓词，满足即撤退（上抬 8cm）成功。
    """
    _snap_reset()
    try:
        xy, surface = _resolve_target(adapter, target)
    except KeyError as e:
        return _fail(str(e), "perception_fail", target=target)
    try:
        if bool(adapter.eval_subgoal(predicate, obj, target)):
            return _ok(obj=obj, target=target, note="已在目标区域")
    except Exception:
        pass

    hand = measure_hand(adapter)
    tip_r = float(hand["tip_r"])
    b = adapter.object_bounds(obj)
    obj_xy = np.array([float(b["center"][0]), float(b["center"][1])])
    z_bot, z_top = float(b["z_bottom"]), float(b["z_top"])
    d_vec = np.asarray(xy, float) - obj_xy
    dist = float(np.linalg.norm(d_vec))
    if dist < IK_TOL:
        return _ok(obj=obj, target=target, note="已在目标区域")
    u = d_vec / dist
    u3 = np.array([float(u[0]), float(u[1]), 0.0])
    # AABB（含旋转修正）沿 u 的支持半径：触点在物体背向区域一侧边缘
    r_u = float(b["half_x"]) * abs(float(u[0])) \
        + float(b["half_y"]) * abs(float(u[1]))
    contact_xy = np.array([obj_xy[0] - float(u[0]) * (r_u + tip_r),
                           obj_xy[1] - float(u[1]) * (r_u + tip_r)])
    # 腕姿态：x=开合轴竖直，z=接近轴水平沿 −u（y 由右手系给出）
    zax = -u3
    xax = np.array([0.0, 0.0, 1.0])
    yacc = np.cross(zax, xax)
    ny = float(np.linalg.norm(yacc))
    if ny < 1e-6:
        yacc = np.array([1.0, 0.0, 0.0])
    else:
        yacc /= ny
    xacc = np.cross(yacc, zax)
    R_push = np.column_stack([xacc, yacc, zax])

    # 指落差在线测量：transit 终点构型（scracth，不动真机）上取
    # site_z − 下指球底的最大值。腕姿态固定后该量只取决于手几何。
    import mujoco
    mm, live = _native_md(adapter)
    csid = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    arm_j = _arm_joints(mm)
    dof_adr = [int(mm.jnt_dofadr[j]) for j in arm_j]
    probe = np.array([contact_xy[0], contact_xy[1],
                      z_top + 3.0 * tip_r])
    m2, sd = _scratch(adapter, -1.0)
    pads = _pad_geom_set(m2)
    drop = None
    for q, _rp, _rr in _ik_in_tol_solutions(adapter, probe, R_push,
                                            arm_j, dof_adr,
                                            restarts=4 * IK_RESTARTS):
        for j, v in zip(arm_j, q):
            sd.qpos[int(m2.jnt_qposadr[j])] = float(v)
        mujoco.mj_forward(m2, sd)
        sz = float(sd.site_xpos[csid][2])
        d_lo = max(float(sz - (sd.geom_xpos[g][2] - m2.geom_size[g][0]))
                   for g in pads)
        if drop is None or d_lo > drop:
            drop = d_lo
    if drop is None:
        _snap(adapter, "fail_push_ik", obj=obj, target=target)
        snap_dump(adapter, obj, "推入腕姿态无可行构型")
        return _fail("推入腕姿态无可行构型", "ik_unreachable",
                     obj=obj, target=target)
    # 指高 = 物体底面（自身支撑面）+ 指落差 + 跟踪精度地板：下指球底
    # 不探入支撑面，下指球心自然落在物体侧沿高度
    press = np.array([contact_xy[0], contact_xy[1],
                      max(z_bot + max((z_top - z_bot) / 2.0, 1.1 * tip_r),
                          z_bot + drop + IK_TOL)])

    # transit：触点正上方、高过全场景可动物体最高顶（越顶走廊，纯
    # 几何量：可动物体顶 + 3·tip_r 手几何比例）
    try:
        hover_z = max(float(adapter.object_bounds(o)["z_top"])
                      for o in (getattr(adapter, "object_names", [])
                                or [obj])) + 3.0 * tip_r
    except Exception:
        hover_z = press[2] + 6.0 * tip_r
    hover = np.array([press[0], press[1], max(hover_z,
                                              press[2] + 2.0 * tip_r)])
    path = plan_arm_path(adapter, hover, grip=-1.0, target_rot=R_push)
    if path is None:
        _snap(adapter, "fail_push_plan", obj=obj, target=target)
        snap_dump(adapter, obj, "推入 transit 无可达无碰撞路径")
        return _fail("推入 transit 无可达无碰撞路径", "ik_unreachable",
                     obj=obj, target=target)
    r = execute_arm_path(adapter, path, grip=-1.0, strict=False)
    if r is not None:
        snap_dump(adapter, obj, r.get("reason") or "推入 transit 执行失败")
        return r
    # serve 下探到指高：差速 IK 从 transit 终点单分支下滑，在工作区
    # 远缘会卡进肘向错误的局部分支报 ik_unreachable（位姿多分支可达，
    # goal:5 探针实证：同点 24 重启 IK 到 z=0.85 有解而 serve 卡在
    # z=1.0）——沿用 place_at 的恢复模式：关节路径重规划换分支执行
    # （终点允许接触目标/支撑：按压就绪位本身在物体邻域）。
    r = serve(adapter, press, gripper=-1.0, k=K_FINE, target_rot=R_push,
              budget=600)
    if r is not None:
        allow = _site_grasp_geoms(adapter, obj) | _support_geoms(adapter, obj)
        path = plan_arm_path(adapter, press, grip=-1.0,
                             target_rot=R_push, allow_contact=allow)
        if path is not None:
            execute_arm_path(adapter, path, grip=-1.0, strict=False)
    _snap(adapter, "push_start", obj=obj, target=target, press=press,
          u=u.tolist(), drop=round(float(drop), 4))

    # 进给段：闭环推——每步从指尖当前位置朝**物体当前位置**指向区域
    # 中心的方向前进一个步距（圆盘类物体会被推得旋转/漂离初始进给
    # 线，开环直线进给会滑脱目标，goal:5 实证：物体偏移进给线 5cm
    # 后停滞）。步后查谓词与物体随动。
    step = 0.01
    stall = 0
    obj_prev = obj_xy.copy()
    steps_max = int(dist / step) + 60
    for _t in range(steps_max):
        bc = adapter.object_bounds(obj)
        obj_now = np.array([float(bc["center"][0]),
                            float(bc["center"][1])])
        d_now = np.asarray(xy, float) - obj_now
        d_n = float(np.linalg.norm(d_now))
        if d_n > 1e-6:
            u_now = d_now / d_n
            tip = _eef(adapter)
            tgt = np.array([tip[0] + float(u_now[0]) * step,
                            tip[1] + float(u_now[1]) * step,
                            press[2]])
            r = serve(adapter, tgt, gripper=-1.0, k=K_FINE,
                      target_rot=R_push, budget=200)
            if r is not None:
                # 同下探：差速 IK 单分支停滞 ≠ 不可达，关节路径恢复
                allow = _site_grasp_geoms(adapter, obj) \
                    | _support_geoms(adapter, obj)
                path = plan_arm_path(adapter, tgt, grip=-1.0,
                                     target_rot=R_push,
                                     allow_contact=allow)
                if path is not None:
                    execute_arm_path(adapter, path, grip=-1.0, strict=False)
        try:
            if bool(adapter.eval_subgoal(predicate, obj, target)):
                cur = _eef(adapter)
                goto(adapter, [cur[0], cur[1], cur[2] + 0.08],
                     gripper=-1.0)
                return _ok(obj=obj, target=target,
                           moved=round(dist, 3))
        except Exception:
            pass
        moved_step = float(np.linalg.norm(obj_now - obj_prev))
        obj_prev = obj_now
        if moved_step < 0.3 * step:
            # 推压中 pad 持续受力是预期物理（serve 的接触分类不判
            # 败）；唯一判据是物体是否随动。
            stall += 1
            if stall >= STATIONARY_MAX:
                _snap(adapter, "fail_push_stall", obj=obj, target=target,
                      stall=stall)
                snap_dump(adapter, obj,
                          f"推入受阻: 物体不随推进移动（{moved_step:.4f}/{step}）")
                return _fail("推入受阻: 物体不随推进移动",
                             "contact_blocked", obj=obj, target=target,
                             moved=round(float(np.linalg.norm(
                                 obj_now - obj_xy)), 3))
        else:
            stall = 0
    snap_dump(adapter, obj, "推入步数耗尽未到目标区域")
    return _fail("推入步数耗尽未到目标区域", "timeout", obj=obj,
                 target=target)


def place_at(adapter, obj: str, target: str, predicate: str = "On",
             mem: Optional[Dict[str, dict]] = None) -> Dict[str, Any]:
    """放到 target（On=物体顶面落座 / In=容器 region 释放）。全程
    模型与求解，无 per-task 分支：
      0. 抬升：按 footprint 内最高障碍几何竖直抬升（锁定当前抓姿），
         扶正后物体最低点悬垂更深，抬升量按**扶正后**几何预留；
      1. 放置点：xy 按偏心补偿让物体中心对准目标中心，姿态直接取
         扶正姿态 Rs_up——**扶正不做独立子目标**（原实现：原地重定向
         RRT，旋转只能在抓取点处进行，工作区边界处肘部重配置空间
         被桌/邻物挤占，终点 8~9 个 IK 解全碰撞、规划必败；object:7
         milk 实证）。扶正姿态作为放置 transit 的终点姿态交给
         RRT-Connect：旋转发生在去向放置点的空中路径上，那里肘部
          room 充足（放置点上方合成实证：全部容差解零碰撞）；
      2. 到位后按实测夹持几何（扶正姿态下重测）闭环修正放置点，
         消除 transit 中抓持滑移误差；On 时物体底=支撑面（接触几
         何零），In 时物体底高于容器沿口 IK_TOL；
      3. 下降伺服，On 以物体-目标接触事件复验，In 直接释放。
    """
    mrec = (mem or {}).get(obj)
    if mrec is None:
        return _fail("缺少抓取几何记忆（place_at 前须成功 grasp）",
                     "wrong_state", obj=obj)
    try:
        xy, surface = _resolve_target(adapter, target)
    except KeyError as e:
        return _fail(str(e), "perception_fail", target=target)

    # 被持物 root body：transit 规划的碰撞检查把物体按实测相对位姿
    # 重挂手上（焊接），规划看到持物 swept volume（见 _make_free_fn；
    # 无 free 关节的物体退回不焊，语义同旧）。
    import mujoco as _mj
    _wm, _ = _native_md(adapter)
    _ob = _mj.mj_name2id(_wm, _mj.mjtObj.mjOBJ_BODY,
                         adapter._resolve_body(obj))
    weld_obj = (int(_wm.body_rootid[_ob]) if _ob >= 0 else None)

    # ---- 1. 持物扶正 ----
    Rs_up = _upright_site_R(adapter)
    # ---- 0. 抬升到 clearance：碗仅几何脱离支撑数 mm，直接在柜体上
    # 方扶正/transit 会与周边障碍干涉。先按 footprint 内最高障碍几何
    # 竖直抬升（锁定当前抓姿；grasp 的 lift 已验证该运动可行）。
    # 扶正绕 site 旋转：物体-site 固连偏移在扶正姿态下可能悬垂更深
    # （侧抓时物体在手侧，扶正后物体吊在手下方），抬升量必须按
    # **扶正后**的物体最低点预留，否则扶正终点构型即碰撞（纯几何，
    # 无任务参数）。
    cl_z = _clearance_height(adapter, obj)
    bc = adapter.object_bounds(obj)
    R_now = np.asarray(adapter.get_site_rot(GRIP_SITE), float)
    eef_now = np.array(_eef(adapter), float)
    c_now = np.array([float(bc["center"][0]), float(bc["center"][1]),
                      (float(bc["z_top"]) + float(bc["z_bottom"])) / 2.0])
    off_s = R_now.T @ (c_now - eef_now)       # 物体在 site 系的固连偏移
    half_h0 = (float(bc["z_top"]) - float(bc["z_bottom"])) / 2.0
    bottom_up = float(eef_now[2] + (Rs_up @ off_s)[2] - half_h0)
    need = max(cl_z - float(bc["z_bottom"]),  # 当前底面越障
               cl_z - bottom_up,              # 扶正后最低点越障
               0.0)
    if need > 0.0:
        lift_tgt = eef_now.copy()
        lift_tgt[2] += need
        r = serve(adapter, lift_tgt, gripper=+1.0, k=K_FINE,
                  target_rot=R_now)
        if r is not None:
            return r

    # ---- 1. 扶正姿态下的放置点（几何预测；到位后实测闭环修正） ----
    # 物体-site 固连偏移在扶正姿态下的世界系分量：rel=物体中心在手
    # 下方深度，off_xy=横向偏心。transit 终点 = 放置点 + 该偏移补偿。
    off_up = Rs_up @ off_s
    rel = float(off_up[2])
    off_xy = [float(off_up[0]), float(off_up[1])]
    half_h = half_h0
    tip_xy = np.array([float(xy[0]) - off_xy[0],
                       float(xy[1]) - off_xy[1]])
    if predicate == "In":
        rim_z = surface
        try:
            rim_z = float(adapter.object_bounds(target)["z_top"])
        except Exception:
            pass
        tip_z = rim_z - rel + half_h + IK_TOL
    else:
        tip_z = surface - rel + half_h          # 物体底 = 支撑面
    final = np.array([tip_xy[0], tip_xy[1], tip_z])

    # ---- 2. 扶正+放置合并 transit：终点姿态 = 扶正姿态 ----
    # 焊接被持物（weld_obj）+ transit 跟踪包络余量（inflation）：规划
    # 看到持物 swept volume 且留足 OSC 稳态偏置空间——零余量盲规划的
    # 路径执行时被物理按压至关节停滞（goal:4 柜门面/桌面实证）。
    _pinfo: Dict[str, Any] = {}
    _infl0 = _transit_inflation(adapter)
    path_pl = plan_arm_path(adapter, final, grip=+1.0, target_rot=Rs_up,
                            info=_pinfo, weld_obj=weld_obj,
                            inflation=_infl0)
    if path_pl is None:
        # transit 余量是**碰撞自由的充分条件而非必要条件**（
        # _transit_inflation：live 手相对规划手的最坏跟踪包络×2）——
        # 全余量下无可行路径不蕴涵零余量下不可行（object:7 milk 实
        # 证：焊接持物 swept volume 贴障碍 ~1cm 内的直达路径在全余量
        # 下 53s/6000 迭代双树饥饿告败，零余量下同一对起终点 1.2s 即
        # 得 n=2 近平直路径；末端自由度冗余但容差解唯一，旋转+长程
        # 耦合搜索是次要因素）。处置＝**余量减半阶梯**取"能出解的最
        # 大余量"：全余量→半余量→零余量，逐级降级规划安全性而非一
        # 步放弃——零余量路径只保证模型无穿透，执行期跟踪偏置可能把
        # 持物按压上障碍（goal:4 柜门面实证 7-11N），该失稳由既有的
        # 非严格执行（strict=False + watch_obj + allow_contact）与
        # 下方 live 重规划阶梯承接。fallback 只在全余量首规划失败时
        # 触发；首规划成功的路径（含全部回归任务的成功放置）零影响。
        for _infl in (_infl0 * 0.5, 0.0):
            path_pl = plan_arm_path(adapter, final, grip=+1.0,
                                    target_rot=Rs_up, info=_pinfo,
                                    weld_obj=weld_obj, inflation=_infl)
            if path_pl is not None:
                break
    if path_pl is None:
        _snap(adapter, "fail_place_plan", obj=obj, target=target,
              info=_pinfo)
        return _fail(f"放置点无可达无碰撞路径: {obj}→{target} {_pinfo}",
                     "ik_unreachable", target=target)

    if predicate != "In":
        def _contact_verify(end):
            return _bodies_contact(adapter, obj, target), {}
    else:
        _contact_verify = None
    # 放置路径非严格执行：末点残余（OSC 稳态亚毫米偏置）由后续
    # "越过接触面一个跟踪门"的接触事件 serve 闭环消除；严格收敛会
    # 在接触前亚毫米处关节停滞（reach_limit），而接触尚未发生。
    r = execute_arm_path(adapter, path_pl, grip=+1.0, strict=False,
                         watch_obj=obj, allow_contact=True)
    if r is not None:
        # transit 执行失败≠目标不可达（与下方修正恢复同语义）：失稳
        # 机理是持物滑移在开口预算内累积、把焊接快照外的碗压上障
        # 碍（goal:4 实证：7-11N 按压、差 13-19mm 关节停滞），或 OSC
        # 伺服在伸展分支局部停滞。处置：以 live 实测构型重规划——
        # weld_obj 每次检查重读实时相对位姿，滑移后的真实偏置被纳
        # 入 swept volume，新路径绕开按压态；按压中的起点常超出浅
        # 接触门不可规划，故败后竖直退回复中性构型（卸载接触、抬升
        # 种子远离卡死分支）再规划重入；全败才放弃。
        _done = False
        for _round in range(2):
            if _round == 1:
                _cur = _eef(adapter)
                r2 = serve(adapter, [_cur[0], _cur[1],
                                     _cur[2] + 0.10], gripper=+1.0,
                           k=K_FINE)
                if r2 is not None:
                    break
            for _st, _mn in ((RRT_STEP, RRT_MAX_NODES),
                             (RRT_STEP / 4.0, 4 * RRT_MAX_NODES)):
                _pinfoT: Dict[str, Any] = {}
                pathT = plan_arm_path(adapter, final, grip=+1.0,
                                      target_rot=Rs_up, info=_pinfoT,
                                      step=_st, max_nodes=_mn,
                                      weld_obj=weld_obj,
                                      inflation=_transit_inflation(adapter))
                if pathT is None:
                    continue
                r = execute_arm_path(adapter, pathT, grip=+1.0,
                                     strict=False, watch_obj=obj,
                                     allow_contact=True)
                if r is None:
                    _done = True
                    break
            if _done:
                break
        if not _done:
            return r

    # ---- 3. 实测闭环修正：扶正姿态下重测夹持几何，消除 transit 滑移 ----
    b0 = adapter.object_bounds(obj)
    tip0 = _eef(adapter)
    half_h = (float(b0["z_top"]) - float(b0["z_bottom"])) / 2.0
    rel = float((b0["z_top"] + b0["z_bottom"]) / 2.0) - float(tip0[2])
    off_live = [float(b0["center"][0]) - float(tip0[0]),
                float(b0["center"][1]) - float(tip0[1])]
    corr = float(np.linalg.norm(np.array(off_live)
                                 - np.array(off_xy)))
    if corr > IK_TOL:
        final = np.array([float(xy[0]) - off_live[0],
                          float(xy[1]) - off_live[1], tip_z])
        if predicate == "In":
            final[2] = rim_z - rel + half_h + IK_TOL
        else:
            final[2] = surface - rel + half_h
        r = serve(adapter, final, gripper=+1.0, k=K_FINE,
                  target_rot=Rs_up)
        if r is not None and r.get("mechanism") == "ik_unreachable":
            # 局部伺服在关节限位停滞≠目标不可达：差速 IK 从当前构型
            # 下滑会卡进最近的 IK 分支（肘朝错侧），而端点 IK 多重启
            # 动已证 final 有解（object:9 实证：修正伺服 reach_limit
            # 放弃、常规/细步长 RRT 均 0.5s 告败——IK 重启种子半数以
            # live 构型为基准，live 卡死在伸展分支时种子全体被污染；
            # 同位姿在 home 种子下探出 14 个无碰撞容差解）。处置：
            # 先常规全局重规划；败则竖直退回复中性构型（抬升种子远离
            # 卡死分支，持物竖直抬升 grasp 的 lift 已验证可行）再规划
            # 重入；再败才放弃。窄走廊由降步长提节点重试覆盖。
            _done = False
            for _st, _mn in ((RRT_STEP, RRT_MAX_NODES),
                             (RRT_STEP / 4.0, 4 * RRT_MAX_NODES)):
                _pinfo2: Dict[str, Any] = {}
                path2 = plan_arm_path(adapter, final, grip=+1.0,
                                      target_rot=Rs_up, info=_pinfo2,
                                      step=_st, max_nodes=_mn,
                                      weld_obj=weld_obj,
                                      inflation=_transit_inflation(adapter))
                if path2 is None:
                    continue
                r = execute_arm_path(adapter, path2, grip=+1.0,
                                     strict=True, watch_obj=obj,
                                     allow_contact=True)
                if r is None:
                    _done = True
                    break
            if not _done and r is not None:
                _cur = _eef(adapter)
                r2 = serve(adapter, [_cur[0], _cur[1],
                                     _cur[2] + 0.10], gripper=+1.0,
                           k=K_FINE)
                if r2 is None:
                    for _st, _mn in ((RRT_STEP, RRT_MAX_NODES),
                                     (RRT_STEP / 4.0, 4 * RRT_MAX_NODES)):
                        path3 = plan_arm_path(adapter, final, grip=+1.0,
                                              target_rot=Rs_up, step=_st,
                                              max_nodes=_mn,
                                              weld_obj=weld_obj,
                                              inflation=_transit_inflation(
                                                  adapter))
                        if path3 is None:
                            continue
                        r = execute_arm_path(adapter, path3, grip=+1.0,
                                             strict=True, watch_obj=obj,
                                             allow_contact=True)
                        if r is None:
                            _done = True
                            break
                else:
                    r = r2
        if r is not None and predicate == "In" \
                and _release_preview_contained(adapter, obj, target):
            # 精修不可达但原位松手的落点预览仍满足 In 几何：释放。
            # 释放的目标是被满足的关系而非手到位姿（见
            # _release_preview_contained）。
            if os.environ.get("LIBERO_DEBUG"):
                print("[dbg] place_at 精修不可达→预览包含→原位释放",
                      flush=True)
            _move_gripper(adapter, -1.0)
            cur = _eef(adapter)
            goto(adapter, [cur[0], cur[1], cur[2] + 0.08],
                 gripper=-1.0)
            try:
                b = adapter.object_bounds(obj)
                dxy = float(np.linalg.norm(np.asarray(b["center"], float)[:2]
                                          - xy))
            except Exception:
                dxy = -1.0
            return _ok(obj=obj, target=target,
                       tip_z=round(float(tip_z), 3),
                       dxy_to_target=round(dxy, 3),
                       released="preview_contained")
        if r is not None:
            return r
        off_xy = off_live
    if predicate != "In":
        # 落座是接触事件：最终伺服目标在几何接触面下方一个跟踪门，
        # 物理接触必先于到达目标发生，接触验证立即放行（不会被
        # slow_budget 截断在 3mm 悬空）。
        descend = _site_step_envelope(adapter)
        contact_tgt = np.array([final[0], final[1], final[2] - descend])
        r = serve(adapter, contact_tgt, gripper=+1.0, k=K_FINE,
                  target_rot=Rs_up, verify=_contact_verify)
        if r is not None:
            return r
        # 支撑面释放：开爪全程保持下压伺服，摩擦抵抗开爪侧向推力，
        # 消除落座滑移（见 _open_on_support）。
        _open_on_support(adapter, contact_tgt)
    else:
        # 容器释放：慢开爪。快开（0.2s 全行程）的摩擦冲量把轻物体
        # 侧向抛出——object:7 milk 实证：修正闭环把奶盒居正到 ~5mm
        # 后，0.2s 快开仍被甩偏 2cm（10cm/s 侧向），落点压篮沿口翻倒、
        # In 谓词不满足；慢开让开爪过程保持准静态（正压力渐退、物体
        # 在摩擦力不足时由重力铅直滑落而非被抛出），这是 _open_on_
        # support（On 谓词下压伺服抵抗开爪侧推力）在"无支撑可压"的
        # 容器释放上的对偶处置。分档行程与档间 settle 步数是离散化
        # 约定（同数值积分的步长选取），非物理经验值。
        for _g in (-0.25, -0.5, -0.75, -1.0):
            _move_gripper(adapter, _g)
            for _ in range(25):
                adapter.step(np.zeros(adapter.action_dim))

        # 释放落位须等物理 settle 后才可交付判定：容器内释放是自由
        # 落体+沿口滑移入位过程（释放几何只保证奶盒底在沿口 IK_TOL
        # 上方，入位靠掉落中滑移），runner 在技能返回**立即**求值谓
        # 词，未落位即误败重规划（object:7 milk 实证：释放瞬间奶盒
        # 压沿口、In=False；同状态 settle 后奶盒滑入篮内、In=True，
        # 全量 attempt 1 因此谓词误败烧毁）。处置：释放后空步进至物
        # 体中心位移收敛（数值收敛门，同优化器容差语义）或步数预算
        # 耗尽——预算耗尽照常返回，语义不劣于现状。
        def _settle_in() -> None:
            try:
                _prev = np.asarray(
                    adapter.object_bounds(obj)["center"], float)
            except Exception:
                return
            for _ in range(300):
                adapter.step(np.zeros(adapter.action_dim))
                try:
                    _c = np.asarray(
                        adapter.object_bounds(obj)["center"], float)
                except Exception:
                    return
                if float(np.linalg.norm(_c - _prev)) < 1e-4:
                    break
                _prev = _c
    cur = _eef(adapter)
    goto(adapter, [cur[0], cur[1], cur[2] + 0.08], gripper=-1.0)
    if predicate == "In":
        _settle_in()

    try:
        b = adapter.object_bounds(obj)
        dxy = float(np.linalg.norm(np.asarray(b["center"], float)[:2] - xy))
    except Exception:
        dxy = -1.0
    return _ok(obj=obj, target=target, tip_z=round(float(tip_z), 3),
               dxy_to_target=round(dxy, 3))


# ==================== 开合（抽屉/柜门/微波炉） ====================

def _joint_decl(adapter, joint: str) -> Dict[str, Any]:
    m = adapter.mj_model
    d = adapter.mj_data
    jid = int(m.joint_name2id(joint))
    bid = int(m.jnt_bodyid[jid])
    jtype = int(m.jnt_type[jid])
    xmat = np.asarray(d.body_xmat[bid], float).reshape(3, 3)
    axis = xmat @ np.asarray(m.jnt_axis[jid], float)
    n = float(np.linalg.norm(axis))
    axis = axis / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])
    point = (np.asarray(d.body_xpos[bid], float)
             + xmat @ np.asarray(m.jnt_pos[jid], float))
    return {"jid": jid, "bid": bid, "joint": joint,
            "body": str(m.body_id2name(bid)),
            "jtype": "revolute" if jtype == 3 else "prismatic",
            "axis": axis, "point": point,
            "range": (float(m.jnt_range[jid][0]), float(m.jnt_range[jid][1])),
            "qpos_adr": int(m.jnt_qposadr[jid])}


def _read_q(adapter, decl: Dict[str, Any]) -> float:
    return float(adapter.mj_data.qpos[decl["qpos_adr"]])


def _subtree_geoms(adapter, bid: int) -> List[int]:
    """运动 body 子树的全部 geom id（wrapper/native 同 id 空间）。"""
    m = adapter.mj_model
    subtree = {int(bid)}
    grew = True
    while grew:
        grew = False
        for b in range(m.nbody):
            if b not in subtree and int(m.body_parentid[b]) in subtree:
                subtree.add(b)
                grew = True
    geoms: List[int] = []
    for b in subtree:
        for g in range(int(m.body_geomadr[b]),
                       int(m.body_geomadr[b] + m.body_geomnum[b])):
            geoms.append(g)
    return geoms


def _fixture_geoms(adapter, decl: Dict[str, Any]) -> set:
    """夹具（关节体）全部 geom：接近段的"到达"接触证据只认这些 geom，
    途经桌面杂物的擦碰不停车（goal:0 实证：0.2N 盘子擦碰把接近段停在
    把手前 7mm，闭爪捏空）。"""
    return set(_subtree_geoms(adapter, decl["bid"]))


def _handle_geom(adapter, decl: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """作用 geom 求解（返回其中心与 id）。名字含 handle/knob 的 geom
    优先；否则 revolute 取子树上力臂（离轴距离）最大者，prismatic 取
    沿开合行进方向最外凸 face 者（引领面：把手/抽屉面板前沿，背板
    恒被排除）。纯几何，无任务参数。"""
    m, d = adapter.mj_model, adapter.mj_data
    bid = decl["bid"]
    geoms = _subtree_geoms(adapter, bid)
    if not geoms:
        return np.asarray(d.body_xpos[bid], float), -1

    def _lever(g: int) -> float:
        p = np.asarray(d.geom_xpos[g], float)
        if decl["jtype"] == "prismatic":
            # 把手在开合行程的引领面：沿行进方向最外凸的 face 坐标。
            # 力臂/轴投影会把抽屉背板误判为把手（goal:0 实证：背板在
            # 面板后方 17cm，approach 够不着、手指压面板空转）。
            D = np.asarray(decl["axis"], float) * decl.get("open_dir", 1.0)
            R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
            ext = np.abs(R) @ np.asarray(m.geom_size[g], float)
            return float(np.dot(p, D) + np.dot(ext, D))
        v = p - decl["point"]
        return float(np.linalg.norm(v - decl["axis"] * np.dot(v, decl["axis"])))

    named = [g for g in geoms
             if any(h in str(m.geom_id2name(g) or "").lower()
                    for h in ("handle", "knob", "bar"))]
    g_best = max(named or geoms, key=_lever)
    return np.asarray(d.geom_xpos[g_best], float), int(g_best)


def _handle_point(adapter, decl: Dict[str, Any]) -> np.ndarray:
    return _handle_geom(adapter, decl)[0]


def _geom_obb_top(adapter, g: int) -> float:
    """geom OBB 的世界系 z 向上表面：中心 z + Σ|R[2,j]|·halfsize_j。
    网格级精确（LIBERO 家具 geom 均为 box），替代中心估计。"""
    m, d = adapter.mj_model, adapter.mj_data
    if g < 0:
        return float("nan")
    R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
    size = np.asarray(m.geom_size[g], float)
    return float(d.geom_xpos[g][2] + np.abs(R[2, :]) @ size)


def _geom_world_aabb(adapter, g: int, inflate: float = 0.0) -> np.ndarray:
    """geom OBB 的世界轴对齐包围盒 [[lo],[hi]]，外扩 inflate。用于
    "接触是否发生在作用 geom 邻域"的位置门控：夹具 body 常把多个
    旋钮/装饰 geom 焊在同一刚体上（goal:7 灶台四旋钮同 body），
    geom 级到达判定会把邻居/外壳的擦碰误判为到达（停在作用点
    1.5cm 外闭爪夹空），必须按接触点位置再分一遍。"""
    m, d = _native_md(adapter)
    R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
    ext = np.abs(R) @ np.asarray(m.geom_size[g], float) + inflate
    c = np.asarray(d.geom_xpos[g], float)
    return np.stack([c - ext, c + ext])


def _site_ik_viable(adapter, site: np.ndarray, rot: Optional[np.ndarray],
                    allow_geoms: set, dbg: bool = False) -> bool:
    """位姿可达性预审：存在至少一个容差内 IK 解，其
    (a) 最紧关节限位裕度 ≥ 2% 行程——OSC 伺服只在贴限位（裕度≈0，
        工作空间边界）的深伸构型上跟踪饱和（goal:7 顶压角点实证：
        残差 1cm 处停摆、pad 悬空）；裕度 4% 的构型跟踪正常，0.05
        的初版门槛把可达大力臂角点误杀（g19 西角点 margin=0.043
        实证）。2% 是算法精度下限（仍拒绝收敛到限位的解堆）。
    (b) 终点构型除 allow_geoms（site 所在 geom，到达即接触）外无
        异物体碰撞。
    角点候选按力臂降序逐个送审，第一个通过者入选。"""
    import mujoco
    m, d = _native_md(adapter)
    joints = _arm_joints(m)
    dof_adr = [int(m.jnt_dofadr[j]) for j in joints]
    robot = _robot_geom_set(m)
    Rt = None if rot is None else np.asarray(rot, float).reshape(3, 3)
    n_sols = 0
    for q, _rp, _rr in _ik_in_tol_solutions(adapter, site, Rt, joints,
                                            dof_adr):
        n_sols += 1
        marg = _limit_margin(m, q, joints)
        if marg < 0.02:
            if dbg:
                print(f"[dbg] knob_corner reject: margin={marg:.3f}",
                      flush=True)
            continue
        sd = mujoco.MjData(m)
        sd.qpos[:] = np.asarray(d.qpos, float)
        for j, v in zip(joints, q):
            sd.qpos[int(m.jnt_qposadr[j])] = float(v)
        mujoco.mj_forward(m, sd)
        bad = set()
        for ci in range(sd.ncon):
            c = sd.contact[ci]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in robot) != (g2 in robot):
                o = g2 if g1 in robot else g1
                if o not in allow_geoms:
                    bad.add(str(mujoco.mj_id2name(
                        m, mujoco.mjtObj.mjOBJ_GEOM, o)))
        if bad:
            if dbg:
                print(f"[dbg] knob_corner reject: collide={sorted(bad)[:3]}",
                      flush=True)
            continue
        if dbg:
            print(f"[dbg] knob_corner accept: margin={marg:.3f}", flush=True)
        return True
    if dbg and not n_sols:
        print("[dbg] knob_corner reject: no IK solutions", flush=True)
    return False


def _knob_halfgap_for(adapter, g_site: int,
                      Rs: Optional[np.ndarray],
                      hand: Dict[str, Any]) -> float:
    """knob_tab 两种腕姿的进入构型半开口：Rs 给定时按 Rs[:,0]（捏夹
    轴）；Rs=None（顶压，沿用当前腕姿）按现测开合轴。详见
    _knob_pinch_halfgap。"""
    if Rs is not None:
        return _knob_pinch_halfgap(adapter, g_site, Rs[:, 0], hand)
    import mujoco
    m, d0 = _native_md(adapter)
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
    sR = np.asarray(d0.site_xmat[sid], float).reshape(3, 3)
    return _knob_pinch_halfgap(adapter, g_site, sR @ hand["u_site"],
                               hand, press=True)


def _knob_pinch_halfgap(adapter, g_h: int, u: np.ndarray,
                        hand: Dict[str, Any], press: bool = False) -> float:
    """旋钮薄耳片的进入构型目标半开口。

    press=False（侧向捏夹）：耳片半厚 + 1.5·tip_r（接触距 + 下行
    净距裕量）——钳口收到比目标略宽，进入后再闭爪。
    press=True（顶压窄盒）：盒顶面半宽 − 0.5·tip_r（钳口全张时
    ±a_open 实测 ~4cm 远超盒宽，pad 落在盒两侧永远压不到顶面——
    goal:7 实证），收到盒宽以内 pad 才压在顶面上；收到 a_closed
    为止（pad 在盒顶中心并扰无害）。
    全部 geom 现测，无任务常数。"""
    m, d = _native_md(adapter)
    R = np.asarray(d.geom_xmat[g_h], float).reshape(3, 3)
    sizes = np.asarray(m.geom_size[g_h], float)
    half_u = float(np.sum(np.abs(R.T @ np.asarray(u, float)) * sizes))
    if press:
        return max(half_u - 0.5 * hand["tip_r"], hand["a_closed"])
    return half_u + 1.5 * float(hand["tip_r"])


def _manifold_point(decl: Dict[str, Any], h0: np.ndarray, q0: float,
                    q: float) -> np.ndarray:
    """作用点随关节配置的轨迹（revolute=Rodrigues，prismatic=平移）。"""
    dq = q - q0
    if decl["jtype"] == "prismatic":
        return h0 + decl["axis"] * dq
    v = h0 - decl["point"]
    k = decl["axis"]
    return decl["point"] + (v * np.cos(dq)
                            + np.cross(k, v) * np.sin(dq)
                            + k * np.dot(k, v) * (1.0 - np.cos(dq)))


def _axis_angle(k: np.ndarray, ang: float) -> np.ndarray:
    """轴角 → 旋转矩阵（Rodrigues）。"""
    k = np.asarray(k, float)
    k = k / max(float(np.linalg.norm(k)), 1e-12)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + np.sin(ang) * K
            + (1.0 - np.cos(ang)) * (K @ K))


def _follow_manifold(adapter, decl, h0, q0, q_goal, target, steps_max,
                     track_rot: bool = False):
    """抓住作用点后沿流形跟随到 q_goal。到位 = 残差进入在线测量的
    q 跟踪噪声 3σ；停滞 = q 无进展时沿用统计接触/限位判据。

    跟随由指令 θ 驱动而非实测 q 驱动：θ 每步前进，但领先 |θ−q| 不
    超上限。静摩擦下被驱动件需持续滑移才能拖动（纯挤压的接触力矩被
    捏夹摩擦阻力矩抵消，goal:7 旋钮实证：手冻结挤压 300 tick，q 仅
    蠕动 0.002rad）；θ 驱动让手保持绕行拖拽，q 跟上后领先自然缩小，
    被挡住时手停在领先上限处维持按压，由停滞判据收尾。

    步距按"手空间每 tick 位移 ≈ 4mm"换算（s = 流形单位 q 位移）：
    prismatic s=1 → 步距 0.01（1cm/tick，兼容柜门抽屉原行为）；
    旋钮 s=力臂≈5cm → 步距 ≈0.07rad，否则固定 1cm 步距对应的
    tol=5mm 比位移大一个量级，手从不追踪、θ 空转。接近目标 1.5 个
    大步后切回 0.01 细步（大步量化误差会跨过 1% 行程到达带），θ 也
    永不越过 q_goal。领先上限按位移 8mm 换算（lead×力臂 = 手相对
    接触斑的绕行量，限制在不脱接触尺度内）。

    track_rot（旋钮顶压/捏夹用）：腕姿态随关节角同步绕轴旋转
    R(θ−q0, axis)·R0。默认跟随不伺服姿态（OSC 保持腕姿），但接触
    面随刚体 yaw 转、pad 面不跟转时，pad 棱边挖进接触面形成几何
    楔死——goal:7 实证：27N 捏夹 + θ 驱动下 servo 停滞在目标前
    9.5mm、q 纹丝不动；姿态跟随后 pad 面与盒面保持平行，摩擦拖
    矩才能传出。"""
    import mujoco
    forward = 1.0 if q_goal > q0 else -1.0
    s = float(np.linalg.norm(
        _manifold_point(decl, h0, q0, float(q0) + 1.0)
        - _manifold_point(decl, h0, q0, float(q0))))
    s = max(s, 1e-6)
    step_big = max(0.01, 0.004 / s)
    lead_max = 0.008 / s
    theta = float(q0)
    R0 = None
    if track_rot:
        R0 = np.asarray(adapter.get_site_rot(GRIP_SITE), float).reshape(3, 3)
    qhist: List[float] = []
    stationary = 0
    for t in range(steps_max):
        q = _read_q(adapter, decl)
        qhist.append(q)
        residual_q = forward * (q - q_goal)
        # 到位包络 = max(3σ, 行程 1%)：σ 是在线测量的跟踪噪声，但关节
        # 平滑收敛时拟合残差 σ 可比残差还小一个量级，纯 3σ 判据漏检
        # （goal:4 柜门实证：q=-0.1557/目标 -0.156，残差 3e-4 未进
        # 包络，下一 tick 手柄转出可达域 goto 失败直接判败）。1% 行程
        # 是算法精度下限；卡死在起点残差 ≈ 全行程，任何 <1% 行程的
        # 下限都不会把"根本没动"误判为到达（双侧 abs 判定）。
        arrive_tol = 0.01 * abs(float(q_goal) - q0)
        if len(qhist) >= WIN:
            arr = np.array(qhist[-WIN:], float)
            x = np.arange(WIN)
            slope, icept = np.polyfit(x, arr, 1)
            sig = float((arr - (slope * x + icept)).std()) + 1e-7
            arrive_tol = max(arrive_tol, CONF_K * sig)
            # 到达是双侧判定：|残差| 进噪声包络。单侧 <= 会把"根本没动"
            # （残差 = -总行程，大负数）误判为到达——跟随 30 tick 预热
            # 后卡死的关节残差恒为 -|q_goal-q0|，单侧判定直接假成功
            # （goal:0 木柜实证：手指未抓住把手时 q 不动，每段假成功
            # 续程循环 300s 空转）。
            if abs(residual_q) <= arrive_tol:
                _move_gripper(adapter, -1.0)
                cur = _eef(adapter)
                goto(adapter, [cur[0], cur[1], cur[2] + 0.08], gripper=-1.0)
                return _ok(target=target, q=round(q, 4),
                           q_goal=round(float(q_goal), 4), steps=t)
            se = sig / max(float(np.sqrt(np.sum((x - x.mean()) ** 2))), 1e-9)
            if forward * slope / max(se, 1e-9) < TSTAT:
                stationary += 1
                fnow = float(adapter.gripper_contact_force())
                if stationary >= STATIONARY_MAX:
                    return _fail(
                        f"关节卡死在 q={q:.3f}（目标 {q_goal:.3f}，"
                        f"指令 θ={theta:.3f} 领先 "
                        f"{forward * (theta - q):.3f}）",
                        "contact_blocked", target=target, q=q,
                        q_goal=float(q_goal))
            else:
                stationary = 0
        # 指令推进：θ 领先实测 q、上限 lead_max、永不越过 q_goal；距
        # 目标 1.5 个大步后切细步（大步量化误差 > 1% 行程到达带）。
        # track_rot（旋钮）用 θ 驱动：静摩擦下被驱动件需持续滑移拖拽，
        # q 驱动会互相等死（见 docstring）。非旋钮（柜门/抽屉等自由
        # 运动件）保持 q 驱动——θ 抢跑会让 aim 甩开末端，低速件上
        # servo 差几毫米永远进不了 tol，预算耗尽判 slow_budget（goal:0
        # 抽屉回归实证）。
        if track_rot:
            step_q = 0.01 if abs(q_goal - theta) < 1.5 * step_big \
                else step_big
            if forward * (theta - q) < lead_max:
                theta = theta + forward * min(step_q, abs(q_goal - theta))
            if forward * (theta - q_goal) > 0:
                theta = float(q_goal)
        else:
            step_q = 0.01
            theta = q + forward * step_q
        tip = _eef(adapter)
        aim = _manifold_point(decl, h0, q0, theta)
        m_q = _manifold_point(decl, h0, q0, q)
        offset = tip - m_q
        # 跟踪容忍必须显著小于每 tick 位移（goal:7 实证：revolute 力
        # 臂 5cm 时每 tick 仅 0.5mm，固定 tol=5mm 让手从不追踪、θ 空
        # 转 lead_max 后判卡死）；取每 tick 位移一半。非旋钮路径维持
        # 原固定 5mm（柜门抽屉原行为，回归基线）。
        if track_rot:
            step_next = _manifold_point(decl, h0, q0,
                                        theta + forward * step_q)
            step_dist = float(np.linalg.norm(step_next - aim))
            tol_track = min(0.005, max(1e-4, 0.5 * step_dist))
        else:
            tol_track = 0.005
        Rtrk = None
        if R0 is not None:
            Rtrk = _axis_angle(decl["axis"], theta - float(q0)) @ R0
        r = goto(adapter, aim + offset, gripper=+1.0, k=K_FINE,
                 tol=tol_track, timeout=80, target_rot=Rtrk)
        if r is not None:
            # goto 失败（手柄随关节开到底转出可达域）不等于关节未到位：
            # 先按同一到位包络复核，已到位则松爪撤退判成功——继续跟随
            # 已无意义，子目标是关节状态不是手的轨迹（goal:4 实证）。
            if abs(residual_q) <= arrive_tol:
                _move_gripper(adapter, -1.0)
                cur = _eef(adapter)
                goto(adapter, [cur[0], cur[1], cur[2] + 0.08], gripper=-1.0)
                return _ok(target=target, q=round(q, 4),
                           q_goal=round(float(q_goal), 4), steps=t,
                           note="goto 失败但关节已到位")
            return _fail(f"流形跟随失败: {r['reason']}",
                         "contact_blocked", target=target, q=q)
    return _fail(f"流形跟随超时 q={_read_q(adapter,decl):.3f}"
                 f"（目标 {q_goal:.3f}）", "timeout", target=target,
                 q_goal=float(q_goal))


def _articulate_site(adapter, decl, h0, g_h, hand):
    """作用点与目标接近姿态（纯几何，无任务分支）。

    revolute（柜门/旋钮）：顶落式——指尖从上方落到作用 geom OBB 顶
    沿（site_z = 顶 − (b_depth+tip_r)），姿态沿用当前腕姿态。

    prismatic 且开合轴近水平（抽屉）：正面水平接近 + 最薄截面捏夹。
    抽屉把手常上下堆叠（goal:0 木柜实证：相邻横杆间距 7.4cm、掌心
    包络 ~9cm），掌心朝下从上方下探会先撞上层把手（垂直通道被遮），
    侧向进入也被同 x 跨度上层把手挡住；唯一可行构型是 gripper 接近
    轴沿水平指向本体（site +z = −行进方向）、手指沿 geom 最薄截面
    轴合拢捏横杆两面、掌心停在把手前方开阔区。横杆从张开的两指之间
    穿过（开放半间距 a_open 实测 ~4cm > 杆厚），接近段无接触可达——
    到达语义是几何到位（止挡/停滞按到达处理），随后闭爪把 pad 压上
    横杆最薄截面两面，摩擦拖动。site 嵌到前缘内，吸收 OSC 稳态偏置。"""
    m, d = _native_md(adapter)
    top = _geom_obb_top(adapter, g_h)
    if not np.isfinite(top):
        top = float(h0[2])
    site = np.array([h0[0], h0[1],
                     top - (hand["b_depth"] + hand["tip_r"])])
    Rs = None
    D = np.asarray(decl["axis"], float) * decl.get("open_dir", 1.0)
    # 竖直转轴的旋钮薄耳片：顶面比指尖球窄（min 水平半宽 < tip_r），
    # 且同按钮体常把整排旋钮/顶板焊在一起（goal:7 灶台实证：耳片
    # 与悬挑顶板 g19 盒体重叠、被压在正下方），顶落耳片与水平捏夹
    # 均被同体几何封死。改顶压式：site 取按钮体**最顶 geom 的顶面**
    # 上离转轴最远的角点（矩形面上离轴最远点必为角点）——大顶面
    # 暴露无遮拦，力臂最大化（摩擦力矩 = μN·r），闭爪压住后流形跟
    # 随的切向拖动力经摩擦驱动整体旋转。柜门/门把手顶面宽于指尖球
    # 且无同体悬挑，顶落式不受影响。
    knob_tab = False
    if decl["jtype"] == "revolute" and g_h >= 0 \
            and abs(float(np.asarray(decl["axis"], float)[2])) > 0.7:
        Rg = np.asarray(d.geom_xmat[g_h], float).reshape(3, 3)
        eg = np.abs(Rg) @ np.asarray(m.geom_size[g_h], float)
        knob_tab = min(float(eg[0]), float(eg[1])) < float(hand["tip_r"])
    g_site = g_h
    if knob_tab:
        import mujoco
        p0 = np.asarray(decl["point"], float)
        # 候选 = 按钮体各 geom OBB **顶面**的真实角点（矩形面上离转轴
        # 最远点必为角点），按力臂降序送可达性预审，取第一个 viable——
        # 最大力臂角点可能深伸不可达（goal:7 西北角 OSC 饱和实证），
        # 力臂稍逊但可达的角点更优。AABB 角点不可用：旋转盒的 AABB
        # 角点在物体外（goal:7 实证：pad 悬在板缘外 5mm 虚空，永不
        # 接触）。全部不可达则退回最大力臂（接近段干净失败，交调用
        # 方换腿/换 attempt）。
        cands: List[Tuple[float, float, float, float, int]] = []
        seen = set()
        for g in _subtree_geoms(adapter, decl["bid"]):
            tg = _geom_obb_top(adapter, g)
            if not np.isfinite(tg):
                continue
            R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
            sz = np.asarray(m.geom_size[g], float)
            c = np.asarray(d.geom_xpos[g], float)
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    for szz in (-1.0, 1.0):
                        p = c + R @ (np.array([sx, sy, szz]) * sz)
                        if p[2] < tg - 1e-6:
                            continue        # 只要顶面角点
                        key = (round(float(p[0]), 3),
                               round(float(p[1]), 3))
                        if key in seen:
                            continue
                        seen.add(key)
                        cands.append((float(np.hypot(p[0] - p0[0],
                                                     p[1] - p0[1])),
                                      float(p[0]), float(p[1]), tg, g))
        cands.sort(key=lambda t: -t[0])
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, GRIP_SITE)
        rot_now = np.asarray(d.site_xmat[sid], float).reshape(3, 3)
        dbg_knob = os.environ.get("LIBERO_DEBUG")
        for _r, x, y, tg, g in cands:
            s = np.array([x, y,
                          tg - (hand["b_depth"] + hand["tip_r"])])
            if dbg_knob:
                print(f"[dbg] knob_corner lever={_r:.3f} "
                      f"xy=({x:.3f},{y:.3f}) top={tg:.3f} g{g}", flush=True)
            if _site_ik_viable(adapter, s, rot_now, {g}, dbg=dbg_knob):
                site, g_site = s, g
                break
        else:
            if cands:
                _r, x, y, tg, g = cands[0]
                site = np.array([x, y,
                                 tg - (hand["b_depth"] + hand["tip_r"])])
                g_site = g
    if (decl["jtype"] == "prismatic" and g_h >= 0
            and abs(float(D[2])) < 0.3) and not knob_tab:
        D = D / max(float(np.linalg.norm(D)), 1e-9)
        R = np.asarray(d.geom_xmat[g_h], float).reshape(3, 3)
        e = np.abs(R) @ np.asarray(m.geom_size[g_h], float)
        zax = np.array([0.0, 0.0, 1.0])
        lat = np.cross(zax, D)
        nlat = float(np.linalg.norm(lat))
        # 捏夹轴取 geom 最薄截面方向（竖直优先，竖直不薄时取侧向），
        # 接近轴（行进反方向）不参与候选——沿接近轴捏夹下垫pad无
        # 进入空间。
        u = zax if e[2] <= float(np.abs(lat) @ e) else lat / max(nlat, 1e-9)
        a = -D
        v = np.cross(a, u)
        v /= max(float(np.linalg.norm(v)), 1e-9)
        Rs = np.column_stack([u, v, a])
        # 接触点嵌到前缘面内：pregrasp 直查（free 才返回 site 自身）
        # 必然被拒，强制沿撤退向搜出真预达点；plan_arm_path 的
        # transit 膨胀（~2cm）到不了嵌入点，此前直接规划接触点
        # 返回 None（goal:0 实证）。接近段在球触面时触发接触事件。
        site = h0 + D * max(float(e @ D) - 0.5 * hand["tip_r"], 0.0)
    return site, Rs, knob_tab, g_site


def articulate(adapter, target: str, direction: str,
               strategy: int = 0) -> Dict[str, Any]:
    """开/关关节体：目标 qpos 由 adapter.articulation_info 现解，
    抓住作用点沿关节流形跟随。到位判据是 q 残差进入跟踪噪声
    （模型真值统计），不是固定步数/力阈值。"""
    try:
        info = adapter.articulation_info(target)
    except KeyError as e:
        return _fail(str(e), "perception_fail", target=target)
    q_goal = info.get("open_qpos") if direction == "open" \
        else info.get("close_qpos")
    lo, hi = info.get("range", [None, None])
    if q_goal is None:
        q_goal = hi if direction == "open" else lo
    if q_goal is None:
        return _fail(f"{target} 关节无目标 qpos", "perception_fail")

    decl = _joint_decl(adapter, info["joint"])
    q0 = _read_q(adapter, decl)
    if abs(float(q_goal) - q0) < IK_TOL:
        return _ok(target=target, note="已在目标态", q=q0)
    decl["open_dir"] = 1.0 if q_goal > q0 else -1.0
    h0, g_h = _handle_geom(adapter, decl)
    hand = measure_hand(adapter)

    site, Rs, knob, g_site = _articulate_site(adapter, decl, h0, g_h, hand)
    fgeoms = _fixture_geoms(adapter, decl)
    zone = pre = None
    if knob and g_site >= 0:
        zone = _geom_world_aabb(adapter, g_site, 2.0 * hand["tip_r"])
        pre = _knob_halfgap_for(adapter, g_site, Rs, hand)
    r = _reach_site(adapter, site, Rs, tgt_geoms=fgeoms, zone=zone,
                    preshape=pre)
    if r is not None:
        return r
    _move_gripper(adapter, +1.0)
    r = _follow_manifold(adapter, decl, h0, q0, float(q_goal),
                         target, 500)
    if not r.get("success") or direction not in ("open", "close"):
        return r
    # 谓词驱动续程：跟随的到位判据是 q 残差进噪声 3σ 包络，停点可能
    # 未过谓词阈值——木柜开区间宽仅 0.02、目标裕量 0.016 可被 3σ 停
    # 点吃掉（goal:0 实证：停在阈值上方谓词永假，8 attempt 同路径
    # 死循环）。未过则沿同方向推进到关节行程极端，直至谓词满足或
    # 卡死。语义阈值由夹具自身 articulation 属性给出，非任务参数。
    try:
        done = adapter.fixture_open(target) if direction == "open" \
            else adapter.fixture_close(target)
    except Exception:
        done = True
    _legs = 0
    while not done and _legs < 4:
        _legs += 1
        q_now = _read_q(adapter, decl)
        fwd = float(decl.get("open_dir", 1.0))
        q_ext = hi if fwd > 0 else lo
        if q_ext is None or fwd * (float(q_ext) - q_now) <= IK_TOL:
            break
        # 每段续程重新抓把手：跟随完成后手已开爪上抬，作用点随
        # 门体移动过，必须重接近再跟随。
        h1, g1 = _handle_geom(adapter, decl)
        site1, Rs1, knob1, gs1 = _articulate_site(adapter, decl, h1, g1, hand)
        zone1 = pre1 = None
        if knob1 and gs1 >= 0:
            zone1 = _geom_world_aabb(adapter, gs1, 2.0 * hand["tip_r"])
            pre1 = _knob_halfgap_for(adapter, gs1, Rs1, hand)
        rr = _reach_site(adapter, site1, Rs1, tgt_geoms=fgeoms, zone=zone1,
                         preshape=pre1)
        if rr is not None:
            break
        _move_gripper(adapter, +1.0)
        r2 = _follow_manifold(adapter, decl, h1, q_now,
                              float(q_ext), target, 500)
        if not r2.get("success"):
            break
        try:
            done = adapter.fixture_open(target) if direction == "open" \
                else adapter.fixture_close(target)
        except Exception:
            done = True
    return r


def toggle(adapter, target: str, direction: str,
           strategy: int = 0) -> Dict[str, Any]:
    """旋钮 turnon/turnoff：与 articulate 同一套作用点求解（竖直转轴
    薄耳片 → 切向捏夹 + 径向接近，见 _articulate_site）+ 流形跟随。

    102° 级旋转手掌扫掠躲不开灶面外壳/邻居旋钮（goal:7 实证），且
    旋钮关节无弹簧回位（拨动后 200 步保持不动，probe 实测）——跟随
    被阻挡后松爪撤退、按当前耳片位姿重接近重抓续转，分段到位。每段
    的旋转量由接触阻挡现决，不是固定角度；无进展（<5% 行程，算法
    精度下限，同 _follow_manifold 的 1% 约定量级）才判真失败。"""
    try:
        info = adapter.articulation_info(target)
    except KeyError as e:
        return _fail(str(e), "perception_fail", target=target)
    q_goal = info.get("turnon_qpos") if direction == "turnon" \
        else info.get("turnoff_qpos")
    if q_goal is None:
        lo, hi = info.get("range", [0.0, 0.0])
        q_goal = hi if direction == "turnon" else lo
    decl = _joint_decl(adapter, info["joint"])
    q0 = _read_q(adapter, decl)
    if abs(float(q_goal) - q0) < IK_TOL:
        return _ok(target=target, note="已在目标态", q=q0)
    decl["open_dir"] = 1.0 if q_goal > q0 else -1.0
    hand = measure_hand(adapter)
    fgeoms = _fixture_geoms(adapter, decl)
    travel = abs(float(q_goal) - q0)
    r: Optional[Dict[str, Any]] = None
    for leg in range(6):
        q_now = _read_q(adapter, decl)
        h0, g_h = _handle_geom(adapter, decl)
        site, Rs, knob, g_site = _articulate_site(adapter, decl, h0, g_h, hand)
        zone = pre = None
        if knob and g_site >= 0:
            zone = _geom_world_aabb(adapter, g_site, 2.0 * hand["tip_r"])
            pre = _knob_halfgap_for(adapter, g_site, Rs, hand)
        rr = _reach_site(adapter, site, Rs, tgt_geoms=fgeoms, zone=zone,
                         preshape=pre)
        if rr is not None:
            # 首段失败 = 真的到不了；续程失败 = 用已转的进度收尾
            if leg == 0:
                return rr
            break
        _move_gripper(adapter, +1.0)
        r = _follow_manifold(adapter, decl, h0, q_now, float(q_goal),
                             target, 400, track_rot=bool(knob))
        if r.get("success"):
            return r
        q_end = _read_q(adapter, decl)
        if abs(q_end - q_now) < 0.05 * travel:
            return r            # 这段根本没转动，再试同理
        if abs(float(q_goal) - q_end) < 0.05 * travel:
            break               # 已到目标邻域，收尾复核
        # 松爪撤退，按耳片新位姿重抓续转（关节摩擦保持位姿）
        _move_gripper(adapter, -1.0)
        cur = _eef(adapter)
        goto(adapter, [cur[0], cur[1], cur[2] + 0.08], gripper=-1.0)
    if r is not None:
        q_fin = _read_q(adapter, decl)
        if abs(float(q_goal) - q_fin) < 0.05 * travel:
            return _ok(target=target, q=round(q_fin, 4),
                       q_goal=round(float(q_goal), 4),
                       note="分段续转到位")
        return r
    return _fail("旋钮无可达作用点", "ik_unreachable", target=target)


# ==================== 兜底 ====================

def move_to(adapter, point, gripper: float = 0.0) -> Dict[str, Any]:
    """自由移动到指定坐标（走廊现解；无标准模板时的兜底）。"""
    hand = measure_hand(adapter)
    legs = corridor_plan(adapter, np.asarray(point, float), hand)
    if legs is None:
        return _fail(f"目标点无可达走廊: {point}", "ik_unreachable")
    r = execute_corridor(adapter, legs, hand, gripper=gripper)
    return r if r is not None else _ok(target=list(point))
