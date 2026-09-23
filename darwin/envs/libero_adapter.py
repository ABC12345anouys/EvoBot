"""LIBERO（robosuite/BDDL）环境适配器。

把 LIBERO 的 OffScreenRenderEnv 包装成 darwin runner 需要的 robopal 风格接口
（get_body_pos / get_site_pos / mj_model / mj_data / step / reset），使
DynamicEpisodeRunner 的感知链（object_features/点云采样）、EpisodeLogger、
条件-方法闭环可以原样跑在 LIBERO 后端上，runner 主体零分支。

设计要点：
- env_id 编码 "libero:<suite>:<task_idx>"（如 libero:libero_spatial:0）。
- reset() 内部 reset + set_init_state：初始状态取**该任务 demo_0 的 states[0]**
  （而非 pruned init），保证 demo 动作回放与录制时的初始构型一致、确定性可复现。
- body 名兼容两种：mujoco body 名（akita_black_bowl_1_main，perception 用）
  与 BDDL 对象名（akita_black_bowl_1，任务语义用）。
- LIBERO 装在独立源码目录 / 同级 site-packages，import 失败时按常见路径兜底。
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

import numpy as np

# LIBERO 源码常见位置（editable install 在 libero conda env，darwin env 靠路径兜底）
_LIBERO_SRC_CANDIDATES = (
    os.environ.get("DARWIN_LIBERO_ROOT", ""),
    "/home/lifd/Public/LIBERO",
)


def _ensure_libero_path() -> None:
    try:
        import libero  # noqa: F401
        return
    except Exception:
        pass
    for p in _LIBERO_SRC_CANDIDATES:
        if p and os.path.isdir(os.path.join(p, "libero")):
            if p not in sys.path:
                sys.path.insert(0, p)
            return


def parse_env_id(env_id: str) -> Optional[Tuple[str, int]]:
    """libero:<suite>:<idx> → (suite, idx)；非 libero env_id 返回 None。"""
    if not env_id.startswith("libero:"):
        return None
    parts = env_id.split(":")
    if len(parts) != 3:
        raise ValueError(f"LIBERO env_id 格式应为 libero:<suite>:<idx>，得到 {env_id}")
    return parts[1], int(parts[2])


@lru_cache(maxsize=8)
def _task_info(suite: str, idx: int):
    """缓存 task 元信息与 demo 初始 state（hdf5 读一次）。"""
    import h5py
    from libero.libero import benchmark, get_libero_path

    bench = benchmark.get_benchmark_dict()[suite]()
    task = bench.get_task(idx)
    bddl = os.path.join(get_libero_path("bddl_files"),
                        task.problem_folder, task.bddl_file)
    demo = os.path.join(get_libero_path("datasets"), suite,
                        task.bddl_file.replace(".bddl", "_demo.hdf5"))
    state0 = None
    n_actions = 0
    if os.path.exists(demo):
        with h5py.File(demo, "r") as f:
            state0 = np.array(f["data"]["demo_0"]["states"][0])
            n_actions = len(f["data"]["demo_0"]["actions"])
    return {
        "bench": bench, "task": task, "bddl": bddl, "demo": demo,
        "state0": state0, "n_actions": n_actions,
    }


def demo_actions(suite: str, idx: int, demo_key: str = "demo_0") -> np.ndarray:
    """读取指定 demo 的动作序列（供 ReplayDemoMethod 展开 skill 链）。"""
    import h5py
    info = _task_info(suite, idx)
    with h5py.File(info["demo"], "r") as f:
        return np.array(f["data"][demo_key]["actions"])


class LiberoEnvAdapter:
    """LIBERO env → darwin robopal 风格接口适配器。"""

    def __init__(self, suite: str, task_idx: int, camera_size: int = 128):
        _ensure_libero_path()
        from libero.libero.envs import OffScreenRenderEnv

        info = _task_info(suite, task_idx)
        self.suite, self.task_idx = suite, task_idx
        self.task = info["task"]
        self._inner_env = OffScreenRenderEnv(
            bddl_file_name=info["bddl"],
            camera_heights=camera_size, camera_widths=camera_size)
        self._inner_env.seed(0)
        self.action_dim = int(self._inner_env.env.action_dim)
        # perception 链（sample_object_point_cloud/object_features）直接读 mj_model
        self.mj_model = self._inner_env.sim.model
        # 注意：robosuite binding_utils 的 sim.data 每次访问返回新的 MjData 包装，
        # 不能在此缓存（缓存会得到 time/qpos/ctrl 全部冻结的过时快照），
        # 用 property 实时取 sim.data。
        self._state0 = info["state0"]
        self._body_cache: Dict[str, str] = {}

    @property
    def mj_data(self):
        return self._inner_env.sim.data

    # ---- robopal 风格基础接口 ----

    @property
    def sim(self):
        return self._inner_env.sim

    def _resolve_body(self, name: str) -> str:
        """BDDL 对象名 → 装配根 mujoco body 名（几何遍历须从 root 子树）。"""
        if name in self._body_cache:
            return self._body_cache[name]
        model = self._inner_env.sim.model
        # 优先 objects_dict 的 root_body（cream_cheese 等装配体的权威根）
        inner = self._inner_env.env
        obj = getattr(inner, "objects_dict", {}).get(name)
        if obj is not None:
            self._body_cache[name] = obj.root_body
            return obj.root_body
        try:
            model.body_name2id(name)
            self._body_cache[name] = name
            return name
        except Exception:
            pass
        # 兜底：BDDL 名 + _main 后缀
        cand = f"{name}_main"
        try:
            model.body_name2id(cand)
            self._body_cache[name] = cand
            return cand
        except Exception:
            pass
        raise KeyError(f"LIBERO env 中找不到 body/object: {name}")

    def get_body_pos(self, name: str) -> np.ndarray:
        sim = self._inner_env.sim
        bid = sim.model.body_name2id(self._resolve_body(name))
        return np.array(sim.data.body_xpos[bid], dtype=float)

    def get_site_pos(self, name: str) -> np.ndarray:
        sim = self._inner_env.sim
        sid = sim.model.site_name2id(name)
        return np.array(sim.data.site_xpos[sid], dtype=float)

    def contact_force_on_body(self, name: str,
                              gripper_prefix: str = "gripper0") -> float:
        """夹爪与该物体装配子树之间的接触力模长之和（N），无接触=0。

        物理 profile 的可观测探针：用 MuJoCo contact 数组逐对求 mj_contactForce，
        一侧属于物体 root 子树、另一侧 body 名以 gripper_prefix 开头才计入。
        """
        import mujoco
        sim = self._inner_env.sim
        model, data = sim.model, sim.data  # wrapper：提供 name2id/id2name 与数组代理
        # mj_contactForce 只接受原生 MuJoCo struct，robosuite 包装层需解包
        native_m = getattr(model, "_model", model)
        native_d = getattr(data, "_data", data)
        root_id = model.body_name2id(self._resolve_body(name))
        obj_bodies = {root_id}
        # 沿 body_parentid 扩张整棵子树（抽屉/容器等装配体）
        grew = True
        while grew:
            grew = False
            for bid in range(model.nbody):
                if bid not in obj_bodies:
                    par = int(model.body_parentid[bid])
                    if par in obj_bodies:
                        obj_bodies.add(bid)
                        grew = True
        def _body_name(bid: int) -> str:
            try:
                return str(model.body_id2name(bid) or "")
            except Exception:
                try:
                    return str(model.body_names[bid])
                except Exception:
                    return ""

        res = np.zeros(6, dtype=np.float64)
        total = 0.0
        for i in range(data.ncon):
            c = data.contact[i]
            b1 = int(model.geom_bodyid[c.geom1])
            b2 = int(model.geom_bodyid[c.geom2])
            if b1 in obj_bodies:
                other = b2
            elif b2 in obj_bodies:
                other = b1
            else:
                continue
            if not _body_name(other).startswith(gripper_prefix):
                continue
            mujoco.mj_contactForce(native_m, native_d, i, res)
            total += float(np.linalg.norm(res[:3]))
        return total

    def check_success(self) -> bool:
        """BDDL 谓词终态判定（任务成功）。"""
        inner = self._inner_env.env
        return bool(inner._check_success())

    # ---- 形式化子目标：逐谓词判定（多子目标顺序执行用）----

    def eval_subgoal(self, predicate: str, obj: str = None,
                     target: str = None) -> bool:
        """按 BDDL goal_state 中的同名谓词元组求值（与 _check_success 同口径）。

        On/In/Close/Turnon/Turnoff 显式写在 goal 段；隐式补出的 Open 不在
        goal_state 中 → KeyError，由调用方走 fixture_open 兜底。
        """
        inner = self._inner_env.env
        want = tuple(x for x in (predicate, obj, target) if x)
        wl = tuple(str(x).lower() if isinstance(x, str) else x
                   for x in want)
        for state in getattr(inner.parsed_problem, "get",
                             lambda k: [])("goal_state", []):
            stt = tuple(state)
            if stt == want or tuple(
                    str(x).lower() if isinstance(x, str) else x
                    for x in stt) == wl:
                return bool(inner._eval_predicate(state))
        raise KeyError(f"goal_state 中找不到谓词元组: {want}")

    def fixture_open(self, name: str) -> bool:
        """region/夹具名 → 是否处于打开态（隐式 Open 子目标兜底判定）。"""
        inner = self._inner_env.env
        st = inner.object_states_dict.get(name)
        if st is None:
            raise KeyError(f"object_states_dict 中找不到: {name}")
        # site 关节为空（如柜门 side）时，回退到驱动关节 + fixture 的 is_open
        site = inner.object_sites_dict.get(name)
        if site is not None and not getattr(site, "joints", None):
            info = self.articulation_info(name)
            sim = self._inner_env.sim
            adr = sim.model.get_joint_qpos_addr(info["joint"])
            qpos = float(sim.data.qpos[adr])
            fx = inner.fixtures_dict.get(info["fixture"])
            if fx is not None and hasattr(fx, "is_open"):
                return bool(fx.is_open(qpos))
        return bool(st.is_open())

    def fixture_close(self, name: str) -> bool:
        inner = self._inner_env.env
        st = inner.object_states_dict.get(name)
        if st is None:
            raise KeyError(f"object_states_dict 中找不到: {name}")
        site = inner.object_sites_dict.get(name)
        if site is not None and not getattr(site, "joints", None):
            info = self.articulation_info(name)
            sim = self._inner_env.sim
            adr = sim.model.get_joint_qpos_addr(info["joint"])
            qpos = float(sim.data.qpos[adr])
            fx = inner.fixtures_dict.get(info["fixture"])
            if fx is not None and hasattr(fx, "is_close"):
                return bool(fx.is_close(qpos))
            if fx is not None and hasattr(fx, "is_open"):
                return not bool(fx.is_open(qpos))
        if hasattr(st, "is_close"):
            return bool(st.is_close())
        return not bool(st.is_open())

    def articulation_info(self, name: str) -> Dict[str, Any]:
        """region/夹具名 → 驱动关节 + 各语义目标 qpos（articulate/toggle 技能）。

        选择顺序：fixture 对象 joints 里的名字（多抽屉按 top/middle/bottom
        token 过滤）→ 模型中以夹具名为前缀的关节（如 stove 的 button）。
        目标 qpos 取对象 articulation 默认范围中点（开/关/turnon/turnoff）。
        """
        sim = self._inner_env.sim
        model, data = sim.model, sim.data
        inner = self._inner_env.env
        st = inner.object_states_dict.get(name)
        fixture_name = getattr(st, "parent_name", None) or name
        fx = inner.fixtures_dict.get(fixture_name)
        props = (fx.object_properties.get("articulation", {})
                 if fx is not None else {})
        tokens = [t for t in ("top", "middle", "bottom") if t in name]

        cands = [j for j in (getattr(fx, "joints", []) or [])
                 if isinstance(j, str) and j]
        matched = [j for j in cands
                   if not tokens or any(t in j for t in tokens)]
        if not matched:
            pre = f"{fixture_name}_"
            allj = [model.joint_id2name(i) for i in range(model.njnt)
                    if str(model.joint_id2name(i) or "").startswith(pre)]
            matched = [j for j in allj
                       if not tokens or any(t in j for t in tokens)] or allj
        if not matched:
            raise KeyError(f"找不到 {name} 的驱动关节（fixture={fixture_name}）")
        joint = matched[0]
        jid = int(model.joint_name2id(joint))
        adr = int(model.jnt_qposadr[jid])
        qpos = float(data.qpos[adr])
        rng = [float(model.jnt_range[jid][0]), float(model.jnt_range[jid][1])]

        def _conservative(key: str, toward: str):
            """靠近范围极值端（toward='min'|'max'），留漂移余量。"""
            r = props.get(key)
            if r and len(r) == 2:
                a, b = float(r[0]), float(r[1])
                # 80% 靠向极值端
                return float(a + 0.8 * (b - a)) if toward == "max" else float(a + 0.2 * (b - a))
            return None

        return {
            "fixture": fixture_name,
            "joint": joint,
            "qpos": qpos,
            "range": rng,
            "open_qpos": _conservative("default_open_ranges", "min"),
            "close_qpos": _conservative("default_close_ranges", "max"),
            "turnon_qpos": _conservative("default_turnon_ranges", "max"),
            "turnoff_qpos": _conservative("default_turnoff_ranges", "min"),
        }

    def fixture_joint_qpos(self, name: str) -> Dict[str, Any]:
        """region/夹具名 → 其驱动关节名与当前 qpos（articulate 技能绑定用）。"""
        inner = self._inner_env.env
        obj = (inner.fixtures_dict.get(name)
               or inner.objects_dict.get(name))
        joints = []
        if obj is not None:
            for j in getattr(obj, "joints", []) or []:
                jn = (getattr(j, "joint_name", None)
                      or getattr(j, "name", None) or str(j))
                try:
                    qp = float(self._inner_env.sim.data.get_joint_qpos(jn))
                except Exception:
                    qp = 0.0
                joints.append({"joint": jn, "qpos": qp})
        return {"name": name, "joints": joints}

    # ---- 任务物体几何（robosuite 装配体：真实 geom 分布在 root 子树）----

    @property
    def object_names(self) -> list:
        """BDDL 声明的全部可操作对象名。"""
        return list(getattr(self._inner_env.env, "objects_dict", {}).keys())

    def object_bounds(self, name: str) -> Dict[str, Any]:
        """收集物体 root body 子树全部 geom 的世界包围盒。

        robosuite 部分物体（cream_cheese 等）root body 在世界原点、
        碰撞体在子 body 上，default_site 也在原点，因此不能只看 root；
        必须按 body_rootid 遍历整个装配子树。
        box/sphere/cylinder 的 geom_size 是可靠半尺寸；mesh 的 size 是
        AABB 缩放不可信，只用其世界位置。
        返回 center(xy), z_top, z_bottom。
        """
        sim = self._inner_env.sim
        model, data = sim.model, sim.data
        rid = model.body_name2id(self._resolve_body(name))
        geoms = []
        for bid in range(model.nbody):
            if int(model.body_rootid[bid]) != int(rid):
                continue
            for g in range(int(model.body_geomadr[bid]),
                           int(model.body_geomadr[bid] + model.body_geomnum[bid])):
                geoms.append(g)
        if not geoms:
            raise KeyError(f"物体 {name} 子树无 geom")
        xys = np.array([data.geom_xpos[g][:2] for g in geoms])
        zs = []
        xmin = ymin = np.inf
        xmax = ymax = -np.inf
        for g in geoms:
            p = data.geom_xpos[g]
            z = float(p[2])
            half = float(model.geom_size[g][2]) if int(model.geom_type[g]) in (2, 5, 6) else 0.0
            zs.append((z + half, z - half))
            # xy 包围（box 用半宽；球/圆柱用半径；mesh 尺寸不可信只取位置）
            gt = int(model.geom_type[g])
            hx = float(model.geom_size[g][0]) if gt in (2, 5, 6) else 0.0
            hy = float(model.geom_size[g][1]) if gt == 6 else (hx if gt in (2, 5) else 0.0)
            xmin, xmax = min(xmin, float(p[0]) - hx), max(xmax, float(p[0]) + hx)
            ymin, ymax = min(ymin, float(p[1]) - hy), max(ymax, float(p[1]) + hy)
        cxy = xys.mean(axis=0)
        return {"center": [float(cxy[0]), float(cxy[1])],
                "z_top": float(max(t for t, _ in zs)),
                "z_bottom": float(min(b for _, b in zs)),
                "half_x": float((xmax - xmin) / 2.0),
                "half_y": float((ymax - ymin) / 2.0)}

    # 容器/空腔类物体（需要"一指在内、一指在外"的偏心插指夹法）
    _CONTAINER_HINTS = ("bowl", "basket", "pot", "mug", "cup", "bucket",
                        "caddy", "container", "pan", "tray")

    def _is_container(self, name: str) -> bool:
        n = name.lower()
        return any(h in n for h in self._CONTAINER_HINTS)

    def grasp_point(self, name: str) -> np.ndarray:
        """物体抓取点（含夹持策略），返回末端 grip_site 目标 [x,y,z]。

        实心物体：居中、指尖在顶面下约 9mm（避开顶棱）。
        容器（碗/篮/杯…）：居中直降两指尖都会被沿口顶棱挡住、闭合时翻过
        棱边滑入空腔，必然夹空。改用 demo 同款**偏心插指**：夹爪中线沿
        开合轴（世界 y）偏离物体中心 0.7·半宽，使一只指尖在容器外、另一只
        越过沿口进入内腔，指尖降到沿顶下约 30mm（与沿壁中部齐平）后闭合，
        夹住沿壁内外两侧。夹爪姿态恒为初始朝下，开合轴固定沿世界 y。
        """
        b = self.object_bounds(name)
        cx, cy, z_top = b["center"][0], b["center"][1], b["z_top"]
        if self._is_container(name):
            offset = 0.70 * float(b["half_y"])
            return np.array([cx, cy - offset, z_top - 0.030 + 0.012], dtype=float)
        return np.array([cx, cy, z_top - 0.009 + 0.012], dtype=float)

    def support_point(self, name: str, clearance: float = 0.0) -> np.ndarray:
        """支撑物（plate 等）顶面中心点，供 On 放置目标使用。"""
        b = self.object_bounds(name)
        return np.array([b["center"][0], b["center"][1],
                         b["z_top"] + clearance], dtype=float)

    # ---- 生命周期 ----

    def reset(self):
        obs = self._inner_env.reset()
        if self._state0 is not None:
            obs = self._inner_env.set_init_state(self._state0)
        # 记录初始末端位姿（home 原语用；LIBERO 原生 env 不暴露 init_pos）。
        # reset 与每次 attempt 都会调用，回退快照后 home 仍指向开局安全位。
        self._darwin_done = False  # episode 终止标记（步数耗尽后置 True）
        try:
            sim = self._inner_env.sim
            sid = int(sim.model.site_name2id("gripper0_grip_site"))
            self.init_pos = {"agent0": np.asarray(
                sim.data.site_xpos[sid], float).copy()}
        except Exception:
            pass
        return obs

    def step(self, action):
        # robosuite 原生 7 维 OSC action：[dx,dy,dz,dax,day,daz,gripper]
        if getattr(self, "_darwin_done", False):
            raise RuntimeError("episode_terminated")
        # 预防性回卷：单个 attempt 可能跑超 horizon（carry_vcap 限速时
        # carry 单段就 500+ 步，加上 home/above/lift 常超 1000 步）。
        # darwin 自管 episode 长度，终态由 BDDL 核验，timestep 到点即回卷。
        if self._inner_timestep() >= self._inner_horizon():
            self._rewind_inner_timestep()
        try:
            obs, reward, done, info = self._inner_env.step(
                np.asarray(action, dtype=np.float64))
        except (ValueError, RuntimeError) as e:
            # done=True 后仍被调 step 时抛 "executing action in terminated
            # episode"；回卷后重试一次（仍失败才转 terminated）。
            if "terminated" in str(e).lower():
                self._rewind_inner_timestep()
                try:
                    obs, reward, done, info = self._inner_env.step(
                        np.asarray(action, dtype=np.float64))
                except Exception:
                    self._darwin_done = True
                    raise RuntimeError("episode_terminated") from e
            else:
                raise
        if done:
            # done=True 来自 horizon 到点（回卷继续）或 BDDL 成功（交上层核验）。
            self._rewind_inner_timestep()
            done = False
        return obs, reward, done, info

    def _inner_timestep(self) -> int:
        _o = self._inner_env
        while getattr(_o, "env", None):
            _o = _o.env
        return int(getattr(_o, "timestep", 0))

    def _inner_horizon(self) -> int:
        _o = self._inner_env
        while getattr(_o, "env", None):
            _o = _o.env
        return int(getattr(_o, "horizon", 1000))

    def _rewind_inner_timestep(self):
        """沿 wrapper 链逐层重置 robosuite 步数计数器。

        Wrapper 只有 __getattr__（读转发）没有 __setattr__：直接对 wrapper
        赋值会落在外层实例上，必须递归 .env 到 MujocoEnv 本体才真正生效。
        """
        _obj = self._inner_env
        for _ in range(8):
            try:
                _obj.timestep = 0
                _obj.done = False
            except Exception:
                pass
            _inner = getattr(_obj, "env", None)
            if _inner is None or _inner is _obj:
                break
            _obj = _inner

    # ---- 物理状态快照/恢复（失败回退续跑用）----

    def save_state(self) -> np.ndarray:
        """完整物理状态扁平向量（time/qpos/qvel/act/udd，含夹爪驱动状态）。

        与 LIBERO demo hdf5 的 states 同一格式，可直接互换恢复。
        """
        return np.array(self._inner_env.sim.get_state().flatten(), dtype=np.float64)

    def restore_state(self, token: np.ndarray) -> None:
        """恢复 save_state 的快照并前向传播（物体位姿/夹爪开闭全部还原）。"""
        self._inner_env.sim.set_state_from_flattened(np.asarray(token, dtype=np.float64))
        self._inner_env.sim.forward()
        # MjSimState 不含 robosuite OSC 控制器内部目标（goal_origin 跨步积分）；
        # 物理回退后必须把控制器目标对齐到当前末端位姿，否则头几十步控制器按
        # 旧目标发力，回退续跑的 place 会跑偏。
        for robot in getattr(self._inner_env, "robots", []) or []:
            ctrl = getattr(robot, "controller", None)
            if ctrl is not None and hasattr(ctrl, "reset_goal"):
                try:
                    ctrl.reset_goal()
                except Exception:
                    pass
        # 对齐 regenerate_obs_from_state：刷新谓词内部状态（没有该方法也无妨）
        fn = getattr(self._inner_env, "check_success", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass
        # 重置 step 计数器：sim_worker 多 attempt 共用同一 episode（restore 回
        # 快照而非 robosuite reset），robosuite 的 timestep/done 不随 MuJoCo
        # state 恢复，累积到 horizon 后 done=True → episode_terminated
        self._darwin_done = False
        self._rewind_inner_timestep()

    def servo_step(self, site, target, gripper=0.0, k=5.0, vcap=1.0,
                   actor: str = "agent0") -> None:
        """末端朝 target P 伺服一步；gripper 语义 +1=闭合/-1=张开/0=保持。

        LIBERO robosuite OSC 物理 +1=合/-1=开，与统一语义同号，直传。
        通用 skill 调此方法跨 env 共享，无需关心底层 7 维 OSC 格式。
        episode 步数耗尽后抛 EpisodeTerminated，让 skill/runner 干净退出。
        """
        if getattr(self, "_darwin_done", False):
            raise RuntimeError("episode_terminated")
        end = np.asarray(self.get_site_pos(site), float)
        vmax = min(1.0, float(vcap))
        delta = np.clip(k * (np.asarray(target, float) - end), -vmax, vmax)
        g = float(max(-1.0, min(1.0, float(gripper))))
        self.step(np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, g],
                           dtype=np.float64))

    def close(self) -> None:
        try:
            self._inner_env.close()
        except Exception:
            pass
