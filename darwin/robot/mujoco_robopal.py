"""MujocoRobopalBackend:mujoco env + robopal 机器人 + mplib 规划器后端。

把原 motion.py 的 6 个机型耦合点(URDF 路径/joint 前缀/eef 偏移/qpos 直写/base_pose/convex hull)
从 Skill 内部搬到 Backend,改为读 self.profile。Skill 只调接口,换机型只换 Backend+Profile。

mplib 限制:mplib.Planner(move_group=link7) 只规划单链。双臂=每臂一个 planner(缓存 (manipulator,actor)),
顺序规划(左臂动右臂保持),跨臂碰撞靠全 URDF + SRDF 不禁用跨臂对自动检测。
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .backend import RobotBackend, register_backend


# mplib planner 全局缓存:{(manipulator, actor): mplib.Planner}
# 跨 env 复用(mplib planner 与 mujoco env 解耦,靠 set_base_pose + set_qpos 同步)
_PLANNER_CACHE: Dict[Tuple[str, str], Any] = {}


class MujocoRobopalBackend(RobotBackend):
    """mujoco env + robopal 机器人 + mplib 规划器(原 motion.py 逻辑搬迁,配置化)。"""

    # ------------------------------------------------------------------
    # 内部:actor 配置解析
    # ------------------------------------------------------------------
    def _arm_cfg(self, actor: str):
        arm = self.profile.actors.get(actor)
        if arm is None:  # 兜底:用第一个 actor
            arm = next(iter(self.profile.actors.values()))
        return arm

    def _robopal_assets_dir(self) -> Path:
        import robopal
        return Path(robopal.__file__).parent / "assets"

    def _resolve_path(self, p: str) -> Path:
        """profile 里的相对路径按 robopal assets 解析;绝对路径原样返回。"""
        pp = Path(p)
        if pp.is_absolute():
            return pp
        return self._robopal_assets_dir() / pp

    # ------------------------------------------------------------------
    # 状态读取(原 _get_arm_qpos / _get_eef_to_link7_offset / base_pose)
    # ------------------------------------------------------------------
    def get_arm_qpos(self, actor: str) -> np.ndarray:
        return np.array(self.env.robot.get_arm_qpos(actor), float).reshape(-1)[:7]

    def _get_arm_qposadr(self, actor: str) -> np.ndarray:
        import mujoco
        model = self.env.mj_model
        adr = []
        for jn in self.env.robot.arm_joint_names[actor]:  # mujoco 名(带前缀)
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            adr.append(model.jnt_qposadr[jid])
        return np.array(adr, int)

    def get_eef_pose(self, actor: str) -> Tuple[np.ndarray, np.ndarray]:
        """末端 world 位姿 [pos(3), quat(4) [w,x,y,z]](body 真值)。"""
        import mujoco
        import robopal.commons.transform as T
        arm = self._arm_cfg(actor)
        try:
            bid = mujoco.mj_name2id(self.env.mj_model, mujoco.mjtObj.mjOBJ_BODY, arm.mujoco_eef_body)
            pos = np.array(self.env.mj_data.body(bid).xpos, float).reshape(3)
            mat = np.array(self.env.mj_data.body(bid).xmat, float).reshape(3, 3)
            return pos, T.mat_2_quat(mat)
        except Exception:
            pos, quat = self.env.controller.forward_kinematics(
                self.env.robot.get_arm_qpos(actor), actor)
            return np.array(pos, float), np.array(quat, float)

    def get_eef_to_move_group_offset(self, actor: str) -> np.ndarray:
        """eef→move_group_link 位置偏移(world):move_group_pos - eef_pos。

        mplib move_group=link7,goal 是 link7 目标。上层传的 goal 通常是 eef(夹爪末端)目标,
        需转:link7_goal = eef_goal + (link7_pos - eef_pos)。姿态不转(竖直夹爪场景够用)。
        """
        import mujoco
        arm = self._arm_cfg(actor)
        # move_group 在 mujoco 里的 body 名 = 前缀 + URDF link 名(如 0_link7)
        mv_link = (arm.mujoco_joint_prefix + arm.move_group_link
                   if arm.mujoco_joint_prefix else arm.move_group_link)
        try:
            eef_bid = mujoco.mj_name2id(self.env.mj_model, mujoco.mjtObj.mjOBJ_BODY, arm.mujoco_eef_body)
            mv_bid = mujoco.mj_name2id(self.env.mj_model, mujoco.mjtObj.mjOBJ_BODY, mv_link)
            if eef_bid < 0 or mv_bid < 0:
                return np.zeros(3)
            eef_pos = np.array(self.env.mj_data.body(eef_bid).xpos, float).reshape(3)
            mv_pos = np.array(self.env.mj_data.body(mv_bid).xpos, float).reshape(3)
            return mv_pos - eef_pos
        except Exception:
            return np.zeros(3)

    def get_base_pose(self, actor: str) -> np.ndarray:
        """base body 在 world 的位姿 [pos(3), quat(4)](供 mplib set_base_pose)。"""
        from transforms3d.quaternions import mat2quat
        arm = self._arm_cfg(actor)
        if not arm.mujoco_base_body:
            return np.concatenate([np.zeros(3), [1.0, 0.0, 0.0, 0.0]])  # 原点单位姿态
        import mujoco
        try:
            model, data = self.env.mj_model, self.env.mj_data
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, arm.mujoco_base_body)
            pos = np.array(data.body(bid).xpos, float).reshape(3)
            mat = np.array(data.body(bid).xmat, float).reshape(3, 3)
            return np.concatenate([pos, mat2quat(mat)])  # [w,x,y,z]
        except Exception:
            return np.concatenate([np.zeros(3), [1.0, 0.0, 0.0, 0.0]])

    # ------------------------------------------------------------------
    # 准静态执行(原 _set_arm_qpos_quasi_static)
    # ------------------------------------------------------------------
    def _set_arm_qpos_quasi_static(self, q: np.ndarray, actor: str) -> None:
        """设臂关节角 + mj_forward(保留夹爪当前 qpos)。

        mj_forward 跳过动力学:夹爪保持当前值(夹持状态保持),夹持物体几何跟随,
        无惯性/无力觉(path_plan 远距离迁移不需力觉)。手动增 _timestep(耗 episode 步数)。
        """
        import mujoco
        model, data = self.env.mj_model, self.env.mj_data
        adr = self._get_arm_qposadr(actor)
        data.qpos[adr] = np.asarray(q, float).reshape(-1)[:7]
        mujoco.mj_forward(model, data)
        if hasattr(self.env, "_timestep"):
            self.env._timestep += 1

    def execute_trajectory(self, traj, actor: str, skip: int = 2,
                          timeout: int = 200) -> Dict[str, Any]:
        traj = np.asarray(traj, float)
        if traj.ndim == 1:
            traj = traj.reshape(1, -1)
        traj = traj[:, :7]
        executed, n_steps = 0, 0
        last_idx = -1
        for i in range(0, len(traj), skip):
            if n_steps >= timeout:
                return {"executed": executed, "reason": "exec_timeout",
                        "final_qpos": self.get_arm_qpos(actor).tolist()}
            self._set_arm_qpos_quasi_static(traj[i], actor)
            executed += 1
            n_steps += 1
            last_idx = i
        # 保证执行末位 waypoint(目标构型):skip 降采样可能跳过最后一帧,需补执行
        if len(traj) > 0 and last_idx != len(traj) - 1 and n_steps < timeout:
            self._set_arm_qpos_quasi_static(traj[-1], actor)
            executed += 1
        return {"executed": executed, "reason": "ok",
                "final_qpos": self.get_arm_qpos(actor).tolist()}

    # ------------------------------------------------------------------
    # URDF/SRDF/mesh/convex 准备(原 _make_fixed_urdf_srdf,配置化)
    # ------------------------------------------------------------------
    def prepare_collision_assets(self) -> None:
        """准备 mplib 可用的 URDF + SRDF + mesh symlink + convex hull(幂等缓存)。

        缓存到 darwin/assets/mplib_cache/。读 self.profile 的 urdf_path/mesh_root/
        urdf_mesh_rewrite/adjacent_link_pairs/extra_disabled_pairs。
        """
        cache_dir = Path(__file__).resolve().parent.parent / "assets" / "mplib_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        fixed_urdf = cache_dir / f"{self.profile.manipulator}_fixed.urdf"
        fixed_srdf = cache_dir / f"{self.profile.manipulator}_fixed.srdf"
        mesh_link_dir = cache_dir / "meshes"

        if not fixed_urdf.exists() and self.profile.urdf_path:
            src_urdf = self._resolve_path(self.profile.urdf_path)
            if not src_urdf.exists():
                raise FileNotFoundError(f"profile urdf_path 不存在: {src_urdf}")
            text = src_urdf.read_text()
            for old, new in self.profile.urdf_mesh_rewrite.items():
                text = text.replace(old, new)
            fixed_urdf.write_text(text)

        # symlink mesh → cache/meshes/(首次建,STL 不变则复用)
        if not mesh_link_dir.exists() and self.profile.mesh_root:
            mesh_src = self._resolve_path(self.profile.mesh_root)
            if mesh_src.exists():
                mesh_link_dir.mkdir(parents=True, exist_ok=True)
                for stl in mesh_src.glob("*.STL"):
                    link = mesh_link_dir / stl.name
                    if not link.exists():
                        try:
                            link.symlink_to(stl)
                        except OSError:
                            shutil.copy(stl, link)

        # 预生成 convex hull(mplib 加载 convex=True 期望 <mesh>.convex.stl 已存在)
        try:
            import trimesh
            for stl in mesh_link_dir.glob("*.STL"):
                convex = stl.with_name(f"{stl.name}.convex.stl")
                if not convex.exists():
                    m = trimesh.load(str(stl), force="mesh")
                    if m.is_empty:
                        continue
                    m.convex_hull.export(str(convex))
        except Exception:
            pass  # convex 生成失败不阻断

        if not fixed_srdf.exists():
            pairs = self.profile.adjacent_link_pairs or []
            extra = self.profile.extra_disabled_pairs or []
            lines = [f'  <disable_collisions link1="{a}" link2="{b}" reason="adjacent"/>'
                     for a, b in pairs]
            lines += [f'  <disable_collisions link1="{a}" link2="{b}" reason="working_range"/>'
                      for a, b in extra]
            body = "\n".join(lines)
            fixed_srdf.write_text(
                f'<?xml version="1.0"?>\n'
                f'<robot name="{self.profile.manipulator}">\n'
                f'{body}\n'
                f'</robot>\n'
            )

    # ------------------------------------------------------------------
    # mplib planner(原 _get_planner / _set_base_pose_from_mujoco / _sync_mujoco_to_mplib)
    # ------------------------------------------------------------------
    def _get_planner(self, actor: str) -> Any:
        """懒加载 mplib.Planner(首次 ~秒级,之后缓存复用)。缓存 key=(manipulator, actor)。"""
        import mplib
        arm = self._arm_cfg(actor)
        key = (self.profile.manipulator, actor)
        if key in _PLANNER_CACHE:
            return _PLANNER_CACHE[key]
        self.prepare_collision_assets()
        cache_dir = Path(__file__).resolve().parent.parent / "assets" / "mplib_cache"
        urdf = str(cache_dir / f"{self.profile.manipulator}_fixed.urdf")
        srdf = str(cache_dir / f"{self.profile.manipulator}_fixed.srdf")
        planner = mplib.Planner(
            urdf=urdf,
            srdf=srdf,
            move_group=arm.move_group_link,  # URDF 原生 link 名
            user_joint_names=arm.joint_names,  # URDF 原生 joint 名
        )
        _PLANNER_CACHE[key] = planner
        return planner

    def _set_base_pose_to_planner(self, planner, actor: str) -> None:
        """从 mujoco 读 base 位姿设给 mplib(倒挂 mount 修正)。"""
        planner.set_base_pose(self.get_base_pose(actor))

    def _sync_to_planner(self, planner, actor: str) -> np.ndarray:
        """同步 mujoco 当前状态到 mplib:base_pose + current_qpos。返回 current_qpos。"""
        self._set_base_pose_to_planner(planner, actor)
        qpos = self.get_arm_qpos(actor)
        planner.robot.set_qpos(qpos, True)
        return qpos

    # ------------------------------------------------------------------
    # 高层接口:plan_path / check_*_collision
    # ------------------------------------------------------------------
    def plan_path(self, goal_pose, actor: str, frame: str = "world",
                  current_qpos: Optional[np.ndarray] = None, **opts) -> Dict[str, Any]:
        """mplib RRT-Connect 关节空间规划。返回 {ok, status, trajectory, duration, reason}。

        goal_pose:eef 目标 [x,y,z,qw,qx,qy,qz](world/base)。内部转成 move_group_link 目标。
        current_qpos:忽略(env 是真值源,内部 _sync_to_planner 同步);留作 ABC 对称。
        """
        try:
            planner = self._get_planner(actor)
        except Exception as e:
            return {"ok": False, "reason": "planner_unavailable", "error": str(e),
                    "trajectory": [], "duration": 0.0, "status": ""}
        cur = self._sync_to_planner(planner, actor)
        goal = np.asarray(goal_pose, float).reshape(-1)[:7]
        # eef goal → move_group_link goal(姿态不转:竖直夹爪 eef/move_group 姿态一致)
        eef_offset = self.get_eef_to_move_group_offset(actor)
        goal_mv = goal.copy()
        goal_mv[:3] = goal[:3] + eef_offset
        wrt_world = (frame == "world")
        try:
            result = planner.plan_qpos_to_pose(
                goal_pose=goal_mv, current_qpos=cur,
                time_step=opts.get("time_step", 0.1),
                rrt_range=opts.get("rrt_range", 0.1),
                planning_time=opts.get("planning_time", 1.0),
                wrt_world=wrt_world,
                planner_name=opts.get("planner_name", "RRTConnect"),
                use_point_cloud=False, use_attach=False,
            )
        except Exception as e:
            return {"ok": False, "reason": "planner_exception", "error": str(e),
                    "trajectory": [], "duration": 0.0, "status": ""}
        status = result.get("status", "")
        if status != "Success":
            return {"ok": False, "reason": "no_path", "status": status,
                    "trajectory": [], "duration": 0.0,
                    "current_qpos": cur.tolist(), "goal_pose": goal.tolist()}
        traj = np.asarray(result["position"], float)  # (N,7) 或 (N,8)
        traj = traj[:, :7]  # 只取臂 7 关节
        return {"ok": True, "status": status, "trajectory": traj.tolist(),
                "duration": float(result.get("duration", 0.0)), "reason": "ok"}

    def check_self_collision(self, actor: Optional[str] = None,
                             qpos_arm: Optional[np.ndarray] = None) -> List[Any]:
        """mplib FCL 自碰撞检测(基于 SRDF 排除相邻对)。返回 collision 对象列表。"""
        try:
            actor = actor or next(iter(self.profile.actors))
            planner = self._get_planner(actor)
        except Exception:
            return []
        self._sync_to_planner(planner, actor)
        q = (np.asarray(qpos_arm, float).reshape(-1)[:7]
             if qpos_arm is not None else self.get_arm_qpos(actor))
        try:
            return list(planner.check_for_self_collision(qpos=q) or [])
        except Exception:
            return []

    def check_env_collision(self, actor: str, qpos_arm: Optional[np.ndarray] = None,
                            use_attach: bool = True, with_pc: bool = False) -> List[Any]:
        try:
            planner = self._get_planner(actor)
        except Exception:
            return []
        self._sync_to_planner(planner, actor)
        q = (np.asarray(qpos_arm, float).reshape(-1)[:7]
             if qpos_arm is not None else self.get_arm_qpos(actor))
        try:
            return list(planner.check_for_env_collision(
                qpos=q, with_point_cloud=with_pc, use_attach=use_attach) or [])
        except Exception:
            return []


register_backend("mujoco_robopal", MujocoRobopalBackend)
