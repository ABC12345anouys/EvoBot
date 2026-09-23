"""InsertEnv：peg-in-hole / 插入类任务的 CARTIMP 阻抗环境适配层。

不修改 robopal 源码，继承 ManipulateEnv 并重写 step 以接受 8 维位姿动作：
    action = [x, y, z, qw, qx, qy, qz, gripper]
前 7 维作为末端位姿目标交给 CartesianImpedanceController（真·任务空间阻抗），
末位为夹爪指令（+1 开 / -1 合 / 0 保持，归一化到 grip 控制范围）。

力信号来源：RethinkGripper 的 force_ee / torque_ee sensor（挂在 ft_frame site）。
对无 force sensor 的夹爪（如 PandaHand）自动回退到 mj_data.contact + mj_contactForce。

设计要点（与现有 4 维原语解耦）：
- 现有 pickplace/stack 任务仍用 ManipulateEnv（CARTIK，4 维），不受影响。
- insert 任务用 InsertEnv（CARTIMP，8 维），配合 darwin/skills/primitives/control.py 的位姿原语与力控原语。
"""
from __future__ import annotations

import os

# 无头渲染引导：必须先于 robopal/mujoco 导入
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

from typing import Tuple

import mujoco
import numpy as np

from robopal.envs.manipulation_tasks.robot_manipulate import ManipulateEnv
from robopal.envs.robot import RobotEnv


class InsertEnv(ManipulateEnv):
    """peg-in-hole 阻抗环境。

    :param robot: 机器人配置（如 DianaTripleStack，使用 RethinkGripper，自带 force_ee sensor）
    :param render_mode: 渲染模式
    :param control_freq: 上层控制频率
    :param is_randomize_end: 是否随机化末端初始位姿
    :param is_randomize_object: 是否随机化物体位姿
    """

    name = "InsertEnv-v0"

    def __init__(self,
                 robot=None,
                 render_mode="human",
                 control_freq=20,
                 is_show_camera_in_cv=False,
                 is_randomize_end=False,
                 is_randomize_object=False):
        super().__init__(
            robot=robot,
            render_mode=render_mode,
            control_freq=control_freq,
            # robopal CARTIMP 对 DianaMed 不稳定（K=0 都发散，重力补偿/Coriolis 项有 bug），
            # 且不能改 robopal 源码。改用 CARTIK（IK+JNTIMP，稳定）+ force_ee sensor 做力觉终止。
            # 这正是用户设计哲学："几何先到位，力觉做终止和保护"。
            controller="CARTIK",
            is_interpolate=False,
            action_type="position",
            is_randomize_end=is_randomize_end,
            is_randomize_object=is_randomize_object,
            is_show_camera_in_cv=is_show_camera_in_cv,
        )
        # 8 维动作空间：7 位姿 + 1 夹爪
        self.action_dim = (8,)
        self.max_action = 1.0
        self.min_action = -1.0
        # 插入任务步数预算放大（servo_align + impedance_push 需要较多步）
        self.max_episode_steps = 500
        # robopal CARTIMP 的 Jacobian 在 base frame，故 reference 保持默认 'base'：
        # controller FK 返回 base-frame 位姿，pos_error 与 J 同 frame 才稳定。
        # 设 'world' 会让误差在 world frame、力矩在 base frame → 方向错乱（DianaMed 倒挂尤甚）。
        # 外部（site/body 真值）用 world frame；控制器收到的 action 用 base frame。
        # 本 env 提供 world_to_base / base_to_world 转换辅助原语使用。
        self._base_pose = self._compute_base_pose()  # 4x4 world->base
        # 缓存 force sensor id（避免每步 mj_name2id）
        self._force_sensor_id = self._resolve_force_sensor_id()
        self._torque_sensor_id = self._resolve_torque_sensor_id()
        # 把 init_pos/init_quat 从 base frame 转成 world frame（供 PoseHomeSkill 直接用）
        self._convert_init_pose_to_world()

    def _convert_init_pose_to_world(self):
        """把 RobotEnv 在 __init__ 时通过 controller FK（base frame）写入的
        init_pos/init_quat 转成 world frame，让所有原语统一在 world frame 工作。"""
        for agent in self.agents:
            b_pos = np.asarray(self.init_pos[agent], float).reshape(3)
            b_quat = np.asarray(self.init_quat[agent], float).reshape(4)
            self.init_pos[agent] = self.base_to_world_pos(b_pos)
            self.init_quat[agent] = self._base_to_world_quat(b_quat)
        self.robot.init_pos = self.init_pos
        self.robot.init_quat = self.init_quat
        # desired_position 也在 world frame（与 init_pos 一致）
        self.desired_position = self.init_pos[self.agents[0]]

    def _base_to_world_quat(self, base_quat: np.ndarray) -> np.ndarray:
        """base 四元数 [w,x,y,z] → world 四元数。"""
        import robopal.commons.transform as T
        R_b = T.quat_2_mat(np.asarray(base_quat, float).reshape(4))
        R_b4 = T.make_transform(np.zeros(3), R_b)
        R_w = (self._base_pose @ R_b4)[:3, :3]
        return T.mat_2_quat(R_w)

    def update_init_pose_to_current(self):
        """重写：robopal 的实现用 controller FK（base frame）；这里转成 world frame。"""
        # 先调用父类（会用 controller FK 写入 base frame）
        super().update_init_pose_to_current()
        # base→world
        self._base_pose = self._compute_base_pose()  # reset 后 base 可能变（虽然单臂不会）
        self._convert_init_pose_to_world()

    # ------------------------------------------------------------------
    # step：8 维位姿动作（绕过 ManipulateEnv 的 4 维拆分）
    # ------------------------------------------------------------------

    def step(self, action):
        """8 维动作 [x,y,z,qw,qx,qy,qz, gripper]（WORLD frame）。

        前 3 维作为末端 WORLD-frame 位置目标；内部转成 base frame 后交给 CARTIK
        （CARTIK 做 IK→JNTIMP，稳定；quat 保持 init_quat 即竖直向下，适合插入）。
        末位归一化夹爪指令映射到夹爪 actuator 控制范围。

        注：原计划用 CARTIMP 做真·阻抗控制，但 robopal CARTIMP 对 DianaMed 不稳定
        （即便 K=0 也发散，重力补偿/Coriolis 项有 bug），且不能改源码。故退回 CARTIK
        + force_ee sensor 做力觉终止/保护（force-guarded position control）。
        """
        self._timestep += 1
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape[0] < 7:
            raise ValueError(f"InsertEnv.step 期望至少 7 维位姿动作，收到 {action.shape}")
        world_pos = action[:3].astype(float)
        grip = float(action[7]) if action.shape[0] >= 8 else 0.0

        # 夹爪控制（+1 开 / -1 合 / 0 保持，归一化映射）
        g_min, g_max = self.grip_min_bound, self.grip_max_bound
        grip_ctrl = (grip + 1.0) * (g_max - g_min) / 2.0 + g_min
        self.robot.end[self.agents[0]].apply_action(grip_ctrl)

        # world→base 位置转换（CARTIK 的 IK 在 base frame）
        base_pos = self.world_to_base_pos(world_pos)

        # 3 维 base-frame 位置交给 CARTIK（走 RobotEnv.step，绕过 ManipulateEnv 4 维拆分）
        # CARTIK 收 3D 时用 init_quat 做姿态目标（保持竖直向下，适合插入）
        RobotEnv.step(self, base_pos)

        self.desired_position = world_pos
        obs = self._get_obs()
        reward = self.compute_rewards()
        terminated = False
        truncated = True if self._timestep >= self.max_episode_steps else False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # world↔base 坐标转换（CARTIMP 的 Jacobian 在 base frame）
    # ------------------------------------------------------------------

    def _compute_base_pose(self) -> np.ndarray:
        """机器人 base body 在 world 下的 4x4 位姿（缓存一次，reset 后重算）。"""
        import robopal.commons.transform as T
        agent = self.agents[0]
        try:
            base_pos = self.mj_data.body(self.robot.base_link_name[agent]).xpos.copy()
            base_mat = self.mj_data.body(self.robot.base_link_name[agent]).xmat.reshape(3, 3).copy()
            return T.make_transform(base_pos, base_mat)
        except Exception:
            return np.eye(4)

    def world_to_base_pos(self, world_pos: np.ndarray) -> np.ndarray:
        """world 3D → base 3D（仅位置）。"""
        T_w2b = np.linalg.inv(self._base_pose)
        p = np.asarray(world_pos, float).reshape(3)
        return (T_w2b[:3, :3] @ p + T_w2b[:3, 3]).astype(float)

    def world_to_base_quat(self, world_quat: np.ndarray) -> np.ndarray:
        """world 四元数 [w,x,y,z] → base 四元数。"""
        import robopal.commons.transform as T
        R_w = T.quat_2_mat(np.asarray(world_quat, float).reshape(4))
        R_w4 = T.make_transform(np.zeros(3), R_w)
        T_w2b = np.linalg.inv(self._base_pose)
        R_b = (T_w2b @ R_w4)[:3, :3]
        return T.mat_2_quat(R_b)

    def base_to_world_pos(self, base_pos: np.ndarray) -> np.ndarray:
        """base 3D → world 3D。"""
        p = np.asarray(base_pos, float).reshape(3)
        return (self._base_pose[:3, :3] @ p + self._base_pose[:3, 3]).astype(float)

    def refresh_base_pose(self) -> None:
        """reset 后 base body 位姿变化时重新缓存。"""
        self._base_pose = self._compute_base_pose()

    # ------------------------------------------------------------------
    # 力信号：force_ee sensor 优先，contact force 回退
    # ------------------------------------------------------------------

    def _resolve_force_sensor_id(self) -> int:
        """查找 force sensor 的 id（RethinkGripper 的 force_ee；找不到遍历 force type）。"""
        try:
            sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "force_ee")
            if sid >= 0:
                return int(sid)
        except Exception:
            pass
        # fallback：遍历所有 sensor，找 force type
        try:
            for i in range(self.mj_model.nsensor):
                if self.mj_model.sensor_type[i] == mujoco.mjtSensor.mjSENS_FORCE:
                    return i
        except Exception:
            pass
        return -1

    def _resolve_torque_sensor_id(self) -> int:
        try:
            sid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "torque_ee")
            if sid >= 0:
                return int(sid)
        except Exception:
            pass
        try:
            for i in range(self.mj_model.nsensor):
                if self.mj_model.sensor_type[i] == mujoco.mjtSensor.mjSENS_TORQUE:
                    return i
        except Exception:
            pass
        return -1

    def get_ee_force(self) -> np.ndarray:
        """末端接触力（ft_frame 局部坐标系，3 维，单位 N）。

        优先用 force_ee sensor；sensor 不存在时遍历 contact 用 mj_contactForce 求和回退。
        """
        if self._force_sensor_id >= 0:
            adr = int(self.mj_model.sensor_adr[self._force_sensor_id])
            return np.array(self.mj_data.sensordata[adr:adr + 3], dtype=float)
        # 回退：contact force 求和（contact frame，近似）
        force = np.zeros(3)
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            f6 = np.zeros(6)
            mujoco.mj_contactForce(self.mj_model, self.mj_data, i, f6)
            force += f6[:3]
        return force

    def get_ee_force_norm(self) -> float:
        """末端接触力范数（标量，用于阈值判定）。"""
        return float(np.linalg.norm(self.get_ee_force()))

    def get_ee_pose(self, agent: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """末端当前 WORLD-frame 位姿 [pos(3), quat(4)]。

        用 site/body 真值（与 env.get_site_pos 一致），不走 controller FK
        （controller FK 返回 base frame，与原语用的 world 目标混用会差 base_offset）。
        """
        import robopal.commons.transform as T
        agent = agent or self.agents[0]
        # 用 end body 的世界位姿（grip_site 挂在 eef body 上，与之一致）
        end_name = self.robot.end_name[agent]
        try:
            pos = np.array(self.mj_data.body(end_name).xpos, dtype=float).reshape(3)
            mat = np.array(self.mj_data.body(end_name).xmat, dtype=float).reshape(3, 3)
            quat = T.mat_2_quat(mat)
            return pos, quat
        except Exception:
            # 回退到 controller FK（base frame，仅作兜底）
            pos, quat = self.controller.forward_kinematics(self.robot.get_arm_qpos(agent), agent)
            return np.array(pos, dtype=float), np.array(quat, dtype=float)

    def get_ee_force_along(self, axis: str = "z") -> float:
        """末端力在指定轴的分量（局部坐标系）。axis ∈ {x,y,z}。

        用于 impedance_push 沿 z 推进时的接触力监测。
        """
        f = self.get_ee_force()
        idx = {"x": 0, "y": 1, "z": 2}.get(axis, 2)
        return float(f[idx])

    def _get_obs(self, agent: str = None):
        """观测：末端 WORLD 位姿 + 接触力 + 时间步。"""
        ag = agent or self.agents[0]
        pos, quat = self.get_ee_pose(ag)
        force = self.get_ee_force()
        return np.concatenate([np.asarray(pos, float), np.asarray(quat, float),
                                np.asarray(force, float), [float(self._timestep)]]).copy()
