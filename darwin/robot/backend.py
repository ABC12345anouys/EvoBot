"""RobotBackend 抽象层:让 motion 原语(path_plan/collision_check)与具体机型解耦。

设计(Adapter/Strategy + 配置驱动):
- RobotBackend ABC:定义运动学/碰撞高层接口(get_arm_qpos/plan_path/execute_trajectory/check_*_collision)
- RobotProfile:YAML 配置(per-actor:joint/link 名/URDF 路径/eef body)
- make_backend:按 profile.backend_type 选子类(注册表)

Skill 层只调 backend 接口,签名/Registry/Planner 不动(模板);
换机型只写 Backend 子类 + YAML(动态)。LLM 作为开发时生成器(运行时纯 Python 多态)。

双臂向前兼容:Profile actors dict(单臂一个 entry)、方法全 actor 参数化、
mplib planner 缓存 (manipulator, actor) key、SRDF 跨臂对不禁用→自动检测避碰。
协同双臂规划(相对位姿约束)留给未来 BimanualBackend,mplib 不支持。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml


PROFILES_DIR = Path(__file__).resolve().parent.parent / "assets" / "profiles"


@dataclass
class ArmConfig:
    """单臂配置(URDF 原生名 + mujoco body 前缀)。"""
    joint_names: List[str]          # URDF 原生 j1..j7(无前缀)
    move_group_link: str = ""       # mplib move_group,如 link7
    mujoco_joint_prefix: str = ""   # robopal 双臂前缀 "0_" / "1_";纯 URDF 后端可空
    mujoco_base_body: str = ""      # 如 0_base_link;空=base 在原点
    mujoco_eef_body: str = ""       # 如 0_eef;用于算 eef→move_group 偏移


@dataclass
class RobotProfile:
    """机型配置(从 YAML 加载)。单臂 actors 只一个 entry;双臂两个。"""
    name: str
    backend_type: str               # 选哪个 Backend 子类(注册表 key)
    manipulator: str = ""           # mplib 缓存 key / 机器人名
    urdf_path: str = ""             # URDF 路径(mplib 用)
    mesh_root: str = ""             # mesh 目录(用于 convex hull 生成)
    srdf_path: Optional[str] = None # null=自动生成
    urdf_mesh_rewrite: Dict[str, str] = field(default_factory=dict)  # package:// → meshes/
    adjacent_link_pairs: List[List[str]] = field(default_factory=list)  # 单臂内相邻对
    extra_disabled_pairs: List[List[str]] = field(default_factory=list)  # 工作构型误报对
    actors: Dict[str, ArmConfig] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "RobotProfile":
        data = yaml.safe_load(Path(path).read_text()) or {}
        actors_raw = data.pop("actors", {}) or {}
        actors = {
            k: ArmConfig(
                joint_names=a.get("joint_names", []),
                move_group_link=a.get("move_group_link", ""),
                mujoco_joint_prefix=a.get("mujoco_joint_prefix", ""),
                mujoco_base_body=a.get("mujoco_base_body", ""),
                mujoco_eef_body=a.get("mujoco_eef_body", ""),
            )
            for k, a in actors_raw.items()
        }
        return cls(
            name=data.get("name", Path(path).stem),
            backend_type=data["backend_type"],
            manipulator=data.get("manipulator", data.get("name", Path(path).stem)),
            urdf_path=data.get("urdf_path", ""),
            mesh_root=data.get("mesh_root", ""),
            srdf_path=data.get("srdf_path"),
            urdf_mesh_rewrite=data.get("urdf_mesh_rewrite", {}),
            adjacent_link_pairs=[list(p) for p in data.get("adjacent_link_pairs", [])],
            extra_disabled_pairs=[list(p) for p in data.get("extra_disabled_pairs", [])],
            actors=actors,
        )


class RobotBackend(ABC):
    """机器人后端抽象:屏蔽 mujoco/ROS/Franka SDK 差异。

    Skill 只调本接口;子类实现具体 SDK 调用。所有方法 actor 参数化(双臂向前兼容)。
    """

    def __init__(self, profile: RobotProfile) -> None:
        self.profile = profile
        self._env: Optional[Any] = None

    def bind(self, env: Any) -> None:
        """构造后绑定 env 引用(用于读 mujoco 真值/执行)。"""
        self._env = env

    @property
    def env(self) -> Any:
        if self._env is None:
            raise RuntimeError("Backend 未 bind(env);env 构造后必须调用 bind")
        return self._env

    # ---- 状态读取 ----
    @abstractmethod
    def get_arm_qpos(self, actor: str) -> np.ndarray:
        """当前臂关节角(7 维)。"""

    @abstractmethod
    def get_eef_pose(self, actor: str) -> Tuple[np.ndarray, np.ndarray]:
        """末端 world 位姿 [pos(3), quat(4) [w,x,y,z]]。"""

    @abstractmethod
    def get_eef_to_move_group_offset(self, actor: str) -> np.ndarray:
        """eef→move_group_link 位置偏移(world frame)。"""

    @abstractmethod
    def get_base_pose(self, actor: str) -> np.ndarray:
        """base 在 world 的位姿 [pos(3), quat(4)](供 mplib set_base_pose)。"""

    # ---- 资产准备 ----
    @abstractmethod
    def prepare_collision_assets(self) -> None:
        """URDF/SRDF/mesh/convex 幂等准备(惰性调用,首用时执行)。"""

    # ---- 高层运动操作 ----
    @abstractmethod
    def plan_path(self, goal_pose, actor: str, frame: str = "world",
                  current_qpos: Optional[np.ndarray] = None, **opts) -> Dict[str, Any]:
        """规划关节空间路径。返回 {ok, status, trajectory, duration, reason}。"""

    @abstractmethod
    def execute_trajectory(self, traj, actor: str, skip: int = 2,
                          timeout: int = 200) -> Dict[str, Any]:
        """执行关节轨迹(准静态/流式)。返回 {executed, reason, final_qpos}。"""

    @abstractmethod
    def check_self_collision(self, actor: Optional[str] = None,
                             qpos_arm: Optional[np.ndarray] = None) -> List[Any]:
        """自碰撞检测。None=用 env 当前全 qpos;给 actor+qpos_arm=代入该臂查全 body(含跨臂对)。"""

    @abstractmethod
    def check_env_collision(self, actor: str, qpos_arm: Optional[np.ndarray] = None,
                            use_attach: bool = True, with_pc: bool = False) -> List[Any]:
        """环境碰撞检测。"""


# Backend 子类注册表
_BACKEND_REGISTRY: Dict[str, type] = {}

def register_backend(name: str, cls: type) -> None:
    """注册 Backend 子类(子类模块 import 时调用)。"""
    _BACKEND_REGISTRY[name] = cls

def make_backend(profile_name: str, env: Optional[Any] = None) -> Optional[RobotBackend]:
    """按 profile_name 加载 YAML + 按 backend_type 构造 Backend,可选 bind(env)。

    profile 不存在时返回 None(motion 原语会返回 planner_unavailable,不崩)。
    用于 runner._get_env 在构造 env 时注入 backend。
    """
    yaml_path = PROFILES_DIR / f"{profile_name}.yaml"
    if not yaml_path.exists():
        return None
    profile = RobotProfile.from_yaml(yaml_path)
    cls = _BACKEND_REGISTRY.get(profile.backend_type)
    if cls is None:
        raise ValueError(f"未知 backend_type: {profile.backend_type}(profile={profile_name})")
    backend = cls(profile)
    if env is not None:
        backend.bind(env)
    return backend
