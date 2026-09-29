"""SkillConfigStore：大 skill 的 per-env 参数文件（反思学习的落盘归宿）。

一份标准操作 skill（ik_servo / lift / place / carry…）跨 env 共享同一份
Python；env 之间的差异全部在这里：

    darwin/skills/configs/<skill>.default.yaml   # 底座默认（robopal 实测值）
    darwin/skills/configs/<skill>.<env>.yaml     # 某 env 的覆盖 + 反思 history

agent_learner 进程每轮 attempt 后把 deltas 与结果写回本文件（原子写）；
sim_worker 进程只读不写。换 env = 换一份 YAML，代码一行不改。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent / "configs"

# key: (类型, 默认值, 最小, 最大, 说明)。update 时按此做类型转换与范围裁剪，
# 学习结果越界会被 clip 而不是把 sim 搞崩。
PARAM_SPEC: Dict[str, Tuple[type, Any, float, float, str]] = {
    # ---- 通用伺服几何 ----
    "hover": (float, 0.12, 0.02, 0.25, "悬停高度 m（抓取前/搬运中继）"),
    "k": (float, 2.0, 0.5, 12.0, "末端伺服 P 增益（ik_servo 链）"),
    "k_descend": (float, 2.0, 0.5, 15.0, "下降 P 增益（robopal cart 链）"),
    "vcap": (float, 0.05, 0.01, 2.0, "每步末端速度上限 m"),
    "carry_vcap": (float, 0.05, 0.01, 1.0,
                   "carry 搬运阶段每步速度上限 m（降惯性防夹持物滑脱）"),
    "carry_margin": (float, 0.0, 0.0, 0.30,
                     "绕障余量 m：stall（撞墙磨停）后反思逐步加大，飞越高障碍或侧向绕行"),
    "reach_tol": (float, 0.012, 0.002, 0.03, "到达容差 m"),
    "lift_height": (float, 0.52, 0.05, 0.80,
                    "抬升高度（robopal 绝对 z；libero 为相对 delta）m"),
    "release_offset": (float, 0.0, 0.0, 0.10, "放置释放点距目标面高度 m"),
    "timeout_scale": (float, 1.0, 0.5, 3.0, "skill 超时缩放系数"),
    "stop_above": (float, -0.01, -0.09, 0.10, "下降停在物体上方的距离 m（负=穿入物体；-0.075 档位用于 moka_pot 等高处壶体：z_goal 到壶身中部）"),
    "contact_stop_band": (float, 0.05, 0.02, 0.12, "descend 接触软停豁免带半宽 m；高位物体 OSC 下降受限时容 TCP 偏差"),
    "jit": (float, 0.0, 0.0, 0.03, "抓取点抖动幅度 m（0=关；精密几何默认禁抖，探索须显式开）"),
    # ---- place ----
    "place_k": (float, 1.2, 0.3, 4.0, "放置伺服 P 增益"),
    "place_timeout": (int, 150, 60, 400, "放置超时步数"),
    "place_vcap": (float, 0.05, 0.005, 0.5,
                   "place 下落阶段每步速度上限 m（默认回落 vcap；瓶类滑脱边际"
                   "物体调小——goal:2 实证 place 起始 3~7 步内 slipped 8/8）"),
    # ---- 力控插装 ----
    "stiffness": (float, 100.0, 20.0, 300.0, "CARTIMP 刚度"),
    "damping": (float, 40.0, 5.0, 100.0, "CARTIMP 阻尼"),
    "spiral_radius": (float, 0.004, 0.001, 0.012, "螺旋搜索半径 m"),
    "push_force_n": (float, 3.0, 0.5, 10.0, "压入目标力 N"),
    "push_depth_m": (float, 0.012, 0.002, 0.04, "压入目标深度 m"),
    "force_limit": (float, 15.0, 5.0, 40.0, "力上限 N"),
    # ---- 抓取精细化（容器偏心插指；GraspNet 给方向，几何量在此学）----
    "grasp_max_width": (float, 0.075, 0.04, 0.08,
                        "候选过滤：夹爪允许的最大物体宽度 m"),
    "grasp_container_offset_ratio": (float, 0.70, 0.50, 1.0,
                                     "容器偏心量 / 半宽 比例"),
    "grasp_container_z_delta": (float, 0.030, 0.010, 0.045,
                                "容器插指深度（沿口顶以下）m"),
    # ---- 失败回退续跑（IPC resume）----
    "resume_rollback_steps": (int, 10, 1, 60,
                              "失败后恢复多少个物理步前的状态再原位续跑"),
    # ---- RoboDojo 感知（darwin/perception）----
    "pcd_voxel": (float, 0.005, 0.002, 0.02, "点云体素下采样 m"),
    "pcd_depth_min": (float, 0.10, 0.02, 0.30, "反投影深度下限 m"),
    "pcd_depth_max_head": (float, 1.20, 0.5, 2.5, "head 相机深度上限 m"),
    "pcd_depth_max_wrist": (float, 0.80, 0.3, 1.5, "wrist 相机深度上限 m"),
    "ws_x_min": (float, -0.45, -0.80, 0.0, "工作空间盒 x 下界 m"),
    "ws_x_max": (float, 0.45, 0.0, 0.80, "工作空间盒 x 上界 m"),
    "ws_y_min": (float, -0.35, -0.60, 0.0, "工作空间盒 y 下界 m"),
    "ws_y_max": (float, 0.25, 0.0, 0.60, "工作空间盒 y 上界 m"),
    "ws_z_min": (float, 0.50, 0.30, 0.70, "工作空间盒 z 下界 m"),
    "ws_z_max": (float, 1.10, 0.90, 1.40, "工作空间盒 z 上界 m"),
    "table_margin": (float, 0.006, 0.001, 0.03, "桌面去除阈值 m"),
    "cluster_radius": (float, 0.012, 0.004, 0.03, "欧式聚类体素 m"),
    "cluster_min_pts": (int, 80, 20, 300, "聚类最小点数"),
    "sam_enable": (int, 0, 0, 1, "SAM 点云提纯开关（0/1，只提纯不建实例）"),
    "expected_instances": (int, 3, 1, 10, "期望实例数（stack_bowls=3）"),
    # ---- RoboDojo 走廊/可达/候选 ----
    "corridor_z_max": (float, 0.90, 0.80, 1.00,
                       "巡航高度上限（tip 系 m；0.865=沿口+5cm 实证安全）"),
    "corridor_clear": (float, 0.05, 0.02, 0.15, "巡航高度超出实例顶余量 m"),
    "stall_frames": (int, 15, 5, 60, "腿级 stall 判定连续无位移帧数"),
    "leg_timeout": (int, 120, 30, 400, "单腿超时帧数"),
    "reach_r_min": (float, 0.12, 0.05, 0.30, "可达包络内半径 m"),
    "reach_r_max": (float, 0.48, 0.30, 0.70, "可达包络外半径 m"),
    "reach_rho_comfort": (float, 0.38, 0.20, 0.50, "臂分配舒适半径 m"),
    "reach_z_min": (float, 0.55, 0.40, 0.70, "可达 z 下限 m"),
    "reach_z_max": (float, 1.00, 0.85, 1.20, "可达 z 上限 m"),
    "max_candidates": (int, 4, 1, 8, "每碗抓取候选预算（stall/miss 弹下一个）"),
    "grasp_y_min": (float, -0.25, -0.45, 0.0, "抓取点 y 下界 m（界外翻向丢弃）"),
    "grasp_y_max": (float, 0.06, -0.10, 0.30, "抓取点 y 上界 m（界外翻向丢弃）"),
    # ---- LLM Agent ----
    "max_replans": (int, 4, 0, 10, "Agent Loop 最大重规划次数（失败喂回 LLM）"),
    "grasp_lift_delta": (float, 0.12, 0.04, 0.30, "抓取后抬起高度 m"),
    "open_steps": (int, 10, 3, 30, "释放时夹爪张开保持帧数"),
}

_HEADER = (
    "# 大 skill 参数文件（agent_learner 反思自动维护，可手工编辑）\n"
    "# params: 当前生效参数，范围/含义见 darwin/skills/skill_config.py PARAM_SPEC\n"
    "# history: 每个 attempt 的 cfg/结果/参数变更/反思笔记，换 env 分析与回溯用\n"
)


def spec_defaults() -> Dict[str, Any]:
    return {k: v[1] for k, v in PARAM_SPEC.items()}


def public_spec() -> Dict[str, Any]:
    """ready 消息里带给 agent 的参数 schema（不含内部说明以外的多余字段）。"""
    return {k: {"type": v[0].__name__, "default": v[1],
                "min": v[2], "max": v[3], "desc": v[4]}
            for k, v in PARAM_SPEC.items()}


class SkillConfigStore:
    """加载 / 更新 / 落盘一份 <skill>.<env>.yaml。"""

    def __init__(self, skill_name: str, env: str, params: Dict[str, Any],
                 history: List[Dict[str, Any]], path: Path,
                 learned: bool = False,
                 theta: Optional[Dict[str, Any]] = None) -> None:
        self.skill_name = skill_name
        self.env = env
        self.params = params
        self.history = history
        self.path = path
        self.learned = learned
        # Θ 后验（physics/posterior.py）：这个世界的物理事实（μ、到达
        # 短缩、预算需求），跨 attempt 复用；与 params（本次怎么跑）根本
        # 区别。不入 PARAM_SPEC 过滤——它是测量域，不是调参旋钮。
        self.theta: Dict[str, Any] = dict(theta or {})

    # ---- 加载 ----

    @classmethod
    def load(cls, skill_name: str, env: str,
             task: str = None) -> "SkillConfigStore":
        """default → env → task 三级覆盖。

        default 在底；env（如 libero）覆盖；task（如 libero_spatial_0）再覆盖。
        task 文件不存在 = 无覆盖，正常返回（不种子化，避免文件爆炸）。
        反思写回时写到 task 级文件（若 task 给定），否则写 env 级。
        """
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        default_path = CONFIGS_DIR / f"{skill_name}.default.yaml"
        env_path = CONFIGS_DIR / f"{skill_name}.{env}.yaml"
        task_path = (CONFIGS_DIR / f"{skill_name}.{task}.yaml"
                     if task else None)

        params = spec_defaults()
        history: List[Dict[str, Any]] = []
        learned = False
        theta: Dict[str, Any] = {}

        # 1. default
        default_data = cls._read(default_path)
        if default_data:
            params.update(cls._coerce(default_data.get("params", {})))
        # 2. env 覆盖
        if env_path.exists():
            env_data = cls._read(env_path) or {}
            params.update(cls._coerce(env_data.get("params", {})))
            history = list(env_data.get("history", []) or [])
            learned = bool(env_data.get("learned", False))
            theta = dict(env_data.get("theta", {}) or {})
        else:
            # 首次：种子化 env 文件
            store = cls(skill_name, env, dict(params), history, env_path)
            store.save()
        # 3. task 覆盖（不种子化读取，但 write_path 总指向 task 文件）
        write_path = env_path
        if task_path:
            # task 给定时，写回总写到 task 级文件（首次 save 时创建）
            write_path = task_path
            env = task
            if task_path.exists():
                task_data = cls._read(task_path) or {}
                params.update(cls._coerce(task_data.get("params", {})))
                history = history + list(task_data.get("history", []) or [])
                learned = bool(task_data.get("learned", learned))
                theta.update(dict(task_data.get("theta", {}) or {}))

        return cls(skill_name, env, params, history,
                   write_path, learned=learned, theta=theta)

    # ---- Θ 后验更新（割的落点，agent 侧应用 sim 侧算好的割）----

    def apply_theta_cuts(self, cuts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """应用 theta_cut 列表（physics/cuts.py 的 Cut 序列化形态）。
        返回本次收紧的 {name: [old, new_domain]}（落 history 用）。"""
        from ..physics.posterior import ThetaPosterior
        post = ThetaPosterior(self.theta)
        applied: Dict[str, Any] = {}
        for c in cuts or []:
            if not isinstance(c, dict) or c.get("kind") != "theta_cut":
                continue
            name = c.get("name")
            if not name:
                continue
            old = post.domain(name)
            cur = post.cut(name, lo=c.get("lo"), hi=c.get("hi"),
                           conf=float(c.get("conf", 1.0)))
            new = (cur["lo"], cur["hi"])
            if new != old:
                applied[name] = [list(old), list(new)]
        if applied:
            self.theta = post.to_dict()
        return applied

    # ---- 更新 ----

    def update_params(self, deltas: Dict[str, Any]) -> Dict[str, Any]:
        """应用反思变更。deltas 支持两种形态：

        - {key: new_value}
        - {key: [old, new]}（reflection.adapt_cfg 的输出，取 new）
        返回实际发生的 {key: 新值}。
        """
        applied: Dict[str, Any] = {}
        for key, val in deltas.items():
            if key not in PARAM_SPEC:
                continue  # 未知参数不入文件，防止 LLM/规则写脏 key
            new_val = val[1] if isinstance(val, (list, tuple)) and len(val) == 2 else val
            new_val = self._clip(key, new_val)
            if new_val != self.params.get(key):
                self.params[key] = new_val
                applied[key] = new_val
        return applied

    def append_history(self, record: Dict[str, Any]) -> None:
        self.history.append(record)

    def mark_learned(self, learned: bool = True) -> None:
        self.learned = learned

    # ---- 落盘（原子写 tmp+rename，写坏不毁旧文件）----

    def save(self) -> None:
        payload = {"skill": self.skill_name, "env": self.env,
                   "learned": self.learned,
                   "params": dict(sorted(self.params.items())),
                   "history": self.history}
        if self.theta:
            payload["theta"] = self.theta
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent),
                                   prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(_HEADER)
                yaml.safe_dump(payload, f, allow_unicode=True,
                               sort_keys=False, default_flow_style=False)
            os.replace(tmp, self.path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    # ---- 内部工具 ----

    @staticmethod
    def _read(path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None

    @staticmethod
    def _coerce(raw: Dict[str, Any]) -> Dict[str, Any]:
        """只收 PARAM_SPEC 内的 key，并做类型转换（YAML 手误也不会脏数据）。"""
        out: Dict[str, Any] = {}
        for key, val in raw.items():
            if key in PARAM_SPEC:
                out[key] = SkillConfigStore._clip(key, val)
        return out

    @staticmethod
    def _clip(key: str, val: Any) -> Any:
        typ, _, lo, hi, _ = PARAM_SPEC[key]
        try:
            val = typ(val)
        except (TypeError, ValueError):
            return PARAM_SPEC[key][1]
        if typ is int:
            return int(min(hi, max(lo, val)))
        if typ is float:
            return float(min(hi, max(lo, float(val))))
        return val
