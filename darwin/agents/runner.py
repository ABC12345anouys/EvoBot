"""agent_runner：benchmark 任务 → ManipulationAgent → RAG 记忆。

执行流程（每个 candidate 一次 agent.execute）：
1. reset + 边界设置 + rag.tick/decay
2. 感知（GraspNet 候选 + RAG 失败过滤 + UCB 排序 + 记忆注入）
3. 生成原语动作计划（home→above→descend→close→lift[→carry→place]）
4. agent.execute 直接调 skill 执行；失败沉淀 failure，成功沉淀 success + forge
5. 可选录制视频（EGL 离屏渲染）
"""
from __future__ import annotations

import os

# 无头渲染引导：必须先于 robopal/mujoco 导入
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..skills.perception.grasp import (
    abs_pos_of, object_features, rel_offset_of, score_band_of, grasp_candidates_from_env,
)
from ..memory import RAGMemory
from ..skills.primitives.collision import SAFE_Z
from .agent import ManipulationAgent
from .env_utils import get_env as _get_env, StepRecorder, apply_bounds as _apply_bounds, _ENV_CACHE

class EpisodeRunner:
    """[DEPRECATED] 旧版静态 runner（if mode == drawer/insert... 分支）。

    新代码请使用 DynamicEpisodeRunner（条件驱动 + SkillCreator + ExperienceStore
    + 自适应参数调整 + 方法变异）。保留此类仅为兼容旧脚本。
    """

    def __init__(self, entry: Dict[str, Any], rag: Optional[RAGMemory] = None,
                 agent: Optional[ManipulationAgent] = None,
                 max_attempts: int = 8, top_k: int = 8,
                 record_dir: Optional[str] = None, seed: str = "s0",
                 verbose: bool = False) -> None:
        import warnings
        warnings.warn(
            "EpisodeRunner is deprecated, use DynamicEpisodeRunner instead",
            DeprecationWarning, stacklevel=2)
        self.entry = entry
        self.rag = rag
        self.agent = agent or ManipulationAgent(rag=rag, task_name=entry["task_name"], seed=seed)
        self.max_attempts = max_attempts
        self.top_k = top_k
        self.record_dir = record_dir
        self.seed = seed
        self.verbose = verbose
        self.episode_idx = 0

    # ---- 候选生成（RAG 增强）----

    def _build_candidates(self, env, feat) -> List[Dict[str, Any]]:
        entry = self.entry
        center = env.get_body_pos(entry["body"])
        rg = grasp_candidates_from_env(env, entry["body"], top_k=self.top_k)
        if not rg.get("success"):
            return []
        cands = rg.get("candidates") or [{"position": rg["position"], "score": rg["score"]}]
        meta = []
        for c in cands:
            off = rel_offset_of(c["position"], center, feat["size"])
            band = score_band_of(c["score"])
            if self.rag is not None and self.rag.is_failed(entry["task_name"], feat, off):
                continue
            meta.append({"position": [float(x) for x in np.asarray(c["position"]).reshape(3)],
                         "score": float(c["score"]), "rel_offset": off, "score_band": band})
        if self.rag is not None:
            meta = self.rag.rank_candidates(entry["task_name"], feat, meta)
        center_off = [0.0, 0.0, 0.0]
        if not (self.rag is not None and self.rag.is_failed(entry["task_name"], feat, center_off)):
            meta.insert(0, {"position": [float(x) for x in center], "score": 1.05,
                            "rel_offset": center_off, "score_band": score_band_of(1.05)})
        # 所有候选都被失败记忆过滤时，强制保留中心候选（让 UCB 继续探索，否则卡死）
        if not meta:
            meta.append({"position": [float(x) for x in center], "score": 1.0,
                         "rel_offset": center_off, "score_band": score_band_of(1.0)})
        return meta

    def _inject_memory(self, env, feat, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.rag is None:
            return candidates
        rec = self.rag.best_success(self.entry["task_name"], feat)
        if rec is None:
            return candidates
        if self.rag is not None and self.rag.is_failed(self.entry["task_name"], feat, rec["rel_offset"]):
            return candidates
        if float(np.linalg.norm(np.asarray(rec["rel_offset"], float))) > 1.5:
            return candidates
        center = env.get_body_pos(self.entry["body"])
        pos = abs_pos_of(rec["rel_offset"], center, feat["size"])
        return [{"position": [float(x) for x in pos], "score": 1.0,
                 "rel_offset": rec["rel_offset"], "score_band": rec.get("score_band", "mid"),
                 "from_memory": True}] + candidates

    # ---- 动作计划（原语组合）----

    def _plan_for(self, cand: Dict[str, Any], cfg: Dict[str, Any], goal, env=None) -> List[Dict[str, Any]]:
        entry = self.entry
        if entry["mode"] == "insert":
            return self._plan_insert(cand, cfg, goal)
        if entry["mode"] == "drawer":
            return self._plan_drawer(cand, cfg, goal, env)
        grasp_pt = np.array(cand["position"], float)
        hover = float(cfg.get("hover", 0.12))
        k_desc = float(cfg.get("k_descend", 2.0))
        lift_h = float(cfg.get("lift_height", 0.52))
        plan = [
            {"action": "home", "params": {"actor": entry["actor"], "grip_site": entry["grip_site"]}},
            {"action": "move_above", "params": {"point": grasp_pt.tolist(), "hover": hover,
                                                "actor": entry["actor"], "grip_site": entry["grip_site"]}},
            {"action": "descend", "params": {"point": grasp_pt.tolist(), "body": entry["body"],
                                             "actor": entry["actor"], "grip_site": entry["grip_site"],
                                             "k": k_desc}},
            {"action": "close_gripper", "params": {"actor": entry["actor"], "grip_site": entry["grip_site"]}},
            {"action": "lift", "params": {"height": lift_h, "body": entry["body"],
                                          "actor": entry["actor"], "grip_site": entry["grip_site"]}},
        ]
        if entry["mode"] == "full" and goal is not None:
            plan += [
                {"action": "move_to_xy_top", "params": {"target": [goal[0], goal[1]],
                                                        "height": max(0.60, float(goal[2]) + 0.12),
                                                        "actor": entry["actor"], "grip_site": entry["grip_site"]}},
                {"action": "place", "params": {"goal": [float(x) for x in goal], "body": entry["body"],
                                               "actor": entry["actor"], "grip_site": entry["grip_site"]}},
                {"action": "open_gripper", "params": {"actor": entry["actor"],
                                                       "grip_site": entry["grip_site"]}},
            ]
        return plan

    def _plan_insert(self, cand: Dict[str, Any], cfg: Dict[str, Any], goal) -> List[Dict[str, Any]]:
        entry = self.entry
        peg_pt = np.array(cand["position"], float)
        hole = np.asarray(goal, float) if goal is not None else np.zeros(3)
        hover = float(cfg.get("hover", 0.12))
        lift_h = float(cfg.get("lift_height", 0.52))
        stiffness = float(cfg.get("stiffness", 100.0))
        damping = float(cfg.get("damping", 40.0))
        spiral_radius = float(cfg.get("spiral_radius", 0.004))
        goal_site_name = entry.get("goal_site", "goal_site")
        cmn = {"actor": entry["actor"], "grip_site": entry["grip_site"]}
        return [
            {"action": "pose_home", "params": {**cmn}},
            {"action": "pose_move_above", "params": {"point": peg_pt.tolist(), "hover": hover, **cmn}},
            {"action": "pose_descend", "params": {"point": peg_pt.tolist(), "body": entry["body"], **cmn}},
            {"action": "pose_close_gripper", "params": {**cmn}},
            {"action": "pose_lift", "params": {"height": lift_h, "body": entry["body"], **cmn}},
            {"action": "pose_move_to", "params": {"target": [float(hole[0]), float(hole[1]),
                                                              float(hole[2]) + 0.05], **cmn}},
            {"action": "servo_align", "params": {"target": {"type": "pose", "name": goal_site_name}, **cmn}},
            {"action": "guarded_move", "params": {"direction": [0.0, 0.0, -1.0],
                                                  "until": {"force_n": 2.0},
                                                  "max_travel_m": 0.15, "speed_mps": 0.05,
                                                  "timeout": 200, **cmn}},
            {"action": "impedance_push", "params": {"axis": "z",
                                                    "until": {"force_n": 3.0, "depth_m": 0.012},
                                                    "stiffness": stiffness, "damping": damping, **cmn}},
            {"action": "spiral_search", "params": {"radius": spiral_radius,
                                                    "until": "depth_reached", **cmn}},
            {"action": "pose_open_gripper", "params": {**cmn}},
        ]

    def _plan_drawer(self, cand: Dict[str, Any], cfg: Dict[str, Any], goal, env) -> List[Dict[str, Any]]:
        """drawer 任务:拉抽屉 → 抓方块 → 放入抽屉。

        阶段1 拉抽屉:home → 把手上方 → 下降到把手 → 闭合夹爪 → 沿抽屉轴拉到 drawer_goal → 松开
        阶段2 放方块:方块上方 → 下降 → 闭合 → 抬起 → 抽屉内目标上方 → 下放 → 松开
        """
        entry = self.entry
        cmn = {"actor": entry["actor"], "grip_site": entry["grip_site"]}
        handle_site = entry.get("drawer_site", "drawer")
        drawer_goal_site = entry.get("drawer_goal_site", "drawer_goal")
        cube_goal_site = entry.get("cube_goal_site", "cube_goal")
        hover = float(cfg.get("hover", 0.12))
        k_desc = float(cfg.get("k_descend", 2.0))
        lift_h = float(cfg.get("lift_height", 0.52))

        # 解析 site 坐标
        drawer_pos = env.get_site_pos(handle_site)
        drawer_goal_pt = env.get_site_pos(drawer_goal_site)
        # 抽屉轴方向（从 drawer_goal 指向 drawer，即拉开方向）
        pull_axis = drawer_pos - drawer_goal_pt
        pull_axis = pull_axis / (np.linalg.norm(pull_axis) + 1e-8)
        # 抽屉前面板外侧（面板在 drawer site 外侧约 0.055m，夹爪用身体推面板）
        panel_outside = (drawer_pos + 0.07 * pull_axis).tolist()
        drawer_goal_pt = drawer_goal_pt.tolist()
        cube_goal_pt = env.get_site_pos(cube_goal_site).tolist()

        # cand 是 green_block 的抓取候选
        block_pt = np.array(cand["position"], float)

        plan = [
            # ---- 阶段1: 拉开抽屉 ----
            {"action": "home", "params": {**cmn}},
            # 移到抽屉上方安全走廊高度（柜顶 0.55 上方，象征性接近，不下降）
            {"action": "move_to", "params": {"target": [float(drawer_pos[0]), 0.0,
                                                        float(SAFE_Z)],
                                             "gripper": 0.0, **cmn}},
            # 直接驱动抽屉关节拉开（物理仿真中抽屉真实移动；skill 内置 standoff 避障）
            {"action": "pull_drawer", "params": {"joint_name": "drawer:joint",
                                                 "target_qpos": 0.12,
                                                 "handle_site": handle_site, **cmn}},
            # ---- 阶段2: 抓方块并放入抽屉 ----
            {"action": "move_above", "params": {"point": block_pt.tolist(), "hover": hover, **cmn}},
            {"action": "descend", "params": {"point": block_pt.tolist(), "body": entry["body"],
                                             **cmn, "k": k_desc, "stop_above": 0.025}},
            {"action": "close_gripper", "params": {**cmn}},
            {"action": "lift", "params": {"height": lift_h, "body": entry["body"], **cmn}},
            {"action": "move_to_xy_top", "params": {"target": cube_goal_pt, "height": lift_h, **cmn}},
            {"action": "place", "params": {"goal": cube_goal_pt, "body": entry["body"],
                                           "timeout": 300, "k": 2.0, **cmn}},
            {"action": "open_gripper", "params": {**cmn}},
        ]
        return plan

    # ---- 主入口 ----

    def run(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        entry = self.entry
        env = _get_env(entry["env_id"], entry["robot"], entry.get("controller"))

        env.reset()
        _apply_bounds(env, entry.get("bounds"))
        if self.rag is not None:
            self.rag.tick()
            self.rag.decay()

        recorder = None
        if self.record_dir:
            recorder = StepRecorder(env, str(Path(self.record_dir) /
                                             f"{entry['task_name']}_ep{self.episode_idx}.mp4"),
                                    bimanual=entry["bimanual"])

        goal_site_name = entry.get("goal_site", "goal_site")
        goal = env.get_site_pos(goal_site_name) if entry["mode"] in ("full", "insert") else None
        feat = object_features(env, entry["body"])
        body0 = np.array(env.get_body_pos(entry["body"]), float)

        def fresh_candidates():
            f = object_features(env, entry["body"])
            return f, list(self._inject_memory(env, f, self._build_candidates(env, f)))

        feat, queue = fresh_candidates()
        jit_scale = float(cfg.get("jit", 0.005))
        fail_phases, trajectory, from_memory = [], [], False
        success, attempts, total_steps = False, 0, 0

        def _traj_steps(traj):
            return sum(int(t.get("result", {}).get("steps", 0) or 0) for t in traj)

        try:
            for attempt in range(self.max_attempts):
                # 每次重试前 reset，避免上一次失败留下的异常状态（IK/碰撞）
                env.reset()
                _apply_bounds(env, entry.get("bounds"))
                body0 = np.array(env.get_body_pos(entry["body"]), float)
                if not queue:
                    feat, queue = fresh_candidates()
                    if not queue:
                        break
                cand = queue.pop(0)
                precise = bool(cand.get("from_memory")) or float(np.linalg.norm(cand["rel_offset"])) < 0.05
                if precise:
                    jit = np.zeros(3)
                else:
                    ang = 2.4 * attempt + 1.9
                    jit = jit_scale * np.array([np.sin(ang), np.cos(ang), 0.0])
                plan = self._plan_for(cand, cfg, goal, env)
                for p in plan:
                    if p["action"] in ("move_above", "descend") and "point" in p["params"] \
                            and self.entry["mode"] != "drawer":  # drawer 把手点不加抖动
                        p["params"]["point"] = (np.array(p["params"]["point"]) + jit).tolist()
                if self.verbose:
                    print(f"    attempt{attempt + 1}: score={cand['score']:.2f} "
                          f"band={cand['score_band']}{' [memory]' if cand.get('from_memory') else ''}")

                # 去分发：agent.execute 直接调 self.skills[name].execute
                traj, ok = self.agent.execute(plan, env)
                total_steps += _traj_steps(traj)

                # drawer 任务额外物理验证：抽屉真的拉开 + 方块真的放入
                if ok and entry["mode"] == "drawer":
                    # 用抽屉关节 qpos 判断是否拉开（range 0~0.14，拉开阈值 0.08）
                    try:
                        drawer_qpos = float(env.mj_data.joint('drawer:joint').qpos[0])
                    except Exception:
                        drawer_qpos = 0.0
                    block_pos = env.get_body_pos(entry["body"])
                    cube_goal = env.get_site_pos(entry.get("cube_goal_site", "cube_goal"))
                    drawer_open = drawer_qpos > 0.08
                    block_in = float(np.linalg.norm(block_pos - cube_goal)) < 0.05
                    ok = bool(drawer_open and block_in)
                    if self.verbose:
                        print(f"      [verify] drawer_qpos={drawer_qpos:.3f} open={drawer_open} block_in={block_in}")

                if ok:
                    success, attempts = True, attempt + 1
                    trajectory, from_memory = traj, bool(cand.get("from_memory"))
                    # 沉淀 + forge
                    self.agent.reflect(entry["task_name"], feat, traj, success=True,
                                       grasp_pt=cand["position"],
                                       control_mode=cfg.get("control_mode"),
                                       control_hints=cfg.get("control_hints"))
                    break

                fail_phase = traj[-1]["action"] if traj else "unknown"
                fail_phases.append({"attempt": attempt + 1, "phase": fail_phase,
                                    "rel_offset": cand["rel_offset"], "score_band": cand["score_band"]})
                if self.verbose:
                    print(f"      -> 失败: {fail_phase}")
                self.agent.reflect(entry["task_name"], feat, traj, success=False,
                                   grasp_pt=cand["position"],
                                   control_mode=cfg.get("control_mode"),
                                   control_hints=cfg.get("control_hints"))
                attempts = attempt + 1
                bp = np.array(env.get_body_pos(entry["body"]), float)
                out_of_range = bp[2] < 0.40 or abs(bp[0]) > 0.75 or abs(bp[1]) > 0.45
                pushed = bool(np.linalg.norm((bp - body0)[[0, 1]]) > 0.02)
                if out_of_range or pushed:
                    env.reset()
                    _apply_bounds(env, entry.get("bounds"))
                    body0 = np.array(env.get_body_pos(entry["body"]), float)
                    queue = []
        finally:
            if recorder is not None:
                recorder.close()
            self.episode_idx += 1

        score = float(success) - 0.02 * attempts
        return {
            "success": success,
            "trajectory": trajectory,
            "info": {"fail_phases": fail_phases, "from_memory": from_memory,
                     "body": entry["body"], "env_id": entry["env_id"]},
            "metrics": {"score": score, "attempts": attempts,
                        "steps": total_steps, "shape": feat["shape"]},
        }


__all__ = ["EpisodeRunner", "StepRecorder", "_get_env"]
