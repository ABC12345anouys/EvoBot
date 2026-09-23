"""runner_dynamic:目标条件驱动的动态规划执行器(planner + 规划执行闭环)。
与runner.py的区别（架构层面）：
- runner.py 用 `if entry["mode"] == "drawer"/"insert"/...` 按**任务名写死**整条 skill 链；
  新任务必须改 _plan_for 加分支。
- 本文件把任务声明为一组**有序的目标条件（objectives）**，执行器每轮：
      observe（哪些条件还没满足）
    → select method（方法库里谁能达成该条件）
    → make_steps（用 env 实时坐标现场绑定参数，动态生成 skill 序列）
    → execute（逐 skill 执行，事件写入 EpisodeLogger）
    → 回到 observe（已满足的目标自动跳过，失败则换抓取候选/reset 重试）
  执行端没有任何任务名 if-else；新任务 = 在 benchmark entry 里声明 objectives，
  或在声明适配层 objectives_from_entry 加一行配置映射。
objective 声明（entry["objectives"]，也可由旧 entry 字段自动适配）：
    {"kind": "joint_ge",      "joint": "drawer:joint", "threshold": 0.08,
                            "target_qpos": 0.12, "handle_site": "drawer"}
    {"kind": "body_lifted",  "body": "green_block", "height": 0.52}
    {"kind": "body_near_site", "body": "green_block", "site": "cube_goal",
                            "tol": 0.05, "flavor": "cart", "stop_above": 0.025}
    flavor="pose" 时走位姿/力控原语链（peg-in-hole）；flavor="cart" 走笛卡尔抓放链。
EpisodeLogger 同步挂载：sim step 每 N 步采样一次物体位置/机械臂状态，
skill 边界强制快照，事后看 JSONL 即可定位"哪个 skill、物体在哪、间隙多少"。
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
from .episode_logger import EpisodeLogger
from .experience_store import ExperienceStore
from .env_utils import get_env as _get_env, StepRecorder, apply_bounds as _apply_bounds, _ENV_CACHE
from .objectives import (GoalCond, JointAtLeast, BodyLifted, BodyNearSite,
                          objectives_from_entry)
from .methods import (PlanContext, Method, OpenDrawerMethod, GraspLiftMethod,
                      TransferCartMethod, TransferPoseMethod, _grasp_chain,
                      entry_get, METHOD_REGISTRY, select_method)


# 失败原因分类（供经验库统计与画图）
# 注意顺序：place_failed 须在 timeout 前（place_timeout 属于放置失败，非移动超时）；
# grip_failed 在 place_failed 前，但不含 "slip"（slipped 是放置段滑脱，归 place_failed）；
# descend 卡死（*_descend_stalled）归 grip_failed：对策是提高逼近增益 k_descend
# （见 ik_servo.py 的 reason 注释），绝不能落入 timeout 分支反向降 k。
_FAIL_KEYWORDS = {
    "collision": ["collision", "collide", "撞", "obstacle"],
    "grip_failed": ["no_grip", "grip_fail", "grip_no_contact", "lift_no_grip",
                    "descend_stalled", "没夹", "夹不住", "xy_drift"],
    "place_failed": ["place_unstable", "place_timeout", "unstable", "slipped",
                     "slip", "放置不稳", "放置失败"],
    "timeout": ["timeout", "timed out", "超时", "卡住"],
    "force_exceed": ["force", "force_exceed", "overload", "力超", "stuck"],
    "episode_terminated": ["episode_terminated", "terminated episode"],
}


def _classify_failure(result: Dict[str, Any], action_name: str) -> str:
    """根据 skill 执行结果的 reason/error 文本分类失败原因。"""
    text = f"{result.get('reason', '')} {result.get('error', '')} {action_name}".lower()
    for cat, kws in _FAIL_KEYWORDS.items():
        if any(kw in text for kw in kws):
            return cat
    if "unknown skill" in text:
        return "unknown_skill"
    return "other"


class _EpisodeHandle:
    """prepare_episode 返回的常驻执行环境（env/logger/候选生成器的容器）。"""

    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


# ============================================================
# 3. 动态规划执行器
# ============================================================

class DynamicEpisodeRunner:
    """observe → plan(method) → execute → re-observe 的条件驱动执行器。"""

    def __init__(self, entry: Dict[str, Any], rag: Optional[RAGMemory] = None,
                 agent: Optional[ManipulationAgent] = None,
                 max_attempts: int = 8, top_k: int = 8,
                 record_dir: Optional[str] = None, seed: str = "s0",
                 verbose: bool = False,
                 log_dir: Optional[str] = None, sample_every: int = 10,
                 registry=None, experience=None) -> None:
        self.entry = entry
        self.rag = rag
        self.agent = agent or ManipulationAgent(rag=rag, task_name=entry["task_name"], seed=seed)
        self.max_attempts = max_attempts
        self.top_k = top_k
        self.record_dir = record_dir
        self.seed = seed
        self.verbose = verbose
        self.log_dir = log_dir
        self.sample_every = sample_every
        self.episode_idx = 0
        # 调用链方法注册表：None 时用 default_registry（内置 Python 方法 +
        # methods/*.yaml 声明式覆盖层）；可注入自定义 ChainRegistry 做实验。
        self._registry = registry
        # 经验库：按 method 聚合 success/failure，供 SkillCreator 防重复与总结；
        # 注入 None 时自动建一个 logs/experience 下的实例。
        self.experience = experience or ExperienceStore()

    @property
    def registry(self):
        if self._registry is None:
            from .chain_registry import default_registry
            self._registry = default_registry()
        return self._registry

    def _llm_client(self):
        """取 agent 的 llm 客户端（可能为 None）。"""
        return getattr(self.agent, "llm", None)

    def _log(self, msg: str) -> None:
        print(f"[runner_dynamic] {msg}", flush=True)

    # ---- 候选生成（RAG 增强，与 runner.py 一致）----

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
            # 合法性断言：rel_offset 是 (grasp - center) / half_size，
            # 正常应在 [-2, 2] 内；超出说明感知候选点偏离物体，丢弃用中心兜底
            if any(abs(float(v)) > 2.0 for v in off):
                continue
            band = score_band_of(c["score"])
            if self.rag is not None and self.rag.is_failed(entry["task_name"], feat, off):
                continue
            meta.append({"position": [float(x) for x in np.asarray(c["position"]).reshape(3)],
                         "score": float(c["score"]), "rel_offset": off, "score_band": band})
        if self.rag is not None:
            meta = self.rag.rank_candidates(entry["task_name"], feat, meta)
        # 中心优先：rel_offset 越接近 0 越靠前（边缘抓取不稳定）
        meta.sort(key=lambda m: float(np.linalg.norm(m["rel_offset"])))
        center_off = [0.0, 0.0, 0.0]
        if not (self.rag is not None and self.rag.is_failed(entry["task_name"], feat, center_off)):
            # 中心候选始终置顶（除非 RAG 标记为失败）
            if meta and meta[0]["rel_offset"] != center_off:
                meta.insert(0, {"position": [float(x) for x in center], "score": 1.05,
                                "rel_offset": center_off, "score_band": score_band_of(1.05)})
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

    # ---- 规划：observe → select method → 现场绑定参数生成 skill 链 ----

    def _next_steps(self, cond: GoalCond, cand: Dict[str, Any], cfg: Dict[str, Any],
                    env, is_first: bool, body0=None) -> tuple:
        """为一个未满足条件动态生成 skill 序列（含首方法 home 注入）。

        registry 无匹配方法时，调用 SkillCreator 自动生成新方法（LLM 优先，
        规则回退兜底），注册后重试 select。
        """
        entry = self.entry
        # 方法编排：上一个方法声明了 next 且能匹配当前条件时优先用
        method = None
        pref = getattr(self, "_preferred_next", None)
        if pref:
            cand_m = self.registry.get(pref)
            if cand_m is not None and cand_m.can_achieve(cond, entry):
                method = cand_m
            self._preferred_next = None
        if method is None:
            try:
                method = self.registry.select(cond, entry)
            except ValueError:
                self._log(f"无匹配方法 for {cond.describe()}，尝试自动生成…")
                from .skill_creator import SkillCreator
                creator = SkillCreator(self.registry, llm=self._llm_client(),
                                       experience=self.experience,
                                       available_skills=list(self.agent.skills.keys()))
                method = creator.create_for(cond, task=entry["task_name"])
                if method is None:
                    raise
                method = self.registry.select(cond, entry)
        # 记录本方法的 next 偏好，供下轮 select 使用
        self._preferred_next = getattr(method, "next", None)
        ctx = PlanContext(entry, env, cand, cfg, is_first, body0_pos=body0)
        # make_steps 失败（如 forged skill 环境不匹配）→ 排除该方法换下一个，
        # 而不是拿着残链执行或直接崩溃。
        exclude: set = set()
        while True:
            try:
                steps = method.make_steps(ctx, cond)
                if steps:
                    break
                exclude.add(method.name)
                method = self.registry.select(cond, entry, exclude=exclude)
            except (RuntimeError, ValueError) as e:
                print(f"[plan] 方法 {method.name} 生成链失败: {e}，换下一个方法")
                exclude.add(method.name)
                method = self.registry.select(cond, entry, exclude=exclude)
        # 首个方法负责把臂收回 home（open_drawer 已自带；其余按 flavor 补 home/pose_home）
        if is_first and not method.includes_home:
            home_name = "pose_home" if method.flavor == "pose" else "home"
            steps = [{"action": home_name,
                      "params": {"timeout": int(90 * ctx.timeout_scale), **ctx.cmn}}] + steps
        return method, steps

    def plan_full(self, cand: Dict[str, Any], cfg: Dict[str, Any], env=None,
                  objectives: Optional[List[GoalCond]] = None) -> List[Dict[str, Any]]:
        """一次性展开全部未满足条件（给外部顺序驱动的调用方，如视频脚本）。

        动态闭环 run() 的"无回放"版本：只按当前状态做一轮 observe+plan，
        返回完整 skill 链。等价旧 runner._plan_for，但链是条件/方法动态生成的。
        """
        conds = objectives or objectives_from_entry(self.entry, env)
        plan: List[Dict[str, Any]] = []
        for i, cond in enumerate(conds):
            if cond.check(env):
                continue
            _, steps = self._next_steps(cond, cand, cfg, env, is_first=(len(plan) == 0))
            plan += steps
        return plan

    # ---- 执行（逐 skill 分发 + 日志；语义与 ManipulationAgent.execute 一致）----

    def _execute_chunk(self, steps: List[Dict[str, Any]], env,
                       logger: EpisodeLogger, step_offset: int,
                       on_skill=None) -> tuple:
        """执行 step 链，支持声明式恢复（on_fail 跳转）。

        step 可选字段：
          - name: step 标识符，供 on_fail 跳转目标
          - on_fail: 失败时跳转到的 step name；不设则链失败返回

        这允许 impedance_push 失败后自动跳到 spiral_search 等恢复步骤，
        而不是终止整个 attempt。

        on_skill(i, name, result)：每个 skill 执行完回调一次（此刻 env 仍是
        该 skill 的终态，TCP/body 坐标是实时真值），供 IPC 推流/细粒度观测。
        """
        traj: List[Dict[str, Any]] = []
        name_to_idx = {s.get("name"): i for i, s in enumerate(steps) if s.get("name")}
        i = 0
        while i < len(steps):
            act = steps[i]
            name, params = act["action"], act.get("params", {})
            skill = self.agent.skills.get(name)
            logger.skill_start(name, params)
            if skill is None:
                result = {"success": False, "error": f"unknown skill: {name}"}
            else:
                # 标记当前 skill：物理步快照钩子据此记录每步归属，
                # resume 时才能定位"失败 skill 自己的最后执行步"回退点。
                prev_action = getattr(env, "_darwin_action", None)
                env._darwin_action = name
                try:
                    result = skill.execute(env=env, **params)
                except Exception as e:  # noqa: BLE001
                    result = {"success": False, "error": str(e)}
                finally:
                    env._darwin_action = prev_action
            logger.skill_end(name, result)
            traj.append({"action": name, "params": params, "result": result,
                         "step_idx": step_offset + i})
            if on_skill is not None:
                try:
                    on_skill(i, name, result)
                except Exception:
                    pass
            if not result.get("success", False):
                on_fail = act.get("on_fail")
                if on_fail and on_fail in name_to_idx:
                    # 声明式恢复：跳到指定 step，不终止链
                    i = name_to_idx[on_fail]
                    continue
                fail_cat = _classify_failure(result, name)
                return traj, False, fail_cat
            i += 1
        return traj, True, ""

    # ---- 单 attempt 执行体（run() 进程内循环与 IPC sim_worker 共享）----

    def _settle_to_safe_home(self, env, entry):
        """entry 声明 safe_home_qpos 时：reset 后覆盖构型并 hold 收敛 CARTIK 目标。"""
        q = entry.get("safe_home_qpos")
        if not q:
            return
        import mujoco as _mj
        env.mj_data.qpos[:len(q)] = np.asarray(q, float)
        env.mj_data.qvel[:len(q)] = 0.0
        _mj.mj_forward(env.mj_model, env.mj_data)
        env.init_pos[entry["actor"]] = np.array(
            env.get_site_pos(entry["grip_site"]))
        from ..skills.primitives import step_env, p_action
        for _ in range(40):
            cur = env.get_site_pos(entry["grip_site"])
            step_env(env, entry["actor"],
                     p_action(env, entry["grip_site"], cur, gripper=+1, k=2.0))

    def _place_telemetry(self, env) -> Dict[str, Any]:
        """LIBERO place 类目标的几何遥测（供 agent 进程反思定位失败原因）。

        非 libero/非 place 目标返回 {}。所有读取做防御，感知失败不阻断执行。
        """
        g = self.entry.get("libero_goal") or {}
        if g.get("kind") != "place":
            return {}
        obj, target_name = g.get("object"), g.get("target")
        predicate = g.get("predicate")
        try:
            bp = np.asarray(env.get_body_pos(obj), float)
            if target_name in env.object_names:
                tp = np.asarray(env.support_point(target_name), float)
            else:
                from .libero_tasks import resolve_site_name
                site_name = resolve_site_name(env.mj_model, target_name)
                if site_name is None:
                    return {}
                tp = np.asarray(env.get_site_pos(site_name), float)
            tel = {
                "body_pos": [float(v) for v in bp],
                "target_pos": [float(v) for v in tp],
                "xy_dist": float(np.linalg.norm(bp[:2] - tp[:2])),
                "z_delta": float(bp[2] - tp[2]),
                "bddl_success": bool(env.check_success()),
            }
            # In 盒级诊断：物体质心相对真实 region site 盒的逐轴状态 + 单谓词真值。
            # 只读、不改行为；用于定位“几何在盒内却 goal_not_reached”的动态真因。
            if predicate == "In" and target_name not in env.object_names:
                try:
                    from .libero_tasks import resolve_site_name
                    sn = resolve_site_name(env.mj_model, target_name)
                    sid = int(env.mj_model.site_name2id(sn))
                    sz = np.asarray(env.mj_model.site_size[sid], float)
                    xm = np.asarray(env.mj_data.site_xmat[sid], float).reshape(3, 3)
                    half = np.abs(xm @ sz)
                    sp = np.asarray(env.get_site_pos(sn), float)
                    dd = bp - sp
                    tel["site_half"] = [float(x) for x in half]
                    tel["site_pos"] = [float(x) for x in sp]
                    tel["in_xyz"] = [bool(abs(dd[0]) < half[0]),
                                     bool(abs(dd[1]) < half[1]),
                                     bool(sp[2] - half[2] - 0.01 < bp[2]
                                          < sp[2] + half[2])]
                    try:
                        tel["in_predicate"] = bool(
                            env.eval_subgoal("In", obj, target_name))
                    except Exception:
                        tel["in_predicate"] = None
                    if not tel["bddl_success"]:
                        print(f"[inbox-diag] {self.entry.get('libero_suite')}:"
                              f"{self.entry.get('libero_task_idx')} In {obj}->"
                              f"{sn} d={np.round(dd,4)} half={np.round(half,4)} "
                              f"in_xyz={tel['in_xyz']} pred={tel['in_predicate']}",
                              flush=True)
                except Exception:
                    pass
            return tel
        except Exception:
            return {}

    def _build_fresh_candidates(self, env):
        """构造 (feat, fresh_candidates(cfg))。

        fresh_candidates 每次调用都重新观察世界（reset 会随机化物体位姿，
        reset 前采的点在执行时已过期）。容器偏心策略的精细化参数全部从 cfg
        读（agent 反思学习后写回 per-env YAML），此处不保留任务魔数。
        """
        entry = self.entry
        if entry.get("mode") == "libero":
            feat = {"shape": "libero_object", "size": [0.02, 0.02, 0.02]}
            # 空腔类物体需要"一指在内、一指在外"偏心插指；名单是形状语义，
            # 与任务无关，任何 env 的同类物体共用同一策略（参数在 cfg 里学）。
            container_hints = ("bowl", "basket", "mug", "cup", "bucket",
                               "caddy", "container", "tray")

            def fresh_candidates(cfg):
                body_name = entry["body"]
                is_container = any(h in body_name.lower()
                                   for h in container_hints)
                res = grasp_candidates_from_env(env, body_name, top_k=8)
                all_cands = res.get("candidates") or []
                cands = []
                max_width = float(cfg.get("grasp_max_width", 0.075))

                if is_container and all_cands:
                    # 容器：GraspNet 给方向（碗心→最佳候选），几何偏心量/插指
                    # 深度由 cfg 参数决定（可学习），居中闭合必夹空腔。
                    b = env.object_bounds(body_name)
                    cx, cy = b["center"]
                    half_y, z_top = b["half_y"], b["z_top"]
                    best = max(all_cands, key=lambda c: float(c.get("score", 0)))
                    gp = np.array(best["position"], float)
                    direction = gp[:2] - np.array([cx, cy])
                    norm = float(np.linalg.norm(direction))
                    if norm < 1e-4:
                        direction = np.array([0.0, -1.0])
                    else:
                        direction = direction / norm
                    ratio = float(cfg.get("grasp_container_offset_ratio", 0.70))
                    z_delta = float(cfg.get("grasp_container_z_delta", 0.030))
                    off = ratio * half_y
                    pos = [float(cx + off * direction[0]),
                           float(cy + off * direction[1]),
                           float(z_top - z_delta)]
                    cands.append({"position": pos, "score": 1.0,
                                  "rel_offset": [0.0, 0.0, 0.0],
                                  "score_band": "high", "source": "graspnet_dir"})
                else:
                    # 实心物体：GraspNet position 的 xy 用，但 z 必须取物体顶面
                    # 附近（GraspNet/几何法回质心=物体几何中心，薄物体中心 z 太
                    # 低，手臂下不去 → contact_stop 让夹爪在物体上方闭合夹空）。
                    b = env.object_bounds(body_name)
                    cx, cy = b["center"][0], b["center"][1]
                    thickness = float(b["z_top"] - b["z_bottom"])
                    grasp_z = float(b["z_top"]) - min(0.012, float(b["z_top"]) * 0.5)
                    # 高而规则的物体（橙汁盒/罐头，实测厚度>0.06）：几何中心候选
                    # 置顶——夹爪对中夹住两侧平面最稳。GraspNet 候选 xy 常偏离
                    # 中心 3-5cm，descend 后指尖在物体外侧闭合夹空气（F≈0），
                    # lift 必判 lift_no_grip（libero_object:9：中心 F=53N 一次
                    # 夹住，偏心 6/6 全夹空）。
                    # 扁平物体（cream_cheese 厚 0.041）除外：对中直降两指尖会对称
                    # 压在盒顶被架住，接触软停高位闭合夹空；GraspNet 的偏心点
                    # （+jit 逐 attempt 换方向）让一指先越沿、指尖滑到盒侧中下
                    # 部再闭合（object:1 实测偏心 19mm/end_z=0.014 成功）。
                    _center_thr = float(os.environ.get("DARWIN_CENTER_THRESH", "0.06"))
                    if thickness >= _center_thr:
                        cands.append({
                            "position": [float(cx), float(cy), grasp_z],
                            "score": 1.0, "rel_offset": [0.0, 0.0, 0.0],
                            "score_band": "high", "source": "bounds_center"})
                    for c in all_cands:
                        w = float(c.get("width", 0.0))
                        if w > max_width:
                            continue
                        pos = c["position"]
                        score = float(c.get("score", 0.5))
                        cands.append({
                            "position": [float(pos[0]), float(pos[1]), grasp_z],
                            "score": score,
                            "rel_offset": [0.0, 0.0, 0.0],
                            "score_band": "high" if score > 0.5 else "mid",
                        })
                if not cands:
                    pt = env.grasp_point(body_name)
                    cands.append({"position": [float(x) for x in pt], "score": 1.0,
                                  "rel_offset": [0.0, 0.0, 0.0], "score_band": "high"})
                return feat, cands
        else:
            feat = object_features(env, entry["body"])

            def fresh_candidates(cfg):
                f = object_features(env, entry["body"])
                return f, list(self._inject_memory(env, f, self._build_candidates(env, f)))

        return feat, fresh_candidates

    def _run_single_attempt(self, env, objectives, *, attempt, local_cfg,
                            logger, fresh_candidates, jit_scale,
                            event_cb=None) -> Dict[str, Any]:
        """执行一次 attempt：reset → 观察 → 逐 chunk 规划执行 → 终态核验。

        纯执行 + 核验，不做反思/参数自适应——调用方各负其责：
        run() 进程内自适应；IPC 模式由 agent 进程分析结果后下发新 cfg。
        event_cb(event_dict)：每个 skill 执行完回调（此刻 TCP/body 是实时
        终态），sim_worker 用它向 agent 进程推 skill_event 流。
        """
        entry = self.entry
        env.reset()
        _apply_bounds(env, entry.get("bounds"))
        self._settle_to_safe_home(env, entry)
        body0 = np.array(env.get_body_pos(entry["body"]), float)

        _, queue = fresh_candidates(local_cfg)
        if not queue:
            return {"ok_all": False, "verified": False, "fail_phase": "no_candidate",
                    "measures": {}, "traj": [], "n_steps": 0, "skill_events": [],
                    "exec_pairs": [], "methods_used": [], "cand": None,
                    "feat": {"shape": "unknown"}, "body0": body0,
                    "telemetry": self._place_telemetry(env)}
        cand = queue.pop(0)
        # 横向 jitter 只在反思显式打开 cfg.jit(>0) 时生效：记忆/偏心精确抓取始终
        # 零抖动；扁平/难夹实心物体靠它逐 attempt 换方向蹭入边缘。默认 cfg.jit=0
        # （已 learned 任务/哨兵）→ jit 恒为零，行为不变。
        if bool(cand.get("from_memory")) or jit_scale <= 0.0:
            jit = np.zeros(3)
        else:
            ang = 2.4 * attempt + 1.9
            jit = jit_scale * np.array([np.sin(ang), np.cos(ang), 0.0])

        logger.attempt_start(attempt + 1, cand)
        if self.verbose:
            print(f"    attempt{attempt + 1}: score={cand['score']:.2f} "
                  f"band={cand['score_band']}{' [memory]' if cand.get('from_memory') else ''}")

        traj, ok_all, fail_phase = [], True, ""
        chunk_count = 0
        exec_pairs: List[tuple] = []
        methods_used_chunk: List[str] = []
        skill_events: List[Dict[str, Any]] = []
        # 已满足的子目标索引：一旦满足就标记 done，后续动作即使物理上"反满足"
        # （如 Close 把隐式 Open 的抽屉关上）也不再重选——避免 Open↔Close 死循环。
        done: set = set()

        def _on_skill(i, name, result):
            ev = {
                "idx": len(traj) + i, "action": name,
                "success": bool(result.get("success", False)),
                "reason": str(result.get("reason") or result.get("error") or ""),
                "steps": int(result.get("steps", 0) or 0),
                "tcp": [float(x) for x in env.get_site_pos(entry["grip_site"])],
                "body": [float(x) for x in env.get_body_pos(entry["body"])],
            }
            skill_events.append(ev)
            if event_cb is not None:
                event_cb(ev)

        while True:
            pending = None
            for idx, cond in enumerate(objectives):  # observe：重新评估未完成条件
                if idx in done:
                    continue
                m = cond.measure(env)
                satisfied = cond.check(env)
                logger.condition(attempt + 1, cond.describe(), satisfied, m)
                if satisfied:
                    done.add(idx)               # 已满足 → 锁定，后续不再重选
                elif pending is None:
                    pending = cond
            if pending is None:               # 全部目标达成
                break
            # 多条件任务：每个 chunk 执行前重新观察抓取点（前序 chunk 会改变世界）
            if chunk_count > 0 and len(objectives) > 1:
                _, queue = fresh_candidates(local_cfg)
                if queue:
                    cand = queue.pop(0)

            # body0 透传规则（偏心抓取补偿的基准位置）：
            # - 主物体第一个 chunk（chunk_count==0）：body0=attempt reset 后的
            #   物体静止位=本次抓取前位，正确，透传。
            # - 主物体后续 chunk（chunk_count>0，BDDL false 触发重做）：物体已被
            #   前序 chunk 放到桌上静止（非夹爪中），body0 是 attempt 初始位已
            #   过时，若透传会把物体已发生的位移误算成抓取偏心，place target
            #   被带偏二十几厘米。传 None 让 methods 现场读 env.get_body_pos
            #   （物体在桌上=抓取前静止位，正确）。
            # - 非主物体（多物体任务轮到第二个）：其位置是现场静止位，传 None。
            # - resume 续跑走 _resume_from_action 独立路径，物体在夹爪里被抬走，
            #   那边传 body0 保住夹取前位（methods.py:273 的原始注释防的就是它）。
            cond_body = getattr(pending, "object", None)
            body0_arg = (body0 if (cond_body == entry.get("body")
                                   and chunk_count == 0) else None)
            method, steps = self._next_steps(pending, cand, local_cfg, env,
                                             is_first=(chunk_count == 0),
                                             body0=body0_arg)
            # 抓取点 jitter（drawer 任务整体不加，保持与已验证策略一致）
            # libero 抓放链的抓取步 action 名是 ik_servo（above/descend 共用同一
            # grasp 点），同样横向偏移 → 扁平物体可逐方向蹭入边缘。
            if entry["mode"] != "drawer":
                for p in steps:
                    if p["action"] in ("move_above", "descend", "ik_servo") \
                            and "point" in p["params"]:
                        p["params"]["point"] = (
                            np.array(p["params"]["point"]) + jit).tolist()
            logger.plan(attempt + 1, pending.describe(), method.name, steps,
                        measures=pending.measure(env))
            exec_pairs.append((method.name, pending))
            methods_used_chunk.append(method.name)
            self._last_cond = pending

            chunk, ok_chunk, fail_cat = self._execute_chunk(
                steps, env, logger, len(traj), on_skill=_on_skill)
            traj += chunk
            chunk_count += 1
            if not ok_chunk:
                if fail_cat == "episode_terminated":
                    # episode 步数耗尽：链中断但目标可能已达成，不判失败，
                    # 跳出循环让终态核验决定成功与否。
                    break
                ok_all = False
                fail_phase = fail_cat or (chunk[-1]["action"] if chunk else "unknown")
                logger.note("chunk_failed", method=method.name,
                            goal=pending.describe(),
                            reason=chunk[-1]["result"].get("reason")
                            if chunk else "")
                break
            # 单条件任务：一个 chunk 即完整流程，执行后直接进终态核验，
            # 避免在坏状态上重复执行；多条件任务回到 while 顶部重新 observe。
            if len(objectives) == 1:
                break

        measures = {c.describe(): c.measure(env) for c in objectives}
        # 终态核验只检查显式目标（implicit 子目标如隐式 Open 是手段而非目标，
        # Close 后抽屉关上会让隐式 Open 物理上不满足，但这不影响任务成功）。
        verified = ok_all and all(
            c.check(env) for c in objectives if not getattr(c, "implicit", False))
        if ok_all and not verified:
            # 链全部执行成功但终态不达标：单独分类，避免与 skill 级失败混淆
            fail_phase = "goal_not_reached"
        failed_action = None
        if not ok_all and traj:
            failed_action = traj[-1]["action"]
        return {"ok_all": ok_all, "verified": verified, "fail_phase": fail_phase,
                "failed_action": failed_action,
                "method_name": exec_pairs[-1][0] if exec_pairs else None,
                "measures": measures, "traj": traj,
                "n_steps": sum(int(t.get("result", {}).get("steps", 0) or 0)
                               for t in traj),
                "skill_events": skill_events, "exec_pairs": exec_pairs,
                "methods_used": methods_used_chunk, "cand": cand,
                "feat": cand and {"shape": "libero_object",
                                  "size": [0.02, 0.02, 0.02]}
                        if entry.get("mode") == "libero" else
                        object_features(env, entry["body"]),
                "body0": body0, "telemetry": self._place_telemetry(env)}

    def _resume_from_action(self, env, objectives, *, method_name: str,
                            failed_action: str, local_cfg, logger, cand,
                            event_cb=None, resume_seq: int = 1,
                            body0=None) -> Dict[str, Any]:
        """失败回退后原位续跑（不 reset、不重新观察世界）。

        物理状态已由调用方恢复到失败前 N 步；这里用**新 cfg** 让同一个方法
        重新 make_steps（放置 target 现场重绑），然后从 failed_action 开始
        执行到链尾——已成功的前缀（descend/close/lift/carry）不重跑，因为
        恢复的世界里物体仍在夹爪中。仅支持单条件任务链。
        """
        entry = self.entry
        if len(objectives) != 1:
            return {"ok_all": False, "verified": False,
                    "fail_phase": "resume_unsupported",
                    "failed_action": failed_action, "method_name": method_name,
                    "measures": {}, "traj": [], "n_steps": 0,
                    "skill_events": [], "exec_pairs": [], "methods_used": [],
                    "cand": cand, "feat": None,
                    "telemetry": self._place_telemetry(env)}

        # observe：恢复后重新评估，找仍未满足的那一个条件
        pending = None
        for cond in objectives:
            satisfied = cond.check(env)
            if not satisfied and pending is None:
                pending = cond
        if pending is None:
            # 回退后状态竟已满足（物理松弛恰好落位）：直接核验成功
            measures = {c.describe(): c.measure(env) for c in objectives}
            return {"ok_all": True, "verified": True, "fail_phase": "",
                    "failed_action": None, "method_name": method_name,
                    "measures": measures, "traj": [], "n_steps": 0,
                    "skill_events": [], "exec_pairs": [], "methods_used": [],
                    "cand": cand, "feat": None,
                    "telemetry": self._place_telemetry(env)}

        method = self.registry.get(method_name)
        ctx = PlanContext(entry, env, cand, local_cfg, False, body0_pos=body0)
        steps = method.make_steps(ctx, pending)
        start_idx = next((i for i, s in enumerate(steps)
                          if s["action"] == failed_action), None)
        if start_idx is None:
            return {"ok_all": False, "verified": False,
                    "fail_phase": "resume_action_gone",
                    "failed_action": failed_action, "method_name": method_name,
                    "measures": {}, "traj": [], "n_steps": 0,
                    "skill_events": [], "exec_pairs": [], "methods_used": [],
                    "cand": cand, "feat": None,
                    "telemetry": self._place_telemetry(env)}

        sub_steps = steps[start_idx:]
        logger.plan(resume_seq, pending.describe(),
                    f"{method.name}(resume#{resume_seq})", sub_steps,
                    measures=pending.measure(env))
        if self.verbose:
            print(f"    resume#{resume_seq}: 从 {failed_action} 续跑 "
                  f"（剩余 {len(sub_steps)} 个 skill，新参数）")

        traj, ok_all, fail_phase = [], True, ""
        skill_events: List[Dict[str, Any]] = []

        def _on_skill(i, name, result):
            ev = {
                "idx": i, "action": name,
                "success": bool(result.get("success", False)),
                "reason": str(result.get("reason") or result.get("error") or ""),
                "steps": int(result.get("steps", 0) or 0),
                "tcp": [float(x) for x in env.get_site_pos(entry["grip_site"])],
                "body": [float(x) for x in env.get_body_pos(entry["body"])],
            }
            skill_events.append(ev)
            if event_cb is not None:
                event_cb(ev)

        chunk, ok_chunk, fail_cat = self._execute_chunk(
            sub_steps, env, logger, 0, on_skill=_on_skill)
        traj = chunk
        if not ok_chunk:
            ok_all = False
            fail_phase = fail_cat or (chunk[-1]["action"] if chunk else "unknown")

        measures = {c.describe(): c.measure(env) for c in objectives}
        # 终态核验只检查显式目标（implicit 子目标如隐式 Open 是手段而非目标，
        # Close 后抽屉关上会让隐式 Open 物理上不满足，但这不影响任务成功）。
        verified = ok_all and all(
            c.check(env) for c in objectives if not getattr(c, "implicit", False))
        if ok_all and not verified:
            fail_phase = "goal_not_reached"
        failed_action2 = traj[-1]["action"] if (not ok_all and traj) else None
        return {"ok_all": ok_all, "verified": verified, "fail_phase": fail_phase,
                "failed_action": failed_action2, "method_name": method_name,
                "measures": measures, "traj": traj,
                "n_steps": sum(int(t.get("result", {}).get("steps", 0) or 0)
                               for t in traj),
                "skill_events": skill_events,
                "exec_pairs": [(method.name, pending)],
                "methods_used": [method.name], "cand": cand,
                "feat": {"shape": "libero_object", "size": [0.02, 0.02, 0.02]}
                        if entry.get("mode") == "libero" else None,
                "resume_seq": resume_seq,
                "telemetry": self._place_telemetry(env)}

    # ---- 主入口：observe → plan(method) → execute 闭环 ----

    def prepare_episode(self, record_dir: Optional[str] = None,
                        initial_cfg: Optional[Dict[str, Any]] = None):
        """构造常驻 episode 执行环境（IPC sim_worker 用）。

        run() 是"一次性 prepare+多 attempt+teardown"；sim_worker 需要 env 只
        建一次、跨多条 IPC run 消息存活，故把 prepare 段单独暴露。attempt
        执行仍走共享的 _run_single_attempt。
        """
        entry = self.entry
        env = _get_env(entry["env_id"], entry["robot"], entry.get("controller"))

        objectives = objectives_from_entry(entry, env)
        # 障碍注入：entry 显式声明优先；LIBERO 自动生成（桌面+夹具+非任务
        # 物体，排除 goal 物体与 In 目标容器，防 place/抓取误中止）。
        obstacles = entry.get("obstacles")
        if obstacles is None and entry.get("mode") == "libero":
            try:
                from ..skills.primitives.collision import auto_obstacles
                obstacles = auto_obstacles(env, entry, objectives)
            except Exception as _e:
                print(f"[runner_dynamic] auto_obstacles 失败(降级无避障): {_e}")
                obstacles = None
        if obstacles:
            env.obstacle_bodies = tuple(obstacles)
            # LIBERO 避障灰度：skeleton 模式 AABB 为近似，先 warn 只记录，
            # 观察误报率后再切 abort（env 属性通道，Profile 不存 str 键）
            env.collision_mode = str(entry.get("collision_mode", "warn"))
            print(f"[runner_dynamic] obstacles({len(obstacles)}): {list(obstacles)}")
            print(f"[runner_dynamic] collision_mode={env.collision_mode}")
        named_joints = [c.joint for c in objectives if isinstance(c, JointAtLeast)]
        logger = EpisodeLogger(entry["task_name"], self.log_dir,
                               every=self.sample_every, actor=entry["actor"],
                               grip_site=entry["grip_site"],
                               bodies=[entry["body"]],
                               named_joints=named_joints,
                               verbose=self.verbose)
        logger.attach(env)
        logger.episode_start(entry, initial_cfg or {})

        env.reset()
        _apply_bounds(env, entry.get("bounds"))
        if self.rag is not None:
            self.rag.tick()
            self.rag.decay()

        recorder = None
        if record_dir:
            recorder = StepRecorder(env, str(Path(record_dir) /
                                             f"{entry['task_name']}_ep{self.episode_idx}.mp4"),
                                    bimanual=entry["bimanual"])
        feat, fresh_candidates = self._build_fresh_candidates(env)
        return _EpisodeHandle(env=env, objectives=objectives, logger=logger,
                              fresh_candidates=fresh_candidates,
                              recorder=recorder, feat=feat)

    def teardown_episode(self, handle, *, success: bool = False, attempts: int = 0,
                         total_steps: int = 0,
                         fail_phases: Optional[List[Dict[str, Any]]] = None) -> None:
        """episode 收尾（录屏/日志/环境关闭）。"""
        if handle.recorder is not None:
            handle.recorder.close()
        handle.logger.episode_end(success, attempts, total_steps,
                                  fail_phases=fail_phases or [],
                                  log=str(handle.logger.path))
        handle.logger.detach(handle.env)
        handle.logger.close()
        close = getattr(handle.env, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        self.episode_idx += 1

    def _record_experience(self, task: str, exec_pairs, *, success: bool,
                           fail_phase: str, log_path: str, cfg: Dict = None) -> None:
        """把本次 attempt 执行过的 (method, cond) 结果写入经验库。

        参考 AFlow experience.json：每个 (method, cond) 记
        before=历史最优分, after=本次分, succeed。失败记录 fail_phase 供
        SkillCreator 下次避免重踩。cfg 记录当时参数，供后续规避相同配置。
        """
        if not exec_pairs:
            return
        after = 1.0 if success else 0.0
        params_hint = {k: round(float(v), 4) if isinstance(v, (int, float)) else v
                       for k, v in (cfg or {}).items()
                       if k in ("hover", "k_descend", "lift_height", "stiffness", "damping")}
        seen = set()
        for mname, cond in exec_pairs:
            key = (mname, cond.kind, cond.describe())
            if key in seen:
                continue
            seen.add(key)
            before = self.experience.best_score(task, mname)
            self.experience.add(task, cond, mname, success=success,
                                before_score=before, after_score=after,
                                fail_phase=fail_phase, log_path=log_path,
                                params_hint=params_hint)

    def _maybe_mutate_success(self, exec_pairs, task: str) -> None:
        """成功后对成功方法做小变异（方法空间搜索，参考 EvoAgentX AFlow）。

        只对有一定成功历史的方法变异，避免污染方法库。
        内置 Python 方法不可变异，跳过（与失败路径不同：成功时无需生成替代）。
        """
        if not exec_pairs:
            return
        mname = exec_pairs[-1][0]
        from .chain_registry import DeclarativeMethod
        method = self.registry.get(mname)
        if not isinstance(method, DeclarativeMethod):
            return
        stats = self.experience.summary().get(task, {}).get(mname, {})
        succ = stats.get("success", 0)
        # 至少成功 2 次才变异，且变异体总数不超过 5 个（防止方法库膨胀）
        if succ < 2:
            return
        mutants = [n for n in self.registry.names() if n.startswith(f"{mname}_mutant")]
        if len(mutants) >= 5:
            return
        try:
            from .skill_creator import SkillCreator
            creator = SkillCreator(self.registry, llm=self._llm_client(),
                                   experience=self.experience,
                                   available_skills=list(self.agent.skills.keys()))
            creator.mutate_existing(mname)
        except Exception as e:
            if self.verbose:
                print(f"[runner_dynamic] 变异跳过: {e}")

    # ---------- 失败驱动的自适应调整 ----------

    def _adapt_cfg(self, cfg: Dict[str, Any], fail_cat: str, streak: int = 1,
                   telemetry: Optional[Dict[str, Any]] = None,
                   failed_action: Optional[str] = None) -> None:
        """根据失败分类自适应调整 cfg（就地修改）。

        规则本体在 agents/reflection.py（IPC agent_learner 共用同一份），
        此处仅薄包装：拿 new_cfg 回填 + verbose 打印。
        """
        from .reflection import adapt_cfg
        new_cfg, deltas, note = adapt_cfg(cfg, fail_cat, streak, telemetry,
                                          failed_action)
        cfg.clear()
        cfg.update(new_cfg)
        if self.verbose and (deltas or note):
            changed = " ".join(f"{k}:{v[0]}→{v[1]}" for k, v in deltas.items())
            print(f"      [adapt] fail={fail_cat}(streak={streak}) {changed}")
            if note:
                print(f"              {note}")

    def _maybe_mutate_failure(self, method_name: str, task: str) -> None:
        """某 method 连续失败 ≥3 次后，生成替代方法。

        - YAML 声明式方法：mutate_existing 做小变异
        - 内置 Python 方法：调用 SkillCreator.create_for 生成 YAML 替代
        """
        try:
            from .skill_creator import SkillCreator
            from .chain_registry import DeclarativeMethod
            creator = SkillCreator(self.registry, llm=self._llm_client(),
                                   experience=self.experience,
                                   available_skills=list(self.agent.skills.keys()))
            method = self.registry.get(method_name)
            if isinstance(method, DeclarativeMethod):
                new_name = creator.mutate_existing(method_name, max_mutations=2)
            else:
                # 内置方法无法变异，生成全新 YAML 替代
                new_method = creator.create_for(self._last_cond, task=task)
                new_name = getattr(new_method, "name", None) if new_method else None
            if new_name:
                # 新生成的方法移到 registry 最前面，下次 attempt 优先尝试
                order = [new_name] + [n for n in self.registry.names() if n != new_name]
                try:
                    self.registry.reorder(order)
                except Exception:
                    pass
            if new_name and self.verbose:
                print(f"      [mutate] {method_name} 连续失败，生成替代: {new_name}")
        except Exception as e:
            if self.verbose:
                print(f"      [mutate] 失败变异跳过: {e}")

    def run(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        entry = self.entry
        # per-env 大 skill YAML 作为参数底座（agent_learner 反思写回的成果）；
        # 调用方显式传入的 cfg 优先级更高，不被 YAML 覆盖。IPC 模式下参数由
        # agent 全量下发，sim_worker 不经过这里。
        try:
            from ..skills.skill_config import SkillConfigStore
            env_label = "libero" if entry.get("mode") == "libero" else "default"
            # per-task YAML：libero_spatial:0 → libero_spatial_0
            task_id = None
            if entry.get("libero_suite"):
                task_id = f"{entry['libero_suite']}_{entry['libero_task_idx']}"
            merged = SkillConfigStore.load(
                "ik_servo", env_label, task=task_id).params
            merged.update(cfg)
            cfg = merged
        except Exception:
            pass
        env = _get_env(entry["env_id"], entry["robot"], entry.get("controller"))

        objectives = objectives_from_entry(entry, env)
        # 障碍注入（与 prepare_episode 同一套：显式声明优先，LIBERO 自动生成）
        obstacles = entry.get("obstacles")
        if obstacles is None and entry.get("mode") == "libero":
            try:
                from ..skills.primitives.collision import auto_obstacles
                obstacles = auto_obstacles(env, entry, objectives)
            except Exception as _e:
                print(f"[runner_dynamic] auto_obstacles 失败(降级无避障): {_e}")
                obstacles = None
        if obstacles:
            env.obstacle_bodies = tuple(obstacles)
            # LIBERO 避障灰度：skeleton 模式 AABB 为近似，先 warn 只记录，
            # 观察误报率后再切 abort（env 属性通道，Profile 不存 str 键）
            env.collision_mode = str(entry.get("collision_mode", "warn"))
            print(f"[runner_dynamic] obstacles({len(obstacles)}): {list(obstacles)}")
            print(f"[runner_dynamic] collision_mode={env.collision_mode}")
        named_joints = [c.joint for c in objectives if isinstance(c, JointAtLeast)]
        logger = EpisodeLogger(entry["task_name"], self.log_dir,
                               every=self.sample_every, actor=entry["actor"],
                               grip_site=entry["grip_site"],
                               bodies=[entry["body"]],
                               named_joints=named_joints,
                               verbose=self.verbose)
        logger.attach(env)
        logger.episode_start(entry, cfg)

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

        # 候选点不在 reset 前采样（reset 会随机化物体位姿，reset 前的点执行时
        # 已过期）；fresh_candidates 每次调用重新观察。容器偏心的精细化参数从
        # cfg 读（agent 反思写回 per-env YAML），见 _build_fresh_candidates。
        feat, fresh_candidates = self._build_fresh_candidates(env)
        jit_scale = float(cfg.get("jit", 0.005))
        local_cfg = dict(cfg)  # 本地副本，失败后自适应调整参数
        fail_phases, trajectory, from_memory = [], [], False
        methods_used: List[str] = []
        method_fail_streak: Dict[str, int] = {}  # 每个 method 连续失败次数
        fail_cat_streak: Dict[str, int] = {}     # 每种 fail_cat 连续次数（加速自适应）
        success, attempts, total_steps = False, 0, 0

        try:
            for attempt in range(self.max_attempts):
                ar = self._run_single_attempt(
                    env, objectives, attempt=attempt, local_cfg=local_cfg,
                    logger=logger, fresh_candidates=fresh_candidates,
                    jit_scale=jit_scale)
                traj, cand, body0 = ar["traj"], ar["cand"], ar["body0"]
                total_steps += ar["n_steps"]
                methods_used.extend(ar["methods_used"])
                ok_all, verified = ar["ok_all"], ar["verified"]
                fail_phase, measures = ar["fail_phase"], ar["measures"]
                exec_pairs = ar["exec_pairs"]
                if ar["feat"]:
                    feat = ar["feat"]
                logger.verify(measures, verified)

                if verified:
                    success, attempts = True, attempt + 1
                    trajectory, from_memory = traj, bool(cand.get("from_memory"))
                    self.agent.reflect(entry["task_name"], feat, traj, success=True,
                                       grasp_pt=cand["position"],
                                       control_mode=local_cfg.get("control_mode"),
                                       control_hints=local_cfg.get("control_hints"))
                    self._record_experience(entry["task_name"], exec_pairs,
                                            success=True, fail_phase="",
                                            log_path=str(logger.path), cfg=local_cfg)
                    logger.attempt_end(attempt + 1, True)
                    # 方法空间搜索：对成功方法做小变异，扩大方法库多样性
                    self._maybe_mutate_success(exec_pairs, entry["task_name"])
                    break

                if cand is None:
                    # 感知无候选：不写反思/经验，直接结束本次 episode
                    break

                fail_phases.append({"attempt": attempt + 1,
                                    "phase": traj[-1]["action"] if traj else "unknown",
                                    "fail_cat": fail_phase,
                                    "rel_offset": cand["rel_offset"],
                                    "score_band": cand["score_band"]})
                if self.verbose:
                    print(f"      -> 失败: {fail_phase}")
                self.agent.reflect(entry["task_name"], feat, traj, success=False,
                                   grasp_pt=cand["position"],
                                   control_mode=local_cfg.get("control_mode"),
                                   control_hints=local_cfg.get("control_hints"),
                                   fail_cat=fail_phase)
                self._record_experience(entry["task_name"], exec_pairs,
                                        success=False, fail_phase=fail_phase,
                                        log_path=str(logger.path), cfg=local_cfg)
                logger.attempt_end(attempt + 1, False, fail_phase=fail_phase)
                attempts = attempt + 1
                # 经验反馈1：按成功率重排 method
                try:
                    self.registry.reorder_by_experience(self.experience, entry["task_name"])
                except Exception:
                    pass
                # 经验反馈2：根据 fail_cat 自适应调整参数（连续同种失败加速）
                fail_cat_streak[fail_phase] = fail_cat_streak.get(fail_phase, 0) + 1
                # 不同 fail_cat 出现时重置其他 cat 的 streak
                for k in list(fail_cat_streak.keys()):
                    if k != fail_phase:
                        fail_cat_streak[k] = 0
                self._adapt_cfg(local_cfg, fail_phase, fail_cat_streak[fail_phase],
                                telemetry=ar.get("telemetry"),
                                failed_action=ar.get("failed_action"))
                # 经验反馈3：连续失败触发方法变异（参数变了还不行就换方法结构）
                last_method = exec_pairs[-1][0] if exec_pairs else ""
                if last_method:
                    method_fail_streak[last_method] = method_fail_streak.get(last_method, 0) + 1
                    if method_fail_streak[last_method] >= 3:
                        self._maybe_mutate_failure(last_method, entry["task_name"])
                        method_fail_streak[last_method] = 0  # 重置，避免每轮都变异
                # 物体被推离工作区时额外 reset 一次，避免下一 attempt 在坏状态上规划。
                # 坐标阈值是 robopal 桌面系（桌面 z≈0.46）；LIBERO 坐标系不同
                # （桌面 z≈0.8+），且其 attempt 开头的 reset 必回录制初态，跳过。
                if entry["mode"] != "libero":
                    bp = np.array(env.get_body_pos(entry["body"]), float)
                    out_of_range = bp[2] < 0.40 or abs(bp[0]) > 0.75 or abs(bp[1]) > 0.45
                    pushed = bool(np.linalg.norm((bp - body0)[[0, 1]]) > 0.02)
                    if out_of_range or pushed:
                        env.reset()
                        _apply_bounds(env, entry.get("bounds"))
                        self._settle_to_safe_home(env, entry)
        finally:
            if recorder is not None:
                recorder.close()
            logger.episode_end(success, attempts, total_steps, fail_phases=fail_phases,
                               log=str(logger.path))
            logger.detach(env)
            logger.close()
            self.episode_idx += 1

        score = float(success) - 0.02 * attempts
        return {
            "success": success,
            "trajectory": trajectory,
            "info": {"fail_phases": fail_phases, "from_memory": from_memory,
                     "methods_used": methods_used,
                     "body": entry["body"], "env_id": entry["env_id"],
                     "log": str(logger.path)},
            "metrics": {"score": score, "attempts": attempts,
                        "steps": total_steps, "shape": feat["shape"]},
        }


# import 兼容：替换 runner 时可直接 from .runner_dynamic import EpisodeRunner
EpisodeRunner = DynamicEpisodeRunner

__all__ = ["DynamicEpisodeRunner", "EpisodeRunner", "StepRecorder", "_get_env",
           "GoalCond", "JointAtLeast", "BodyLifted", "BodyNearSite",
           "objectives_from_entry", "select_method", "METHOD_REGISTRY"]
