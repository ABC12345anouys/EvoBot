"""[DEPRECATED] 在线自进化 v3：泛化经验记忆。

新代码请使用 scripts/run_evolution.py（DynamicEpisodeRunner + MAP-Elites +
SkillCreator + ExperienceStore 完整自进化闭环）。

本脚本保留仅供参考历史实现。
"""
import os
import sys
import json
import tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DARWIN_GPU", "0"))
if os.environ.get("MUJOCO_GL") == "egl":
    os.environ.pop("DISPLAY", None)
import robopal
from robopal.envs.base import MujocoEnv
MujocoEnv.close = lambda self: self.renderer.close()

from darwin.memory import RAGMemory
from darwin.skills.perception.grasp import (
    object_features as get_object_features,
    rel_offset_of, abs_pos_of, score_band_of,
)
from darwin.utils.recorder import StepRecorder

VIDEO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "videos", "evolution")
# ============================================================
# 物体特征与归一化（已统一到 darwin.skills.perception.grasp）
# ============================================================

# ============================================================
# 泛化 RAG 记忆库（已统一到 darwin.memory.rag，基于 MemoryStore 落盘）
# ============================================================


# ============================================================
# 在线自进化 episode（泛化版）
# ============================================================

def _p_action(env, site, target, gripper=0.0, k=4.0):
    end = env.get_site_pos(site)
    vel = np.clip(k * (np.asarray(target, float) - end), -1.0, 1.0)
    return np.append(vel, 1.0 if gripper > 0 else (-1.0 if gripper < 0 else 0.0))


def run_episode_online(env, rag, body_name="green_block", task_name="pickplace",
                       actor="agent1", grip_site="1_grip_site",
                       mode="full", top_k=8, max_total_attempts=60,
                       max_steps_per_attempt=260, verbose=False, bimanual=True,
                       home_pos=None, record_path=None):
    """在线自进化：失败→分析→泛化沉淀→重试直至成功。

    mode: "full" = 抓+放到 goal；"grasp" = 抓起即成功（用于跨任务泛化验证）
    bimanual: True=双臂环境(step 用 dict)；False=单臂环境(step 用数组)
    home_pos: 重试前回的安全位；None=env.init_pos（单臂收拢姿态贴桌时需显式给悬停位）
    record_path: 给定则用 EGL 离屏渲染录制本集 MP4
    """
    from darwin.skills.perception.grasp import grasp_candidates_from_env

    other = "agent0" if actor == "agent1" else "agent1"

    env.reset()
    env.desired_positions = {ag: env.init_pos[ag].copy() for ag in env.agents}
    if bimanual:
        # 双臂环境专用边界（DualPandaPickAndPlace）；其他环境用默认
        env.pos_max_bound = {"agent0": np.array([0.65, 0.65, 0.65]), "agent1": np.array([0.65, 0.40, 0.65])}
        env.pos_min_bound = {"agent0": np.array([-0.10, -0.30, 0.02]), "agent1": np.array([0.25, -0.10, 0.02])}

    rag.tick()          # 推进失败记忆 TTL 时钟
    rag.decay()         # 清理过期失败记录（替代旧的整批删除）
    recorder = StepRecorder(env, record_path, bimanual=bimanual) if record_path else None

    goal = env.get_site_pos("goal_site") if mode == "full" else None
    home = np.array(home_pos, float) if home_pos is not None else np.array(env.init_pos[actor], float)
    feat = get_object_features(env, body_name)
    n_fail = 0
    candidates, cand_idx = [], 0
    round_jit = 0

    def build_candidates():
        """GraspNet top-K → 附加泛化元数据 → 过滤失败 → 统计排序。"""
        nonlocal candidates, cand_idx
        center = env.get_body_pos(body_name)
        rg = grasp_candidates_from_env(env, body_name, top_k=top_k)
        if not rg.get("success"):
            candidates, cand_idx = [], 0
            return
        cands = rg.get("candidates") or [{"position": rg["position"], "score": rg["score"]}]
        meta = []
        for c in cands:
            off = rel_offset_of(c["position"], center, feat["size"])
            band = score_band_of(c["score"])
            if rag.is_failed(task_name, feat, off):
                continue  # 泛化跳过：同形状+同任务的该归一化位置失败过
            meta.append({"position": c["position"], "score": c["score"],
                         "rel_offset": off, "score_band": band})
        candidates, cand_idx = rag.rank_candidates(task_name, feat, meta), 0

    def inject_memory_candidates():
        """把历史成功经验（相对量）在新位置重建为候选，排在最前。"""
        rec = rag.best_success(task_name, feat)
        if rec is not None:
            center = env.get_body_pos(body_name)
            pos = abs_pos_of(rec["rel_offset"], center, feat["size"])
            candidates.insert(0, {"position": pos.tolist(), "score": rec.get("score", 1.0),
                                  "rel_offset": rec["rel_offset"],
                                  "score_band": rec["score_band"], "from_memory": True})

    build_candidates()
    inject_memory_candidates()

    t_total = 0
    while t_total < max_total_attempts * max_steps_per_attempt:
        if cand_idx >= len(candidates):
            round_jit += 1
            if round_jit > 6:
                # 防死锁：TTL 清理过期失败记录（旧策略是全删该 shape，会误删有效经验）
                rag.decay()
                round_jit = 0
            build_candidates()
            inject_memory_candidates()
            if not candidates:
                for _ in range(20):
                    idle = _p_action(env, grip_site, home, gripper=+1)
                    if bimanual:
                        env.step({actor: idle, other: np.zeros(4)})
                    else:
                        env.step(idle)
                    t_total += 1
                continue
        cand = candidates[cand_idx]
        grasp_pt = np.array(cand["position"], float)
        ang = 2.4 * cand_idx + 1.9 * round_jit
        jit = 0.005 * np.array([np.sin(ang), np.cos(ang), 0.0])
        grasp_target = grasp_pt + jit

        if verbose:
            print(f"    attempt{n_fail+1}: cand{cand_idx} score={cand['score']:.2f} "
                  f"band={cand['score_band']} off={[round(x,2) for x in cand['rel_offset']]}"
                  f"{' [memory]' if cand.get('from_memory') else ''}")

        phase, t2, hold = "home", t_total, 0
        block_z_before = env.get_body_pos(body_name)[2]
        fail_phase, steps = None, 0
        while steps < max_steps_per_attempt:
            steps += 1
            t_total += 1
            end = env.get_site_pos(grip_site)
            if phase == "home":
                act = _p_action(env, grip_site, home, gripper=+1, k=3.0)
                # xy 接近即视为到位（不同机械臂 z 下限不同，z 不做硬判定）；60 步超时兜底
                if np.linalg.norm((end - home)[[0, 1]]) < 0.03 or t_total - t2 > 60:
                    phase, t2 = "above", t_total
            elif phase == "above":
                act = _p_action(env, grip_site, grasp_target + [0, 0, 0.12], gripper=+1)
                if np.linalg.norm((end - grasp_target)[[0, 1]]) < 0.012 and end[2] > grasp_target[2] + 0.09:
                    phase, t2 = "descend", t_total
                elif t_total - t2 > 90:
                    fail_phase = "above_timeout"
            elif phase == "descend":
                cur = env.get_body_pos(body_name)
                tgt = np.array([grasp_target[0], grasp_target[1], min(grasp_target[2], cur[2])]) + jit
                act = _p_action(env, grip_site, tgt, gripper=+1, k=2.0)
                if end[2] - cur[2] < 0.012:
                    phase, t2 = "grasp", t_total
                elif t_total - t2 > 80:
                    fail_phase = "descend_timeout"
            elif phase == "grasp":
                act = _p_action(env, grip_site, end, gripper=-1, k=1.0)
                if t_total - t2 > 25:
                    phase, t2 = "lift", t_total
            elif phase == "lift":
                act = _p_action(env, grip_site, end + [0, 0, 0.1], gripper=-1, k=2.0)
                cur_z = env.get_body_pos(body_name)[2]
                if cur_z > 0.52:
                    if mode == "grasp":
                        # 抓起即成功（泛化验证模式）
                        rag.add_success(task_name, feat, cand["rel_offset"],
                                        cand["score_band"], n_fail + 1)
                        if recorder is not None:
                            recorder.close()
                        return {"success": True, "attempts": n_fail + 1, "steps": t_total,
                                "body": body_name, "grasp_pos": grasp_pt.tolist(),
                                "score": float(cand["score"]),
                                "from_memory": bool(cand.get("from_memory"))}
                    phase, t2 = "move", t_total
                elif t_total - t2 > 30 and cur_z <= block_z_before + 0.006:
                    fail_phase = "lift_no_grip"
                elif t_total - t2 > 80:
                    fail_phase = "lift_timeout"
            elif phase == "move":
                act = _p_action(env, grip_site,
                                [goal[0], goal[1], max(0.60, goal[2] + 0.12)], gripper=-1, k=2.5)
                if np.linalg.norm((end - goal)[[0, 1]]) < 0.012:
                    phase, t2 = "place", t_total
                elif t_total - t2 > 110:
                    fail_phase = "move_timeout"
            elif phase == "place":
                act = _p_action(env, grip_site, goal + [0, 0, 0.0], gripper=-1, k=1.2)
                hold += 1

            if bimanual:
                env.step({actor: act, other: np.zeros(4)})
            else:
                env.step(act)

            if mode == "full":
                d = float(np.linalg.norm(env.get_body_pos(body_name) - env.get_site_pos("goal_site")))
                if d < 0.02:
                    rag.add_success(task_name, feat, cand["rel_offset"],
                                    cand["score_band"], n_fail + 1)
                    if recorder is not None:
                        recorder.close()
                    return {"success": True, "attempts": n_fail + 1, "steps": t_total,
                            "body": body_name, "grasp_pos": grasp_pt.tolist(),
                            "score": float(cand["score"]),
                            "from_memory": bool(cand.get("from_memory"))}
                if hold > 30:
                    fail_phase = "place_unstable"

        if fail_phase is None:
            fail_phase = "timeout"
        n_fail += 1
        rag.add_failure(task_name, feat, cand["rel_offset"], cand["score_band"], fail_phase)
        cand_idx += 1

        bp = env.get_body_pos(body_name)
        if bp[2] < 0.40 or abs(bp[0]) > 0.75 or abs(bp[1]) > 0.45:
            env.reset()
            env.desired_positions = {ag: env.init_pos[ag].copy() for ag in env.agents}
            build_candidates()
            inject_memory_candidates()

    if recorder is not None:
        recorder.close()
    return {"success": False, "attempts": n_fail, "steps": t_total,
            "body": body_name, "fail_phase": "exhausted"}


# ============================================================
# 实验入口
# ============================================================

def exp_main(n_episodes=5, record=False):
    """主实验：PickAndPlace green_block 在线进化。"""
    env = robopal.make("BimanualPickAndPlace-v0", robot="DualPandaPickAndPlace",
                       render_mode=None, control_freq=20)
    rag = RAGMemory()
    print("=" * 62)
    print(f"在线自进化 v3（泛化记忆，MemoryStore 落盘）| 初始记忆: {rag.n_failure} fail / {rag.n_success} succ")
    print("=" * 62)
    results = []
    if record:
        os.makedirs(VIDEO_DIR, exist_ok=True)
    for ep in range(n_episodes):
        rec = os.path.join(VIDEO_DIR, f"pickplace_ep{ep}.mp4") if record else None
        r = run_episode_online(env, rag, body_name="green_block", task_name="pickplace",
                               record_path=rec)
        tag = "OK" if r["success"] else "FAIL"
        mem = " [记忆命中]" if r.get("from_memory") else ""
        vid = f" [video: {os.path.basename(rec)}]" if rec else ""
        print(f"ep{ep}: {tag} attempts={r['attempts']} steps={r['steps']}{mem}{vid}")
        results.append(r)
    n_ok = sum(r["success"] for r in results)
    print(f"\n结果: {n_ok}/{len(results)} | 记忆: {rag.n_failure} fail / {rag.n_success} succ")
    print(f"统计表: {json.dumps(rag.stats, ensure_ascii=False)}")
    env.close()


def exp_transfer():
    """泛化验证：
    Phase A: PickAndPlace green_block 学 2 episodes（积累泛化经验）
    Phase B: MultiCubeStack 抓 red/blue block（不同物体+不同任务+不同位置）
             对比 Phase 0（无记忆）的尝试次数。
    """
    print("=" * 62)
    print("泛化验证：PickAndPlace 经验 → MultiCubeStack 抓取迁移")
    print("=" * 62)

    # Phase 0: 无记忆基线（stack 抓 red_block）——用一次性临时记忆库
    rag0 = RAGMemory(root=os.path.join(tempfile.mkdtemp(prefix="rag_baseline_")))
    env_s = robopal.make("MultiCubeStack-v1", robot="DianaTripleStack",
                         render_mode=None, control_freq=20)
    base = []
    for i in range(2):
        r = run_episode_online(env_s, rag0, body_name="red_block", task_name="any",
                               actor="agent0", grip_site="0_grip_site", mode="grasp", bimanual=False, home_pos=[0.35, 0.0, 0.40])
        print(f"[Phase0 无记忆] red_block ep{i}: attempts={r['attempts']} "
              f"{'OK' if r['success'] else 'FAIL'}")
        base.append(r["attempts"] if r["success"] else 99)
    env_s.close()

    # Phase A: PickAndPlace 学经验
    env_p = robopal.make("BimanualPickAndPlace-v0", robot="DualPandaPickAndPlace",
                         render_mode=None, control_freq=20)
    rag = RAGMemory()
    for i in range(2):
        r = run_episode_online(env_p, rag, body_name="green_block", task_name="pickplace")
        print(f"[PhaseA 学习] pickplace ep{i}: attempts={r['attempts']} "
              f"{'OK' if r['success'] else 'FAIL'} | 记忆 {rag.n_failure}f/{rag.n_success}s")
    env_p.close()

    # Phase B: 迁移到 stack 抓 red / blue（不同物体、不同位置）
    env_s = robopal.make("MultiCubeStack-v1", robot="DianaTripleStack",
                         render_mode=None, control_freq=20)
    trans = []
    for body in ["red_block", "blue_block", "red_block"]:
        r = run_episode_online(env_s, rag, body_name=body, task_name="any",
                               actor="agent0", grip_site="0_grip_site", mode="grasp", bimanual=False, home_pos=[0.35, 0.0, 0.40])
        mem = " [记忆命中]" if r.get("from_memory") else ""
        print(f"[PhaseB 迁移] {body}: attempts={r['attempts']} "
              f"{'OK' if r['success'] else 'FAIL'}{mem}")
        trans.append(r["attempts"] if r["success"] else 99)
    env_s.close()
    print("\n" + "=" * 62)
    print(f"无记忆基线(red_block): 平均 {np.mean(base):.1f} 次尝试")
    print(f"迁移后(red+blue):      平均 {np.mean(trans):.1f} 次尝试")
    print(f"泛化提升: {(1 - np.mean(trans)/np.mean(base))*100:.0f}%")
    print(f"记忆: {rag.n_failure} fail / {rag.n_success} succ")
    print(f"统计表: {json.dumps(rag.stats, ensure_ascii=False)}")
    print("=" * 62)
if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    if "transfer" in args:
        exp_transfer()
    else:
        n = next((int(a) for a in args if a.isdigit()), 5)
        exp_main(n, record="record" in args)
