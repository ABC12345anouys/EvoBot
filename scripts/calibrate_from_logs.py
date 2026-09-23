"""从 logs/episodes/*.jsonl 遥测反推参数推荐（只读，不改任何配置）。

对应"调参不靠盲试"的第一层：能量化的参数直接从历史 episode 数据里算出来。
只读取 episode 日志与（可选）per-task YAML 的 history，输出一份报告：

  A. 每任务总览：episode 数 / 成功率 / 主导 fail_phase / 关键参数当前值
  B. carry_vcap 标定：move_timeout 的 carry 反推所需巡航速度
       v_need = 跨距 / (0.5 × carry timeout)（一半预算巡航，余量给落座）
  C. contact_stop_band 标定：descend 超时失败的 TCP 残余落差 gap 分位数
  D. 抓取诊断：grasp_pt 对物体中心的横向偏移 + lift 接触力 F 分布，
     区分"候选偏心"（感知侧）与"descend 没到位"（参数侧）
  E. 每个有推荐值的任务打印可直接执行的 update_params 片段（不自动执行）

用法：
  python scripts/calibrate_from_logs.py                      # 全部日志
  python scripts/calibrate_from_logs.py --task libero_goal_2 # 只看某任务
  python scripts/calibrate_from_logs.py --out report.md      # 同时落盘
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from darwin.skills.skill_config import PARAM_SPEC
except Exception:  # 无 darwin 环境也能跑：内置一份范围表兜底
    PARAM_SPEC = {}

# 报告涉及的参数及其推荐允许范围（取自 PARAM_SPEC，缺省用内置值）
_RANGE_FALLBACK = {
    "carry_vcap": (0.01, 1.0), "contact_stop_band": (0.02, 0.12),
    "k_descend": (0.5, 15.0), "stop_above": (-0.09, 0.10),
    "jit": (0.0, 0.03), "timeout_scale": (0.5, 3.0),
}


def _range(key):
    if key in PARAM_SPEC:
        return PARAM_SPEC[key][2], PARAM_SPEC[key][3]
    return _RANGE_FALLBACK.get(key, (float("-inf"), float("inf")))


def _clip(key, v):
    lo, hi = _range(key)
    return min(hi, max(lo, v))


def _q(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return xs[i]


def _xy(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def parse_episode(path):
    """流式解析一个 jsonl，返回该 episode 的结构化摘要。"""
    ep = {"cfg": {}, "attempts": {}, "fail_cats": [], "success": None,
          "task": None, "body": None, "path": path}
    cur = 0  # skill_end 无 attempt 字段，按 attempt_start 顺序归组
    plan_by_att = {}
    for line in open(path, encoding="utf-8"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = d.get("ev")
        if ev == "episode_start":
            ep["task"] = d.get("task")
            ep["body"] = d.get("body")
            ep["cfg"] = d.get("cfg") or {}
        elif ev == "attempt_start":
            cur = d.get("attempt", cur)
            ep["attempts"].setdefault(cur, {})["grasp_pt"] = d.get("grasp_pt")
            ep["attempts"].setdefault(cur, {})["bodies"] = d.get("bodies") or {}
        elif ev == "plan":
            cur = d.get("attempt", cur)
            plan_by_att[cur] = d.get("steps") or []
            ep["attempts"].setdefault(cur, {})["plan"] = plan_by_att[cur]
        elif ev == "skill_end":
            ep["attempts"].setdefault(cur, {}).setdefault("skills", []).append(d)
        elif ev == "episode_end":
            ep["success"] = bool(d.get("success"))
            for f in d.get("fail_phases") or []:
                if f.get("kind") == "attempt":
                    ep["fail_cats"].append(f.get("fail_cat", "unknown"))
    return ep


def analyze(ep):
    """从一个 episode 抽出标定所需的事件级数据。"""
    out = {"carries": [], "descends": [], "grasps": []}
    for att, a in ep["attempts"].items():
        plan = {s.get("skill"): s.get("params", {}) for s in a.get("plan", [])}
        grasp = a.get("grasp_pt")
        bodies = a.get("bodies") or {}
        body = ep.get("body")
        bpos = bodies.get(body) if body else None
        # 抓取诊断：grasp_pt 对物体中心的横向偏移（容器偏心抓取时这是有意偏移）
        if grasp and bpos:
            out["grasps"].append({
                "att": att, "offset": _xy(grasp, bpos),
                "grasp_z": grasp[2], "body_z": bpos[2]})
        # descend：残余落差 = 结束 TCP z − 目标点 z
        dpoint = (plan.get("ik_servo") or {}).get("point")
        # 同一 attempt 可能有多个 ik_servo（above/descend），按 plan 顺序取带 body 的那个
        for s in a.get("plan", []):
            if s.get("skill") == "ik_servo" and s.get("params", {}).get("mode") == "descend":
                dpoint = s["params"].get("point")
        for sk in a.get("skills", []):
            if sk.get("skill") == "ik_servo" and dpoint:
                r = sk.get("result") or {}
                end = r.get("end")
                if end:
                    out["descends"].append({
                        "att": att, "ok": bool(sk.get("ok")),
                        "reason": r.get("reason", "?"),
                        "gap": end[2] - dpoint[2], "steps": r.get("steps")})
            elif sk.get("skill") == "carry":
                r = sk.get("result") or {}
                tgt = (plan.get("carry") or {}).get("target")
                if tgt and grasp:
                    out["carries"].append({
                        "att": att, "ok": bool(sk.get("ok")),
                        "reason": r.get("reason", "?"),
                        "dist": _xy(tgt, grasp),
                        "remain": (_xy(tgt, r["end"]) if r.get("end") else None),
                        "steps": r.get("steps"),
                        "timeout": (plan.get("carry") or {}).get("timeout")})
            elif sk.get("skill") == "lift":
                r = sk.get("result") or {}
                for g in out["grasps"]:
                    if g["att"] == att and "F" not in g:
                        g["F"] = r.get("grip_contact_force_n")
                        g["lift_ok"] = bool(sk.get("ok"))
    return out


def recommend(task, eps, events):
    """汇总一个任务的所有 episode，产出推荐与依据。"""
    recs, why = {}, {}
    cfg = eps[-1]["cfg"]  # 最新一次 episode 的生效参数
    carries = [c for ev in events for c in ev["carries"]]
    descends = [d for ev in events for d in ev["descends"]]
    grasps = [g for ev in events for g in ev["grasps"]]

    # B. carry_vcap：move_timeout 先分"卡住"与"慢"，两者药方相反
    to_fails = [c for c in carries if not c["ok"] and "timeout" in c["reason"]]
    if to_fails:
        for c in to_fails:
            c["progress"] = (1.0 - c["remain"] / c["dist"]
                             if c.get("remain") is not None and c["dist"] else None)
        stuck = [c for c in to_fails
                 if c["progress"] is not None and c["progress"] < 0.3]
        slow = [c for c in to_fails
                if c["progress"] is not None and c["progress"] > 0.7]
        why["_carry分类"] = (
            f"carry 超时 {len(to_fails)} 次：卡住(progress<0.3) {len(stuck)}，"
            f"慢(progress>0.7) {len(slow)}，中间 {len(to_fails) - len(stuck) - len(slow)}")
        if slow:
            v_need = [c["dist"] / (0.4 * c["timeout"]) for c in slow
                      if c.get("timeout") and c["dist"]]
            if v_need:
                need = _q(v_need, 0.75)
                cur = cfg.get("carry_vcap", 0.05)
                if need > cur + 1e-6:
                    recs["carry_vcap"] = round(_clip("carry_vcap", need), 3)
                    why["carry_vcap"] = (
                        f"{len(slow)} 次 carry 走到 70%+ 仍超时（真·慢）；"
                        f"所需 v≈跨距/(0.4·timeout) 的 P75={need:.3f} > 当前 {cur}")
        if stuck:
            why["_carry卡住提示"] = (
                f"{len(stuck)} 次 carry 超时但位移 <30%：撞墙/被卡，"
                f"提 vcap 无效，查 min_clearance/coll_pair 与路径")
    # 成功的 carry 给出实测速度参考
    if carries:
        v_real = [c["dist"] / c["steps"] for c in carries
                  if c["ok"] and c.get("steps")]
        if v_real:
            why.setdefault("_参考", "")
            why["_参考"] += (f"成功 carry 实测速度 P50={_q(v_real, 0.5):.4f}m/步 ")

    # C. contact_stop_band：descend 超时残余落差
    d_fails = [d for d in descends if not d["ok"]]
    if d_fails:
        gaps = [d["gap"] for d in d_fails if d["gap"] is not None]
        big = [g for g in gaps if g > cfg.get("contact_stop_band", 0.05)]
        if big:
            need = _q(big, 0.9) + 0.01
            cur = cfg.get("contact_stop_band", 0.05)
            if need > cur + 1e-6:
                recs["contact_stop_band"] = round(_clip("contact_stop_band", need), 3)
                why["contact_stop_band"] = (
                    f"{len(big)}/{len(d_fails)} 次 descend 失败时 TCP 停在目标上方 "
                    f"{_q(big, 0.5):.3f}m（P90={_q(big, 0.9):.3f}），"
                    f"> 豁免带半宽 {cur}：OSC 下降受限触发不了接触判定")
    return cfg, recs, why, {"n_carry_to": len(to_fails) if to_fails else 0,
                            "n_desc_fail": len(d_fails),
                            "n_grasp": len(grasps)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default="logs/episodes", help="episode 日志目录")
    ap.add_argument("--task", help="只看该任务（如 libero_goal_2）")
    ap.add_argument("--out", help="报告同时写入该文件")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.logs, "*.jsonl")))
    if args.task:
        files = [f for f in files if os.path.basename(f).startswith(args.task)]
    if not files:
        sys.exit(f"无匹配日志: {args.logs}" + (f" task={args.task}" if args.task else ""))

    by_task = defaultdict(list)
    for f in files:
        ep = parse_episode(f)
        if ep["task"]:
            by_task[ep["task"]].append(ep)

    L = []
    L.append("# 参数标定报告（只读，来源 logs/episodes/*.jsonl）")
    L.append(f"日志文件 {len(files)} 个，覆盖任务 {len(by_task)} 个\n")

    for task in sorted(by_task):
        eps = by_task[task]
        events = [analyze(ep) for ep in eps]
        cfg, recs, why, stats = recommend(task, eps, events)
        n_ok = sum(1 for ep in eps if ep["success"])
        fails = [c for ep in eps for c in ep["fail_cats"]]
        dom = (statistics.mode(fails) if fails else "-")
        grasps = [g for ev in events for g in ev["grasps"]]
        lift_f = [g.get("F") for g in grasps if g.get("F") is not None]

        L.append(f"## {task}（{n_ok}/{len(eps)} 成功，主导失败 {dom}）")
        cur = {k: cfg.get(k) for k in
               ("carry_vcap", "contact_stop_band", "k_descend", "stop_above",
                "jit", "timeout_scale") if k in cfg}
        L.append(f"- 当前参数: {cur}")
        if lift_f:
            f_low = sum(1 for v in lift_f if v < 1.0)
            L.append(f"- lift 接触力 F: n={len(lift_f)} P50={_q(lift_f, 0.5):.2f}N "
                     f"P10={_q(lift_f, 0.1):.2f}N，F<1N（夹空）占 {f_low}/{len(lift_f)}")
        if grasps:
            offs = [g["offset"] for g in grasps]
            L.append(f"- grasp_pt 横向偏移: P50={_q(offs, 0.5):.4f}m P90={_q(offs, 0.9):.4f}m")
        for k, v in recs.items():
            L.append(f"- **建议 `{k}`: {cfg.get(k)} → {v}**（{why[k]}）")
        for k in ("_carry分类", "_carry卡住提示", "_参考"):
            if k in why and why[k]:
                L.append(f"- {why[k]}")
        if recs:
            L.append("```python")
            L.append("from darwin.skills.skill_config import SkillConfigStore")
            L.append(f's = SkillConfigStore.load("ik_servo", "libero", task="{task}")')
            L.append(f"s.update_params({json.dumps(recs, ensure_ascii=False)})")
            L.append("s.save()")
            L.append("```")
        L.append("")

    report = "\n".join(L)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"[已写入 {args.out}]", file=sys.stderr)


if __name__ == "__main__":
    main()
