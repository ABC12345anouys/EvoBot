#!/usr/bin/env python3
"""离线生成 LIBERO 任务定义（形式化规格 + LLM 子目标分解），冻结进仓库。

产物：`darwin/skills/configs/task_specs/<suite>_<idx>.yaml`

设计：BDDL goal 是形式化规格，本脚本把它物化成自描述的任务定义
（谓词表带 id、隐式前置候选、词表、夹具属性、可用技能约束、确定性
参考序），交 LLM 只做**谓词级分解**（排序 + 是否插隐式前置），校验通过
后冻结。运行期 `task_spec.load_or_parse` 只读这些 YAML —— 零 token、
完全确定。

用法：
    export DEEPSEEK_API_KEY=...
    python scripts/gen_task_specs.py                       # 全部 30 个任务
    python scripts/gen_task_specs.py --tasks libero_goal:7 # 指定任务
    python scripts/gen_task_specs.py --no-llm              # 只写确定性规格
    python scripts/gen_task_specs.py --force               # 覆盖已存在的
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_SUITES = ("libero_spatial", "libero_object", "libero_goal")


def _parse_tasks(suites: str, tasks: str) -> list:
    if tasks:
        out = []
        for item in tasks.split(","):
            item = item.strip()
            if not item:
                continue
            suite, _, idx = item.rpartition(":")
            out.append((suite or item, int(idx or 0)))
        return out
    return [(s, i) for s in suites.split(",") if s.strip() for i in range(10)]


def _describe(subgoals) -> str:
    def one(s):
        o, t = s.get("object"), s.get("target")
        pred = s.get("predicate")
        if s.get("implicit"):
            return f"!{pred}({t})"
        return f"{pred}({o},{t})" if o else f"{pred}({t})"
    return " → ".join(one(s) for s in subgoals)


def _same_order(a, b) -> bool:
    def key(seq):
        return [(s.get("predicate"), s.get("object"), s.get("target"),
                 bool(s.get("implicit"))) for s in seq]
    return key(a) == key(b)


def main() -> int:
    ap = argparse.ArgumentParser(description="生成并冻结 LIBERO 任务定义")
    ap.add_argument("--suites", default=",".join(DEFAULT_SUITES),
                    help="逗号分隔的 suite 列表（默认三个 LIBERO suite）")
    ap.add_argument("--tasks", default="",
                    help="显式任务列表，如 libero_goal:7,libero_spatial:0（优先于 --suites）")
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "deepseek-v4-pro"))
    ap.add_argument("--base-url",
                    default=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com"))
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY",
                    help="从哪个环境变量取 key（默认 DEEPSEEK_API_KEY）")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-llm", action="store_true",
                    help="不调 LLM，只写确定性规格（formal + reference_order）")
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 YAML")
    args = ap.parse_args()

    _ensure_path()
    from darwin.agents.task_spec import (TASK_SPECS_DIR, build_formal,
                                         cache_path, decompose_spec, save_cache)
    from darwin.envs.libero_adapter import _task_info
    from darwin.llm.client import LLMClient

    client = None
    if not args.no_llm:
        client = LLMClient(base_url=args.base_url,
                           api_key=os.environ.get(args.api_key_env),
                           model=args.model)
        if not client.available():
            print(f"!! 缺少凭证：{args.api_key_env} 未设置 → 回退确定性规格")
    print(f"输出目录: {TASK_SPECS_DIR}")
    print(f"LLM: {'off' if args.no_llm else f'{args.model} @ {args.base_url}'}\n")

    rows, n_llm, n_diff, n_fail = [], 0, 0, 0
    for suite, idx in _parse_tasks(args.suites, args.tasks):
        env_id = f"{suite}:{idx}"
        cached = cache_path(suite, idx)
        if cached.exists() and not args.force:
            rows.append((env_id, "skip(已存在)", "-", "-"))
            continue
        try:
            info = _task_info(suite, idx)
        except Exception as e:
            rows.append((env_id, f"无此任务({type(e).__name__})", "-", "-"))
            continue
        lang = getattr(info["task"], "language", "")
        spec = decompose_spec(info["bddl"], lang, client,
                              temperature=args.temperature)
        save_cache(suite, idx, spec)
        ok = spec.get("source") == "llm"
        diff = ok and not _same_order(spec["subgoals"], spec["reference_order"])
        if ok:
            n_llm += 1
        if diff:
            n_diff += 1
        if spec.get("llm", {}).get("ok") is False:
            n_fail += 1
        rows.append((env_id, spec.get("source", "?"),
                     "同参考序" if not diff else "**与参考序不同**",
                     _describe(spec["subgoals"])))

    print(f"{'task':<22} {'source':<14} {'vs 参考序':<16} 子目标")
    print("-" * 100)
    for env_id, source, diff, desc in rows:
        print(f"{env_id:<22} {source:<14} {diff:<16} {desc}")

    print(f"\n共 {len(rows)} 个任务：LLM 成功 {n_llm}，与参考序不同 {n_diff}，回退 {n_fail}")
    if n_fail:
        print("回退原因见各 YAML 的 llm.error 字段。")
    return 0


def _ensure_path() -> None:
    lib = os.environ.get("LIBERO_PATH", "/home/lifd/Public/LIBERO")
    if lib not in sys.path:
        sys.path.insert(0, lib)
    os.environ.setdefault("MUJOCO_GL", "egl")


if __name__ == "__main__":
    raise SystemExit(main())
