"""场景切换评估：peg_in_hole（力控插装）+ pickplace（双臂）。

收集：attempt 级明细 / fail_phase 分类 / 经验库 / 方法选择 / 自动生成情况。
输出供画图。
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/lifd/Public/darwin-bot")

from darwin.benchmarks import get_benchmark
from darwin.agents.runner_dynamic import DynamicEpisodeRunner
from darwin.agents.chain_registry import ChainRegistry, default_registry, METHODS_DIR
from darwin.agents.experience_store import ExperienceStore
from darwin.memory import RAGMemory

OUT = Path("/home/lifd/Public/darwin-bot/logs/eval_data")
OUT.mkdir(parents=True, exist_ok=True)
results = {}

CFG = {"hover": 0.12, "k_descend": 2.0, "lift_height": 0.52, "jit": 0.0}

def run_scene(name: str, entry_key: str, max_attempts: int = 4,
              use_default_registry: bool = True) -> dict:
    entry = dict(get_benchmark(entry_key))
    rag = RAGMemory(root=tempfile.mkdtemp(prefix=f"rag_{name}_"))
    exp = ExperienceStore(root=tempfile.mkdtemp(prefix=f"exp_{name}_"))
    reg = default_registry() if use_default_registry else ChainRegistry()
    log_dir = tempfile.mkdtemp(prefix=f"log_{name}_")

    runner = DynamicEpisodeRunner(entry, rag=rag, max_attempts=max_attempts,
                                  verbose=True, log_dir=log_dir, sample_every=15,
                                  registry=reg, experience=exp)
    res = runner.run(CFG)
    return {
        "entry_key": entry_key,
        "success": res["success"],
        "metrics": res["metrics"],
        "fail_phases": res["info"]["fail_phases"],
        "methods_used": res["info"]["methods_used"],
        "experience": exp.summary(),
        "registry_methods": [m["name"] for m in reg.list_methods()],
        "auto_generated": [m for m in reg.list_methods()
                           if m["name"].startswith("auto_")],
        "mutants": [m for m in reg.list_methods()
                    if "_mutant_" in m["name"]],
    }

# ===== 场景 C: peg_in_hole（力控插装，pose flavor）=====
print("=" * 60)
print("场景 C: peg_in_hole（力控插装 pose 链路）")
print("=" * 60)
results["scene_c_peg_in_hole"] = run_scene("c", "peg_in_hole", max_attempts=4)

# ===== 场景 D: pickplace（双臂 cart 链路）=====
print("\n" + "=" * 60)
print("场景 D: pickplace（双臂 cart 链路）")
print("=" * 60)
results["scene_d_pickplace"] = run_scene("d", "pickplace", max_attempts=3)

# ===== 保存 =====
out_json = OUT / "eval_results_scene2.json"
out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n数据已保存: {out_json}")

# 打印摘要
for scene, d in results.items():
    print(f"\n[{scene}] success={d['success']} metrics={d['metrics']}")
    print(f"  methods_used: {d['methods_used']}")
    print(f"  fail_phases: {[(fp['attempt'], fp.get('phase'), fp.get('fail_cat')) for fp in d['fail_phases']]}")
    print(f"  auto_generated: {[m['name'] for m in d['auto_generated']]}")
    print(f"  mutants: {[m['name'] for m in d['mutants']]}")
