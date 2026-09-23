"""综合评估：跑 stack_grasp_red 新任务 + 收集经验数据 + 触发 SkillCreator 场景。

输出数据供画图：
- 每次 attempt 的 (attempt, success, fail_phase, steps, method, cond)
- 经验库 summary
- 自动生成方法的 YAML 内容
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/lifd/Public/darwin-bot")

from darwin.benchmarks import get_benchmark
from darwin.agents.runner_dynamic import DynamicEpisodeRunner
from darwin.agents.chain_registry import ChainRegistry, OpenDrawerMethod, METHODS_DIR
from darwin.agents.experience_store import ExperienceStore
from darwin.memory import RAGMemory

OUT = Path("/home/lifd/Public/darwin-bot/logs/eval_data")
OUT.mkdir(parents=True, exist_ok=True)
results = {}

# ========== 场景 A：stack_grasp_red 正常执行（已有 GraspLiftMethod） ==========
print("=" * 60)
print("场景 A: stack_grasp_red 正常执行")
print("=" * 60)
entry = dict(get_benchmark("stack_grasp_red"))
log_dir = tempfile.mkdtemp(prefix="eval_a_logs_")
rag = RAGMemory(root=tempfile.mkdtemp(prefix="eval_a_rag_"))
exp_a = ExperienceStore(root=tempfile.mkdtemp(prefix="eval_a_exp_"))
runner = DynamicEpisodeRunner(entry, rag=rag, max_attempts=3, verbose=True,
                              log_dir=log_dir, sample_every=20, experience=exp_a)
cfg = {"hover": 0.12, "k_descend": 2.0, "lift_height": 0.52, "jit": 0.0}
res_a = runner.run(cfg)
results["scene_a_stack_grasp_red"] = {
    "success": res_a["success"],
    "metrics": res_a["metrics"],
    "fail_phases": res_a["info"]["fail_phases"],
    "experience": exp_a.summary(),
}
print(f"场景 A 结果: success={res_a['success']} metrics={res_a['metrics']}")

# ========== 场景 B：精简 registry 触发 SkillCreator（drawer_place 无 place 方法） ==========
print("\n" + "=" * 60)
print("场景 B: 精简 registry 触发 SkillCreator 自动生成")
print("=" * 60)
reg = ChainRegistry()
reg.register(OpenDrawerMethod())
exp_b = ExperienceStore(root=tempfile.mkdtemp(prefix="eval_b_exp_"))

entry_b = dict(get_benchmark("drawer_place"))
entry_b["safe_home_qpos"] = [-0.614, -0.2586, -0.7121, 2.2855,
                             0.2737, -0.6727, -0.2971]
entry_b["obstacles"] = ["cupboard", "drawer"]

log_dir_b = tempfile.mkdtemp(prefix="eval_b_logs_")
rag_b = RAGMemory(root=tempfile.mkdtemp(prefix="eval_b_rag_"))
runner_b = DynamicEpisodeRunner(entry_b, rag=rag_b, max_attempts=2, verbose=True,
                                log_dir=log_dir_b, sample_every=20,
                                registry=reg, experience=exp_b)

auto_yaml = METHODS_DIR / "auto_body_near_site.yaml"
if auto_yaml.exists():
    auto_yaml.unlink()

res_b = runner_b.run(cfg)
results["scene_b_skill_creator"] = {
    "success": res_b["success"],
    "metrics": res_b["metrics"],
    "fail_phases": res_b["info"]["fail_phases"],
    "registry_after": [m["name"] for m in reg.list_methods()],
    "auto_yaml_generated": auto_yaml.exists(),
    "auto_yaml_content": auto_yaml.read_text() if auto_yaml.exists() else "",
    "experience": exp_b.summary(),
}
print(f"场景 B 结果: success={res_b['success']}")
print(f"  registry: {[m['name'] for m in reg.list_methods()]}")
print(f"  auto_yaml exists: {auto_yaml.exists()}")

# 清理 probe 副产物
if auto_yaml.exists():
    auto_yaml.unlink()

# ========== 保存数据 ==========
out_path = OUT / "eval_results.json"
out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n数据已保存到: {out_path}")
print(json.dumps(results, ensure_ascii=False, indent=2)[:2000])
